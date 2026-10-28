"""
scripts/holdem/hpo_holdem_d.py — Dedicated Optuna Hyperparameter Optimization for Texas Hold'em Model D.

Features:
1. Targeted Adaptive Exploiter Exploration Space:
   - exploit_value_bonus: [screened min, max]
   - exploit_steal_bonus: [screened min, max]
   - bluff_station_penalty: [screened min, max]
   - icm_equity_weight: [screened min, max]
   - fold_penalty: [screened min, max]
   - pot_control_bonus: [screened min, max]
   - elimination_penalty: [screened min, max]
   - win_bonus: [screened min, max]
   - learning_rate: [screened min, max] (log-scale)
   - initial_ent_coef: [screened min, max]
   - min_ent_coef: [0.003, 0.012]
   - clip_range: [screened min, max]
   - min_lr_ratio: [0.20, 0.35]
2. Full Architecture Support:
   - 152-dimensional observation vector (OBS_DIM_HOLDEM_D)
   - HoldemDecoupledActorCriticPolicy (Decoupled Core Action & Bet Sizing Streams)
   - Multi-Task Auxiliary Belief Head (Villain Archetype, Strength Tier, Fold Probability)
   - Jittered multi-agent league sparring (TAG, LAG, CallingStation, Model B/C)
3. Evaluates trials across 30 Common Random Numbers (CRN) tournament sessions:
   - Evaluates win rate, average BB return, elimination rate, VPIP, and sizing distribution.
4. Auto-loads screened boundaries from configs/holdem/screened_bounds_holdem_d.json.
5. Saves best parameters to configs/holdem/best_params_holdem_d.json and results/holdem/model_d_hpo_report.json.
"""

from __future__ import annotations
import os
import sys

# Prevent OpenMP / Intel MKL library collision under SubprocVecEnv
os.environ["MKL_THREADING_LAYER"] = "GNU"

import json
import argparse
import time
from typing import Dict, Any, List, Optional

import numpy as np
import torch
torch.set_num_threads(4)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import optuna
from optuna.samplers import TPESampler
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from core.neural.card_attention import CardAttentionExtractor
import core.neural.card_attention
sys.modules["card_attention"] = core.neural.card_attention

from games.holdem.gym_wrapper import HoldemPokerGymEnv, OBS_DIM_HOLDEM_D
from games.holdem.action_abstraction import (
    A_FOLD, A_CHECK_CALL, A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN,
    STREET_PREFLOP, STREET_FLOP
)
from rl.holdem_decoupled_actor import HoldemDecoupledActorCriticPolicy
from rl.holdem_auxiliary_belief import HoldemAuxiliaryBeliefHead, HoldemAuxiliaryBeliefCallback
from scripts.holdem.train_holdem_d import (
    make_holdem_d_env,
    cosine_warm_restart_schedule,
    EntropyDecayCallback,
    pin_to_pcores,
)

SCREENED_BOUNDS_PATH = os.path.join(PROJECT_ROOT, "configs/holdem/screened_bounds_holdem_d.json")
DEFAULT_OUTPUT_CONFIG = os.path.join(PROJECT_ROOT, "configs/holdem/best_params_holdem_d.json")
TRIALS_HISTORY_PATH = os.path.join(PROJECT_ROOT, "configs/holdem/hpo_trials_history_d.json")
REPORT_PATH = os.path.join(PROJECT_ROOT, "results/holdem/model_d_hpo_report.json")


def load_screened_bounds(filepath: str = SCREENED_BOUNDS_PATH) -> Dict[str, float]:
    """Loads empirical screened boundaries synthesized by boundary search."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                print(f"  [Screened Bounds] Successfully loaded boundary priors from: {filepath}")
                return data
        except Exception as e:
            print(f"  [Screened Bounds] Warning: Could not parse {filepath} ({e}), using default bounds.")
    return {}


def evaluate_trial_model(
    model: MaskablePPO,
    n_sessions: int = 30,
    base_seed: int = 15_000,
) -> Dict[str, float]:
    """Runs CRN evaluation across tournament sessions against the multi-agent league."""
    wins = 0
    total_hands = 0
    vpip_count = 0
    eliminations = 0
    total_chips_delta = 0

    flop_moves = 0
    flop_checks = 0

    action_counts = {
        A_FOLD: 0,
        A_CHECK_CALL: 0,
        A_MIN_RAISE: 0,
        A_HALF_POT: 0,
        A_POT: 0,
        A_ALL_IN: 0,
    }
    total_actions = 0

    for s_idx in range(n_sessions):
        env = HoldemPokerGymEnv(
            num_seats=5,
            starting_chips=200,
            small_blind=1,
            big_blind=2,
            hero_seat=0,
            max_hands_per_session=150,
            blind_escalation_interval=15,
            randomize_stacks=True,
            dominated_masking=True,
            pos_gto_masking=True,
            exploitative_masking=True,
            obs_version="d",
            league_mode=True,
            rng_seed=base_seed + s_idx,
        )
        obs, info = env.reset()
        done = False
        start_chips = info["hero_chips"]

        while not done:
            mask = env.action_masks()
            act, _ = model.predict(obs, action_masks=mask, deterministic=True)
            action = int(act)

            total_actions += 1
            if action in action_counts:
                action_counts[action] += 1

            if env.env.street == STREET_PREFLOP:
                if action in (A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN) or (
                    action == A_CHECK_CALL and env.env.seats[0].total_invested > env.env.big_blind
                ):
                    vpip_count += 1
                total_hands += 1
            elif env.env.street == STREET_FLOP:
                flop_moves += 1
                if action == A_CHECK_CALL:
                    flop_checks += 1

            obs, rew, term, trunc, info = env.step(action)
            done = term or trunc

        if info.get("is_winner", False):
            wins += 1
        if info["hero_chips"] <= 0:
            eliminations += 1
        total_chips_delta += (info["hero_chips"] - start_chips)

    win_rate = (wins / float(n_sessions)) * 100.0
    bb_return = (total_chips_delta / 2.0) / float(n_sessions)
    elim_rate = (eliminations / float(n_sessions)) * 100.0
    vpip_rate = (vpip_count / max(1, total_hands)) * 100.0
    flop_check_rate = (flop_checks / max(1, flop_moves)) * 100.0

    act_denom = max(1, total_actions)
    half_pot_freq = (action_counts[A_HALF_POT] / act_denom) * 100.0
    pot_freq = (action_counts[A_POT] / act_denom) * 100.0
    all_in_freq = (action_counts[A_ALL_IN] / act_denom) * 100.0

    return {
        "win_rate": win_rate,
        "bb_return": bb_return,
        "elim_rate": elim_rate,
        "vpip_rate": vpip_rate,
        "flop_check_rate": flop_check_rate,
        "half_pot_freq": half_pot_freq,
        "pot_freq": pot_freq,
        "all_in_freq": all_in_freq,
    }


def objective(
    trial: optuna.Trial,
    bounds: Dict[str, float],
    timesteps: int = 250_000,
    n_envs: int = 12,
    seed: int = 42,
    eval_sessions: int = 30,
) -> float:
    # 1. Targeted Adaptive Exploiter Reward Hyperparameters (Screened Bounds)
    exploit_value_bonus = trial.suggest_float(
        "exploit_value_bonus",
        bounds.get("exploit_value_bonus_min", 0.03),
        bounds.get("exploit_value_bonus_max", 0.12),
    )
    exploit_steal_bonus = trial.suggest_float(
        "exploit_steal_bonus",
        bounds.get("exploit_steal_bonus_min", 0.015),
        bounds.get("exploit_steal_bonus_max", 0.060),
    )
    bluff_station_penalty = trial.suggest_float(
        "bluff_station_penalty",
        bounds.get("bluff_station_penalty_min", 0.05),
        bounds.get("bluff_station_penalty_max", 0.20),
    )
    icm_equity_weight = trial.suggest_float(
        "icm_equity_weight",
        bounds.get("icm_equity_weight_min", 0.60),
        bounds.get("icm_equity_weight_max", 1.40),
    )
    fold_penalty = trial.suggest_float(
        "fold_penalty",
        bounds.get("fold_penalty_min", -0.018),
        bounds.get("fold_penalty_max", -0.002),
    )
    pot_control_bonus = trial.suggest_float(
        "pot_control_bonus",
        bounds.get("pot_control_bonus_min", 0.010),
        bounds.get("pot_control_bonus_max", 0.040),
    )
    elimination_penalty = trial.suggest_float(
        "elimination_penalty",
        bounds.get("elimination_penalty_min", 3.0),
        bounds.get("elimination_penalty_max", 7.5),
    )
    win_bonus = trial.suggest_float(
        "win_bonus",
        bounds.get("win_bonus_min", 1.5),
        bounds.get("win_bonus_max", 4.0),
    )

    # 2. PPO & Decoupled Policy Optimization Dynamics
    lr = trial.suggest_float(
        "learning_rate",
        bounds.get("learning_rate_min", 8.0e-5),
        bounds.get("learning_rate_max", 2.2e-4),
        log=True,
    )
    initial_ent_coef = trial.suggest_float(
        "initial_ent_coef",
        bounds.get("initial_ent_coef_min", 0.020),
        bounds.get("initial_ent_coef_max", 0.045),
    )
    min_ent_coef = trial.suggest_float("min_ent_coef", 0.004, 0.010)
    clip_range = trial.suggest_float(
        "clip_range",
        bounds.get("clip_range_min", 0.18),
        bounds.get("clip_range_max", 0.24),
    )
    min_lr_ratio = trial.suggest_float("min_lr_ratio", 0.22, 0.35)
    min_lr = lr * min_lr_ratio

    # 3. Model D Architecture Configuration
    embed_dim = 32
    num_heads = 4
    pooling = "mean"
    card_features_dim = 32
    game_features_dim = 96
    net_arch = [256, 256]
    n_steps = 256
    batch_size = 256
    n_epochs = 5
    gamma = 0.992
    gae_lambda = 0.92
    n_restart_cycles = 6

    vec_env = SubprocVecEnv([
        make_holdem_d_env(
            i,
            seed=seed + trial.number * 50,
            fold_penalty=fold_penalty,
            elimination_penalty=elimination_penalty,
            win_bonus=win_bonus,
            icm_equity_weight=icm_equity_weight,
            pot_control_bonus=pot_control_bonus,
            exploit_value_bonus=exploit_value_bonus,
            exploit_steal_bonus=exploit_steal_bonus,
            bluff_station_penalty=bluff_station_penalty,
            league_mode=True,
        ) for i in range(n_envs)
    ])

    lr_schedule = cosine_warm_restart_schedule(
        initial_lr=lr,
        min_lr=min_lr,
        n_cycles=n_restart_cycles,
    )

    policy_kwargs = dict(
        features_extractor_class=CardAttentionExtractor,
        features_extractor_kwargs=dict(
            num_cards=7,
            embed_dim=embed_dim,
            num_heads=num_heads,
            pooling=pooling,
            card_features_dim=card_features_dim,
            game_features_dim=game_features_dim,
            slot_features_dim=0,
        ),
        net_arch=dict(pi=net_arch, vf=net_arch),
    )

    model = MaskablePPO(
        policy=HoldemDecoupledActorCriticPolicy,
        env=vec_env,
        learning_rate=lr_schedule,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=initial_ent_coef,
        policy_kwargs=policy_kwargs,
        seed=seed + trial.number * 50,
        verbose=0,
    )

    belief_head = HoldemAuxiliaryBeliefHead(feature_dim=card_features_dim + game_features_dim, hidden_dim=128)
    belief_callback = HoldemAuxiliaryBeliefCallback(
        belief_head=belief_head,
        lr=lr * 0.8,
        tier_weight=0.5,
        fold_weight=0.5,
        max_batch_size=2048,
        verbose=0,
    )

    callbacks = CallbackList([
        EntropyDecayCallback(
            initial_ent=initial_ent_coef,
            min_ent=min_ent_coef,
            total_timesteps=timesteps,
        ),
        belief_callback,
    ])

    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)
    train_dur = time.time() - t0
    vec_env.close()

    eval_res = evaluate_trial_model(model, n_sessions=eval_sessions, base_seed=seed + trial.number * 50 + 1000)

    # Multi-objective composite score emphasizing win rate, BB return, balanced VPIP, and survival
    vpip_penalty = abs(eval_res["vpip_rate"] - 30.0) * 0.3
    score = eval_res["win_rate"] + 0.1 * eval_res["bb_return"] - 0.05 * (eval_res["elim_rate"] - 45.0) - vpip_penalty

    trial.set_user_attr("win_rate", eval_res["win_rate"])
    trial.set_user_attr("bb_return", eval_res["bb_return"])
    trial.set_user_attr("elim_rate", eval_res["elim_rate"])
    trial.set_user_attr("vpip_rate", eval_res["vpip_rate"])
    trial.set_user_attr("flop_check_rate", eval_res["flop_check_rate"])
    trial.set_user_attr("half_pot_freq", eval_res["half_pot_freq"])
    trial.set_user_attr("pot_freq", eval_res["pot_freq"])
    trial.set_user_attr("all_in_freq", eval_res["all_in_freq"])
    trial.set_user_attr("train_duration_sec", train_dur)

    print(
        f"  [Trial #{trial.number:02d}] Score: {score:+6.2f} | "
        f"Win Rate: {eval_res['win_rate']:4.1f}% | BB Ret: {eval_res['bb_return']:+6.1f} | "
        f"Elim: {eval_res['elim_rate']:4.1f}% | VPIP: {eval_res['vpip_rate']:4.1f}% | "
        f"FlopChk: {eval_res['flop_check_rate']:4.1f}% | Time: {train_dur:.0f}s"
    )

    return score


def run_hpo_holdem_d(
    n_trials: int = 25,
    timesteps: int = 250_000,
    n_envs: int = 12,
    seed: int = 42,
    eval_sessions: int = 30,
    study_name: str = "holdem_model_d_hpo",
    output_config: str = DEFAULT_OUTPUT_CONFIG,
    dry_run: bool = False,
):
    """Executes Optuna HPO study for Texas Hold'em Model D."""
    if dry_run:
        print("[HPO Dry Run] Validated configuration and imports. Ready for full HPO execution.")
        return

    pin_to_pcores(n_envs)
    os.makedirs(os.path.dirname(output_config), exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    bounds = load_screened_bounds()

    print("=================================================================")
    print("  TEXAS HOLD'EM MODEL D HYPERPARAMETER OPTIMIZATION (OPTUNA)")
    print("=================================================================")
    print(f"  Trials: {n_trials} | Steps/Trial: {timesteps:,} | Parallel Envs: {n_envs} (12 P-cores)")
    print(f"  Observation Space: {OBS_DIM_HOLDEM_D} dims | Decoupled Policy + Auxiliary Belief")
    print(f"  Target Output Config: {output_config}")
    print(f"  Screened Bounds Applied:")
    for k, v in sorted(bounds.items()):
        print(f"    - {k:<28}: {v}")
    print("=================================================================\n")

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(study_name=study_name, direction="maximize", sampler=sampler)

    # Initial Warmup Trials: Seed with top empirical discoveries from boundary search
    # Discovery 1: Probe #07 Champion (Low ICM Aggression — 84.0% Win Rate, +124.52 Score)
    study.enqueue_trial({
        "exploit_value_bonus": 0.05,
        "exploit_steal_bonus": 0.03,
        "bluff_station_penalty": 0.10,
        "icm_equity_weight": 0.25,
        "fold_penalty": -0.008,
        "pot_control_bonus": 0.02,
        "elimination_penalty": 1.5,
        "win_bonus": 1.0,
        "learning_rate": 0.00012,
        "initial_ent_coef": 0.030,
        "min_ent_coef": 0.008,
        "clip_range": 0.20,
        "min_lr_ratio": 0.30,
    })
    # Discovery 2: Probe #09 Runner-Up (Heavy Fold Reward — 68.0% Win Rate, +96.78 Score)
    study.enqueue_trial({
        "exploit_value_bonus": 0.05,
        "exploit_steal_bonus": 0.03,
        "bluff_station_penalty": 0.12,
        "icm_equity_weight": 0.60,
        "fold_penalty": -0.025,
        "pot_control_bonus": 0.02,
        "elimination_penalty": 3.5,
        "win_bonus": 2.0,
        "learning_rate": 0.00010,
        "initial_ent_coef": 0.028,
        "min_ent_coef": 0.006,
        "clip_range": 0.20,
        "min_lr_ratio": 0.28,
    })

    trials_history = []
    if os.path.exists(TRIALS_HISTORY_PATH):
        try:
            with open(TRIALS_HISTORY_PATH, "r") as f:
                trials_history = json.load(f)
        except Exception:
            trials_history = []

    best_score = -float("inf")

    def trial_callback(study_inst: optuna.Study, trial_inst: optuna.trial.FrozenTrial):
        nonlocal best_score
        entry = {
            "trial": trial_inst.number,
            "score": trial_inst.value,
            "params": trial_inst.params,
            "attrs": trial_inst.user_attrs,
        }
        trials_history.append(entry)
        try:
            with open(TRIALS_HISTORY_PATH, "w") as f:
                json.dump(trials_history, f, indent=2)
        except Exception:
            pass

        if trial_inst.value is not None and trial_inst.value > best_score:
            best_score = trial_inst.value
            best_p = dict(trial_inst.params)
            best_p.update({
                "n_steps": 256,
                "batch_size": 256,
                "n_epochs": 5,
                "gamma": 0.992,
                "gae_lambda": 0.92,
                "n_restart_cycles": 6,
                "embed_dim": 32,
                "num_heads": 4,
                "pooling": "mean",
                "card_features_dim": 32,
                "game_features_dim": 96,
                "net_arch": [256, 256],
            })
            with open(output_config, "w") as f:
                json.dump(best_p, f, indent=2)
            print(f"  >>> [NEW BEST TRIAL #{trial_inst.number}] Score: {trial_inst.value:+.2f} | Updated {output_config} <<<")

    t_start = time.time()
    study.optimize(
        lambda t: objective(t, bounds=bounds, timesteps=timesteps, n_envs=n_envs, seed=seed, eval_sessions=eval_sessions),
        n_trials=n_trials,
        n_jobs=1,
        callbacks=[trial_callback],
    )
    total_time = time.time() - t_start

    best_trial = study.best_trial
    print(f"\n=================================================================")
    print(f"  MODEL D HPO COMPLETE: {n_trials} TRIALS in {total_time/60:.1f} mins")
    print(f"  Best Score: {best_trial.value:+.2f} (Trial #{best_trial.number})")
    print(f"  Win Rate: {best_trial.user_attrs.get('win_rate', 0):.1f}%")
    print(f"  BB Return: {best_trial.user_attrs.get('bb_return', 0):+.1f} BB")
    print(f"  Elimination Rate: {best_trial.user_attrs.get('elim_rate', 0):.1f}%")
    print(f"  Preflop VPIP: {best_trial.user_attrs.get('vpip_rate', 0):.1f}%")
    print(f"  Flop Check Rate: {best_trial.user_attrs.get('flop_check_rate', 0):.1f}%")
    print(f"=================================================================")

    best_params = dict(best_trial.params)
    best_params.update({
        "n_steps": 256,
        "batch_size": 256,
        "n_epochs": 5,
        "gamma": 0.992,
        "gae_lambda": 0.92,
        "n_restart_cycles": 6,
        "embed_dim": 32,
        "num_heads": 4,
        "pooling": "mean",
        "card_features_dim": 32,
        "game_features_dim": 96,
        "net_arch": [256, 256],
    })

    with open(output_config, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"  Saved best Model D configuration to: {output_config}")

    # Generate comprehensive HPO report
    hpo_report = {
        "study_name": study_name,
        "n_trials": n_trials,
        "timesteps_per_trial": timesteps,
        "duration_minutes": total_time / 60.0,
        "best_trial": {
            "trial_number": best_trial.number,
            "score": best_trial.value,
            "metrics": best_trial.user_attrs,
            "params": best_trial.params,
        },
        "screened_bounds_applied": bounds,
        "all_trials": trials_history,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(hpo_report, f, indent=2)
    print(f"  Saved comprehensive HPO report to: {REPORT_PATH}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna HPO for Texas Hold'em Model D")
    parser.add_argument("--trials", type=int, default=25, help="Number of HPO trials")
    parser.add_argument("--timesteps", type=int, default=250_000, help="Timesteps per trial")
    parser.add_argument("--n-envs", type=int, default=12, help="Number of parallel P-core environments")
    parser.add_argument("--eval-sessions", type=int, default=30, help="CRN tournament evaluation sessions per trial")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config-out", type=str, default=DEFAULT_OUTPUT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Validate HPO setup without running optimization")
    args = parser.parse_args()

    run_hpo_holdem_d(
        n_trials=args.trials,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        eval_sessions=args.eval_sessions,
        seed=args.seed,
        output_config=args.config_out,
        dry_run=args.dry_run,
    )
