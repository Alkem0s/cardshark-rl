"""
scripts/holdem/hpo_holdem_c.py — Dedicated Optuna Hyperparameter Optimization for Texas Hold'em Model C.

Targets:
1. Model C GTO & ICM Hyperparameter Exploration:
   - fold_penalty: [-0.02, 0.005]
   - elimination_penalty: [1.0, 10.0] (ICM survival incentive)
   - win_bonus: [0.5, 5.0] (championship incentive)
   - icm_equity_weight: [0.1, 1.0] (dollar equity reward shaping)
   - pot_control_bonus: [0.01, 0.05] (checking medium hands on flop/turn)
   - learning_rate: [1.0e-4, 3.0e-4]
   - initial_ent_coef: [0.02, 0.05]
   - min_ent_coef: [0.003, 0.01]
   - clip_range: [0.15, 0.25]
   - min_lr_ratio: [0.20, 0.35]
2. P-Core CPU Affinity Pinning (12 parallel envs pinned to P-cores 0-11).
3. Evaluates trials across 30 Common Random Numbers (CRN) tournament sessions:
   - Evaluates win rate, average BB return, elimination rate, and preflop VPIP.
4. Saves best parameters to configs/holdem/best_params_holdem_c.json.
"""

from __future__ import annotations
import os
import sys

# Prevent OpenMP / Intel MKL library collision under SubprocVecEnv
os.environ["MKL_THREADING_LAYER"] = "GNU"

import json
import argparse
import time
from typing import Dict, Any, List

import numpy as np
import torch
torch.set_num_threads(4)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import optuna
from optuna.samplers import TPESampler
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from core.neural.card_attention import CardAttentionExtractor
import core.neural.card_attention
sys.modules["card_attention"] = core.neural.card_attention

from games.holdem.gym_wrapper import HoldemPokerGymEnv, OBS_DIM_HOLDEM_C
from games.holdem.action_abstraction import (
    A_FOLD, A_CHECK_CALL, A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN,
    STREET_PREFLOP, STREET_FLOP
)
from scripts.holdem.train_holdem_c import (
    make_holdem_c_env,
    cosine_warm_restart_schedule,
    EntropyDecayCallback,
    pin_to_pcores,
)


def evaluate_trial_model(
    model: MaskablePPO,
    n_sessions: int = 30,
    base_seed: int = 10_000,
) -> Dict[str, float]:
    """Runs CRN evaluation across tournament sessions against mixed league opponents."""
    wins = 0
    total_hands = 0
    vpip_count = 0
    eliminations = 0
    total_chips_delta = 0

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
            obs_version="c",
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

            if env.env.street == STREET_PREFLOP:
                if action in (A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN) or (
                    action == A_CHECK_CALL and env.env.seats[0].total_invested > env.env.big_blind
                ):
                    vpip_count += 1
                total_hands += 1

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

    return {
        "win_rate": win_rate,
        "bb_return": bb_return,
        "elim_rate": elim_rate,
        "vpip_rate": vpip_rate,
    }


def load_screened_bounds(filepath: str = "configs/holdem/screened_bounds_holdem_c.json") -> Optional[dict]:
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def objective(
    trial: optuna.Trial,
    timesteps: int = 250_000,
    n_envs: int = 12,
    seed: int = 42,
    eval_sessions: int = 30,
) -> float:
    bounds = load_screened_bounds() or {}

    # 1. GTO & ICM Tournament Incentives (using screened bounds if available)
    fold_penalty = trial.suggest_float(
        "fold_penalty",
        bounds.get("fold_penalty_min", -0.02),
        bounds.get("fold_penalty_max", 0.005),
    )
    elimination_penalty = trial.suggest_float(
        "elimination_penalty",
        bounds.get("elimination_penalty_min", 1.0),
        bounds.get("elimination_penalty_max", 10.0),
    )
    win_bonus = trial.suggest_float(
        "win_bonus",
        bounds.get("win_bonus_min", 0.5),
        bounds.get("win_bonus_max", 5.0),
    )
    icm_equity_weight = trial.suggest_float(
        "icm_equity_weight",
        bounds.get("icm_equity_weight_min", 0.1),
        bounds.get("icm_equity_weight_max", 1.0),
    )
    pot_control_bonus = trial.suggest_float(
        "pot_control_bonus",
        bounds.get("pot_control_bonus_min", 0.01),
        bounds.get("pot_control_bonus_max", 0.05),
    )

    # 2. PPO Optimization Dynamics
    lr = trial.suggest_float(
        "learning_rate",
        bounds.get("learning_rate_min", 1.0e-4),
        bounds.get("learning_rate_max", 3.0e-4),
        log=True,
    )
    initial_ent_coef = trial.suggest_float(
        "initial_ent_coef",
        bounds.get("initial_ent_coef_min", 0.02),
        bounds.get("initial_ent_coef_max", 0.05),
    )
    min_ent_coef = trial.suggest_float("min_ent_coef", 0.003, 0.01)
    clip_range = trial.suggest_float(
        "clip_range",
        bounds.get("clip_range_min", 0.15),
        bounds.get("clip_range_max", 0.25),
    )
    min_lr_ratio = trial.suggest_float("min_lr_ratio", 0.20, 0.35)
    min_lr = lr * min_lr_ratio

    # 3. Model C Architecture Configuration
    embed_dim = 32
    num_heads = 4
    pooling = "mean"
    card_features_dim = 32
    game_features_dim = 77
    net_arch = [256, 256]
    n_steps = 256
    batch_size = 256
    n_epochs = 5
    gamma = 0.992
    gae_lambda = 0.92
    n_restart_cycles = 6

    vec_env = SubprocVecEnv([
        make_holdem_c_env(
            i,
            seed=seed,
            fold_penalty=fold_penalty,
            elimination_penalty=elimination_penalty,
            win_bonus=win_bonus,
            icm_equity_weight=icm_equity_weight,
            pot_control_bonus=pot_control_bonus,
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
        policy=MaskableActorCriticPolicy,
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
        seed=seed,
        verbose=0,
    )

    callbacks = [
        EntropyDecayCallback(
            initial_ent=initial_ent_coef,
            min_ent=min_ent_coef,
            total_timesteps=timesteps,
        )
    ]

    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)
    train_dur = time.time() - t0
    vec_env.close()

    eval_res = evaluate_trial_model(model, n_sessions=eval_sessions, base_seed=seed + 500)

    score = eval_res["win_rate"] + 0.1 * eval_res["bb_return"] - 0.05 * (eval_res["elim_rate"] - 50.0)

    trial.set_user_attr("win_rate", eval_res["win_rate"])
    trial.set_user_attr("bb_return", eval_res["bb_return"])
    trial.set_user_attr("elim_rate", eval_res["elim_rate"])
    trial.set_user_attr("vpip_rate", eval_res["vpip_rate"])
    trial.set_user_attr("train_duration_sec", train_dur)

    print(
        f"  [Trial #{trial.number:2d}] Score: {score:6.2f} | "
        f"Win Rate: {eval_res['win_rate']:4.1f}% | BB Ret: {eval_res['bb_return']:+6.1f} | "
        f"Elim: {eval_res['elim_rate']:4.1f}% | VPIP: {eval_res['vpip_rate']:4.1f}% | "
        f"Time: {train_dur:.0f}s"
    )

    return score


def run_hpo_holdem_c(
    n_trials: int = 15,
    timesteps: int = 250_000,
    n_envs: int = 12,
    seed: int = 42,
    study_name: str = "holdem_model_c_hpo",
    output_config: str = "configs/holdem/best_params_holdem_c.json",
    dry_run: bool = False,
):
    """Executes Optuna HPO study for Texas Hold'em Model C."""
    if dry_run:
        print("[HPO Dry Run] Testing single trial setup without running Optuna.")
        return

    pin_to_pcores(n_envs)
    os.makedirs(os.path.dirname(output_config), exist_ok=True)

    print(f"=================================================================")
    print(f"  TEXAS HOLD'EM MODEL C HYPERPARAMETER OPTIMIZATION (OPTUNA)")
    print(f"=================================================================")
    print(f"  Trials: {n_trials} | Steps/Trial: {timesteps:,} | Parallel Envs: {n_envs}")
    print(f"  Observation Space: {OBS_DIM_HOLDEM_C} dims | CPU Cores: 12 P-cores")
    print(f"  Target Output Config: {output_config}")
    print(f"=================================================================\n")

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(study_name=study_name, direction="maximize", sampler=sampler)

    # Initial Warmup Trial (seeded with top discovery from boundary search)
    study.enqueue_trial({
        "fold_penalty": -0.015,
        "elimination_penalty": 3.0,
        "win_bonus": 1.0,
        "icm_equity_weight": 0.5,
        "pot_control_bonus": 0.02,
        "learning_rate": 0.00018,
        "initial_ent_coef": 0.060,
        "min_ent_coef": 0.008,
        "clip_range": 0.20,
        "min_lr_ratio": 0.25,
    })

    trials_history = []
    history_file = "configs/holdem/hpo_trials_history_c.json"
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
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
            with open(history_file, "w") as f:
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
                "game_features_dim": 77,
                "net_arch": [256, 256],
            })
            with open(output_config, "w") as f:
                json.dump(best_p, f, indent=2)
            print(f"  >>> [NEW BEST TRIAL #{trial_inst.number}] Score: {trial_inst.value:.2f} | Updated {output_config} <<<")

    t_start = time.time()
    study.optimize(
        lambda t: objective(t, timesteps=timesteps, n_envs=n_envs, seed=seed),
        n_trials=n_trials,
        n_jobs=1,
        callbacks=[trial_callback],
    )
    total_time = time.time() - t_start

    best_trial = study.best_trial
    print(f"\n=================================================================")
    print(f"  HPO COMPLETE: {n_trials} TRIALS in {total_time/60:.1f} mins")
    print(f"  Best Score: {best_trial.value:.2f} (Trial #{best_trial.number})")
    print(f"  Win Rate: {best_trial.user_attrs.get('win_rate', 0):.1f}%")
    print(f"  BB Return: {best_trial.user_attrs.get('bb_return', 0):+.1f} BB")
    print(f"  Elimination Rate: {best_trial.user_attrs.get('elim_rate', 0):.1f}%")
    print(f"  Preflop VPIP: {best_trial.user_attrs.get('vpip_rate', 0):.1f}%")
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
        "game_features_dim": 77,
        "net_arch": [256, 256],
    })

    with open(output_config, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"  Saved best Model C configuration to: {output_config}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HPO for Texas Hold'em Model C")
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config-out", type=str, default="configs/holdem/best_params_holdem_c.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate HPO arguments without running optimization")
    args = parser.parse_args()

    run_hpo_holdem_c(
        n_trials=args.trials,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        output_config=args.config_out,
        dry_run=args.dry_run,
    )
