"""
scripts/holdem/hpo_holdem_e.py — Dedicated Optuna Hyperparameter Optimization for Texas Hold'em Model E.
The Co-Evolutionary Champion.

Features:
1. Targeted Co-Evolutionary Exploration Space (Auto-loaded from screened_bounds_holdem_e.json):
   - scaffold_steps: [60k, 140k]
   - min_shaping_mult: [0.0, 0.05]
   - exploit_value_bonus: [0.05, 0.14]
   - exploit_steal_bonus: [0.03, 0.08]
   - bluff_station_penalty: [0.08, 0.28]
   - icm_equity_weight: [0.15, 0.70]
   - fold_penalty: [-0.025, -0.005]
   - pot_control_bonus: [0.015, 0.040]
   - elimination_penalty: [1.0, 3.5]
   - win_bonus: [1.0, 2.5]
   - learning_rate: [5.0e-5, 1.2e-4] (log-scale)
   - initial_ent_coef: [0.020, 0.035]
   - min_ent_coef: [0.004, 0.009]
   - clip_range: [0.18, 0.24]
2. Full Architecture Support:
   - 158-dimensional observation vector (OBS_DIM_HOLDEM_E)
   - HoldemDecoupledActorCriticPolicy (Decoupled Core Action & Bet Sizing Streams)
   - Multi-Task Auxiliary Belief Head (Villain Archetype, Strength Tier, Fold Probability)
   - HoldemLeaguePool (PFSP Self-Play Sparring against Anchor Model D + Generational Snapshots)
   - HoldemCurriculumRewardCallback (Two-Stage Reward Shaping Decay)
3. Evaluates trials across 30 Common Random Numbers (CRN) tournament sessions.
4. Auto-saves best parameters to configs/holdem/best_params_holdem_e.json and results/holdem/model_e_hpo_report.json.
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

from games.holdem.gym_wrapper import HoldemPokerGymEnv, OBS_DIM_HOLDEM_E
from games.holdem.action_abstraction import (
    A_FOLD, A_CHECK_CALL, A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN,
    STREET_PREFLOP, STREET_FLOP
)
from rl.holdem_decoupled_actor import HoldemDecoupledActorCriticPolicy
from rl.holdem_auxiliary_belief import HoldemAuxiliaryBeliefHead, HoldemAuxiliaryBeliefCallback
from rl.holdem_league import HoldemLeaguePool, HoldemLeagueOpponent
from rl.holdem_curriculum import HoldemCurriculumRewardCallback
from scripts.holdem.train_holdem_e import (
    make_holdem_e_env,
    linear_warmup_cosine_decay,
    HoldemEntropyDecayCallback,
    pin_to_pcores,
)

SCREENED_BOUNDS_PATH = os.path.join(PROJECT_ROOT, "configs/holdem/screened_bounds_holdem_e.json")
DEFAULT_OUTPUT_CONFIG = os.path.join(PROJECT_ROOT, "configs/holdem/best_params_holdem_e.json")
TRIALS_HISTORY_PATH = os.path.join(PROJECT_ROOT, "configs/holdem/hpo_trials_history_e.json")
REPORT_PATH = os.path.join(PROJECT_ROOT, "results/holdem/model_e_hpo_report.json")


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
    base_seed: int = 25_000,
    league_pool: Optional[HoldemLeaguePool] = None,
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
            obs_version="e",
            league_mode=True,
            league_pool=league_pool,
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
    timesteps: int = 250_000,
    n_envs: int = 12,
    seed: int = 42,
    eval_sessions: int = 30,
    bounds: Optional[Dict[str, float]] = None,
    league_pool: Optional[HoldemLeaguePool] = None,
) -> float:
    """Optuna objective function training a candidate Model E architecture."""
    b = bounds or {}

    # Sample within screened bounds
    scaffold_steps = trial.suggest_int(
        "scaffold_steps",
        b.get("scaffold_steps_min", 60_000),
        b.get("scaffold_steps_max", 140_000),
        step=10_000,
    )
    min_shaping_mult = trial.suggest_float(
        "min_shaping_mult",
        b.get("min_shaping_mult_min", 0.0),
        b.get("min_shaping_mult_max", 0.05),
    )
    exploit_val = trial.suggest_float(
        "exploit_value_bonus",
        b.get("exploit_value_bonus_min", 0.05),
        b.get("exploit_value_bonus_max", 0.14),
    )
    exploit_stl = trial.suggest_float(
        "exploit_steal_bonus",
        b.get("exploit_steal_bonus_min", 0.03),
        b.get("exploit_steal_bonus_max", 0.08),
    )
    bluff_pen = trial.suggest_float(
        "bluff_station_penalty",
        b.get("bluff_station_penalty_min", 0.08),
        b.get("bluff_station_penalty_max", 0.28),
    )
    icm_w = trial.suggest_float(
        "icm_equity_weight",
        b.get("icm_equity_weight_min", 0.15),
        b.get("icm_equity_weight_max", 0.70),
    )
    fold_pen = trial.suggest_float(
        "fold_penalty",
        b.get("fold_penalty_min", -0.025),
        b.get("fold_penalty_max", -0.005),
    )
    pot_ctrl = trial.suggest_float(
        "pot_control_bonus",
        b.get("pot_control_bonus_min", 0.015),
        b.get("pot_control_bonus_max", 0.040),
    )
    elim_pen = trial.suggest_float(
        "elimination_penalty",
        b.get("elimination_penalty_min", 1.0),
        b.get("elimination_penalty_max", 3.5),
    )
    win_b = trial.suggest_float(
        "win_bonus",
        b.get("win_bonus_min", 1.0),
        b.get("win_bonus_max", 2.5),
    )
    lr = trial.suggest_float(
        "learning_rate",
        b.get("learning_rate_min", 5.0e-5),
        b.get("learning_rate_max", 1.2e-4),
        log=True,
    )
    initial_ent_coef = trial.suggest_float(
        "initial_ent_coef",
        b.get("initial_ent_coef_min", 0.020),
        b.get("initial_ent_coef_max", 0.035),
    )
    min_ent_coef = trial.suggest_float(
        "min_ent_coef",
        b.get("min_ent_coef_min", 0.004),
        b.get("min_ent_coef_max", 0.009),
    )
    clip_range = trial.suggest_float(
        "clip_range",
        b.get("clip_range_min", 0.18),
        b.get("clip_range_max", 0.24),
    )

    vec_env = SubprocVecEnv([
        make_holdem_e_env(
            i,
            seed=seed + trial.number * 50,
            league_pool=league_pool,
            fold_penalty=fold_pen,
            elimination_penalty=elim_pen,
            win_bonus=win_b,
            icm_equity_weight=icm_w,
            pot_control_bonus=pot_ctrl,
            exploit_value_bonus=exploit_val,
            exploit_steal_bonus=exploit_stl,
            bluff_station_penalty=bluff_pen,
        ) for i in range(n_envs)
    ])

    lr_schedule = linear_warmup_cosine_decay(
        initial_lr=lr,
        min_lr=lr * 0.20,
        warmup_fraction=0.03,
    )

    policy_kwargs = dict(
        features_extractor_class=CardAttentionExtractor,
        features_extractor_kwargs=dict(
            num_cards=7,
            embed_dim=64,
            num_heads=4,
            pooling="mean",
            card_features_dim=64,
            game_features_dim=94,
            slot_features_dim=0,
        ),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    model = MaskablePPO(
        policy=HoldemDecoupledActorCriticPolicy,
        env=vec_env,
        learning_rate=lr_schedule,
        n_steps=512,
        batch_size=256,
        n_epochs=5,
        gamma=0.992,
        gae_lambda=0.92,
        clip_range=clip_range,
        ent_coef=initial_ent_coef,
        policy_kwargs=policy_kwargs,
        seed=seed + trial.number * 50,
        verbose=0,
    )

    belief_head = HoldemAuxiliaryBeliefHead(feature_dim=64 + 94, hidden_dim=128)
    belief_callback = HoldemAuxiliaryBeliefCallback(
        belief_head=belief_head,
        lr=lr * 0.8,
        tier_weight=0.5,
        fold_weight=0.5,
        max_batch_size=2048,
        verbose=0,
    )

    curriculum_callback = HoldemCurriculumRewardCallback(
        scaffold_steps=scaffold_steps,
        total_timesteps=timesteps,
        min_shaping_mult=min_shaping_mult,
        log_freq=50_000,
        verbose=0,
    )

    entropy_callback = HoldemEntropyDecayCallback(
        initial_ent=initial_ent_coef,
        min_ent=min_ent_coef,
        total_timesteps=timesteps,
    )

    callbacks = CallbackList([
        curriculum_callback,
        entropy_callback,
        belief_callback,
    ])

    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)
    train_dur = time.time() - t0
    vec_env.close()

    eval_res = evaluate_trial_model(
        model,
        n_sessions=eval_sessions,
        base_seed=seed + trial.number * 50 + 1000,
        league_pool=league_pool,
    )

    # Multi-objective composite score emphasizing win rate, BB return, and survival
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
        f"  [Trial #{trial.number:02d} Completed] Score: {score:6.2f} | "
        f"Win Rate: {eval_res['win_rate']:5.1f}% | BB Return: {eval_res['bb_return']:+6.1f} | "
        f"Elim Rate: {eval_res['elim_rate']:4.1f}% | VPIP: {eval_res['vpip_rate']:4.1f}% | "
        f"Dur: {train_dur:.1f}s"
    )

    return score


def run_hpo_holdem_e(
    n_trials: int = 25,
    timesteps_per_trial: int = 250_000,
    n_envs: int = 12,
    seed: int = 42,
    eval_sessions: int = 30,
    output_config: str = DEFAULT_OUTPUT_CONFIG,
    report_path: str = REPORT_PATH,
):
    pin_to_pcores(n_envs)
    bounds = load_screened_bounds()

    print("\n" + "=" * 80)
    print("      CARDSHARK-RL: TEXAS HOLD'EM MODEL E OPTUNA HPO STUDY")
    print("=" * 80)
    print(f"  Study Budget:              {n_trials} trials | {timesteps_per_trial:,} steps/trial")
    print(f"  Parallel Workers:          {n_envs} P-cores")
    print(f"  Evaluation Sessions:       {eval_sessions} CRN tournament sessions")
    print(f"  Screened Bounds Applied:   {len(bounds)} constraints loaded")
    print("=" * 80 + "\n")

    league_pool = HoldemLeaguePool(
        league_dir="models/holdem/league",
        anchor_model_path="models/holdem/model_holdem_d_best.zip",
    )

    sampler = TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(
        study_name="holdem_model_e_hpo",
        direction="maximize",
        sampler=sampler,
    )

    # Seed with top empirical discoveries from boundary search
    # Discovery 1: Probe #07 Champion (High Exploit & Steal — 64.0% Win Rate, +201.8 BB Return, +84.32 Score)
    study.enqueue_trial({
        "scaffold_steps": 120_000,
        "min_shaping_mult": 0.0,
        "exploit_value_bonus": 0.12,
        "exploit_steal_bonus": 0.07,
        "bluff_station_penalty": 0.08,
        "icm_equity_weight": 0.70,
        "fold_penalty": -0.015,
        "pot_control_bonus": 0.025,
        "elimination_penalty": 3.14,
        "win_bonus": 1.50,
        "learning_rate": 7.5e-5,
        "initial_ent_coef": 0.025,
        "min_ent_coef": 0.006,
        "clip_range": 0.20,
    })
    # Discovery 2: Probe #08 Runner-Up (Heavy Bluff Station Penalty — 52.0% Win Rate, +155.3 BB Return, +65.68 Score)
    study.enqueue_trial({
        "scaffold_steps": 120_000,
        "min_shaping_mult": 0.0,
        "exploit_value_bonus": 0.05,
        "exploit_steal_bonus": 0.03,
        "bluff_station_penalty": 0.25,
        "icm_equity_weight": 0.70,
        "fold_penalty": -0.015,
        "pot_control_bonus": 0.025,
        "elimination_penalty": 3.14,
        "win_bonus": 1.50,
        "learning_rate": 7.5e-5,
        "initial_ent_coef": 0.025,
        "min_ent_coef": 0.006,
        "clip_range": 0.20,
    })
    # Discovery 3: Probe #02 3rd Place (Fast Curriculum Anneal — 48.0% Win Rate, +143.8 BB Return, +61.23 Score)
    study.enqueue_trial({
        "scaffold_steps": 70_000,
        "min_shaping_mult": 0.0,
        "exploit_value_bonus": 0.07,
        "exploit_steal_bonus": 0.04,
        "bluff_station_penalty": 0.14,
        "icm_equity_weight": 0.70,
        "fold_penalty": -0.015,
        "pot_control_bonus": 0.025,
        "elimination_penalty": 3.14,
        "win_bonus": 1.50,
        "learning_rate": 7.5e-5,
        "initial_ent_coef": 0.025,
        "min_ent_coef": 0.006,
        "clip_range": 0.20,
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
            os.makedirs(os.path.dirname(TRIALS_HISTORY_PATH), exist_ok=True)
            with open(TRIALS_HISTORY_PATH, "w") as f:
                json.dump(trials_history, f, indent=2)
        except Exception:
            pass

        if trial_inst.value is not None and trial_inst.value > best_score:
            best_score = trial_inst.value
            try:
                os.makedirs(os.path.dirname(output_config), exist_ok=True)
                with open(output_config, "w") as f:
                    json.dump(trial_inst.params, f, indent=2)
            except Exception:
                pass

    t0 = time.time()
    study.optimize(
        lambda t: objective(
            t,
            timesteps=timesteps_per_trial,
            n_envs=n_envs,
            seed=seed,
            eval_sessions=eval_sessions,
            bounds=bounds,
            league_pool=league_pool,
        ),
        n_trials=n_trials,
        callbacks=[trial_callback],
    )
    total_dur_mins = (time.time() - t0) / 60.0

    best_trial = study.best_trial
    print("\n" + "=" * 80)
    print("      CARDSHARK-RL: MODEL E HPO STUDY COMPLETE")
    print("=" * 80)
    print(f"  Total Duration:            {total_dur_mins:.2f} mins")
    print(f"  Best Trial Number:         #{best_trial.number}")
    print(f"  Best Composite Score:      {best_trial.value:.2f}")
    print(f"  Win Rate:                  {best_trial.user_attrs.get('win_rate', 0.0):.1f}%")
    print(f"  BB Return:                 {best_trial.user_attrs.get('bb_return', 0.0):+.1f}")
    print(f"  Elimination Rate:          {best_trial.user_attrs.get('elim_rate', 0.0):.1f}%")
    print("=" * 80 + "\n")

    os.makedirs(os.path.dirname(output_config), exist_ok=True)
    with open(output_config, "w") as f:
        json.dump(best_trial.params, f, indent=2)
    print(f"  [Saved Best Config] -> {output_config}")

    report = {
        "study_name": study.study_name,
        "n_trials": n_trials,
        "timesteps_per_trial": timesteps_per_trial,
        "duration_minutes": total_dur_mins,
        "best_trial": {
            "trial_number": best_trial.number,
            "score": best_trial.value,
            "metrics": best_trial.user_attrs,
            "params": best_trial.params,
        },
        "screened_bounds_applied": bounds,
        "all_trials": [
            {
                "trial": t.number,
                "score": t.value,
                "params": t.params,
                "attrs": t.user_attrs,
            }
            for t in study.trials
        ],
    }
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  [Saved HPO Report] -> {report_path}")

    return best_trial.params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna HPO for Texas Hold'em Model E")
    parser.add_argument("--trials", type=int, default=25, help="Number of trials")
    parser.add_argument("--timesteps", type=int, default=250_000, help="Timesteps per trial")
    parser.add_argument("--n-envs", type=int, default=12, help="Number of worker envs")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--eval-sessions", type=int, default=30, help="Evaluation sessions per trial")

    args = parser.parse_args()
    run_hpo_holdem_e(
        n_trials=args.trials,
        timesteps_per_trial=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        eval_sessions=args.eval_sessions,
    )
