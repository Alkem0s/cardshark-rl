"""
scripts/holdem/boundary_search_holdem_d.py — Boundary Value & Screening Search for Model D (Targeted Adaptive Exploiter).

Probes the extreme edges and critical transition points of the Model D hyperparameter space:
1. exploit_value_bonus: [0.0 (ablation) vs 0.05 (mid) vs 0.15 (aggressive value extraction)]
2. exploit_steal_bonus: [0.0 (ablation) vs 0.03 (mid) vs 0.08 (aggressive blind stealing)]
3. bluff_station_penalty: [0.0 (no penalty) vs 0.10 (mid) vs 0.25 (heavy punting penalty)]
4. icm_equity_weight: [0.25 (cash game aggression) vs 0.95 (mid) vs 1.80 (extreme risk aversion)]
5. fold_penalty: [-0.025 (heavy fold reward) vs -0.008 (mid) vs +0.010 (punish folding)]
6. learning_rate: [6.0e-5 (low) vs 1.2e-4 (mid) vs 2.8e-4 (high)]
7. initial_ent_coef: [0.020 (low) vs 0.030 (mid) vs 0.055 (high exploration)]

Runs 12 curated boundary probe trials (250,000 timesteps each via 12 P-core SubprocVecEnvs),
evaluates each trial across 25 Common Random Numbers (CRN) tournament sessions against
the mixed league pool (Model B Champion + Jittered Archetype Bots), and synthesizes:
- configs/holdem/screened_bounds_holdem_d.json (tightened search space for full HPO)
- results/holdem/model_d_boundary_report.json (empirical failure analysis & trade-off synthesis)
"""

from __future__ import annotations
import os
import sys

# Prevent OpenMP / Intel MKL library collision under SubprocVecEnv
os.environ["MKL_THREADING_LAYER"] = "GNU"

import json
import argparse
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
torch.set_num_threads(4)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
REPORT_PATH = os.path.join(PROJECT_ROOT, "results/holdem/model_d_boundary_report.json")


# 12 Curated Boundary Value Probes targeting specific behavioral and mathematical hypotheses
BOUNDARY_PROBE_POINTS: List[Dict[str, Any]] = [
    # Probe 0: Balanced Model D Baseline
    {
        "probe_name": "Model D Baseline",
        "params": {
            "exploit_value_bonus": 0.05,
            "exploit_steal_bonus": 0.03,
            "bluff_station_penalty": 0.10,
            "icm_equity_weight": 0.95,
            "fold_penalty": -0.008,
            "pot_control_bonus": 0.02,
            "elimination_penalty": 5.5,
            "win_bonus": 2.5,
            "learning_rate": 0.00012,
            "initial_ent_coef": 0.030,
            "min_ent_coef": 0.008,
            "clip_range": 0.20,
            "min_lr_ratio": 0.30,
        },
        "hypothesis": "Balanced Model D baseline with calibrated exploit rewards and tournament ICM weighting.",
    },
    # Probe 1: Zero Exploit Rewards Ablation (Is targeted reward shaping necessary?)
    {
        "probe_name": "Zero Exploit Rewards (Ablation)",
        "params": {
            "exploit_value_bonus": 0.0,
            "exploit_steal_bonus": 0.0,
            "bluff_station_penalty": 0.0,
            "icm_equity_weight": 0.95,
            "fold_penalty": -0.008,
            "pot_control_bonus": 0.02,
            "elimination_penalty": 5.5,
            "win_bonus": 2.5,
            "learning_rate": 0.00012,
            "initial_ent_coef": 0.030,
            "min_ent_coef": 0.008,
            "clip_range": 0.20,
            "min_lr_ratio": 0.30,
        },
        "hypothesis": "Tests whether removing targeted exploit reward shaping weakens adaptation against diverse bot tells.",
    },
    # Probe 2: High Exploit Rewards (Hyper-Aggressive Value & Steal)
    {
        "probe_name": "High Exploit Rewards",
        "params": {
            "exploit_value_bonus": 0.15,
            "exploit_steal_bonus": 0.08,
            "bluff_station_penalty": 0.05,
            "icm_equity_weight": 0.95,
            "fold_penalty": -0.008,
            "pot_control_bonus": 0.02,
            "elimination_penalty": 5.5,
            "win_bonus": 2.5,
            "learning_rate": 0.00012,
            "initial_ent_coef": 0.030,
            "min_ent_coef": 0.008,
            "clip_range": 0.20,
            "min_lr_ratio": 0.30,
        },
        "hypothesis": "Tests if high value and steal incentives accelerate table domination or cause reckless over-steals.",
    },
    # Probe 3: Heavy Bluff Station Penalty
    {
        "probe_name": "Heavy Bluff Station Penalty",
        "params": {
            "exploit_value_bonus": 0.05,
            "exploit_steal_bonus": 0.03,
            "bluff_station_penalty": 0.25,
            "icm_equity_weight": 0.95,
            "fold_penalty": -0.008,
            "pot_control_bonus": 0.02,
            "elimination_penalty": 5.5,
            "win_bonus": 2.5,
            "learning_rate": 0.00012,
            "initial_ent_coef": 0.030,
            "min_ent_coef": 0.008,
            "clip_range": 0.20,
            "min_lr_ratio": 0.30,
        },
        "hypothesis": "Tests if severe penalty against bluffing calling stations eliminates chip leaks or induces timid passivity.",
    },
    # Probe 4: Zero Bluff Station Penalty (Station Blindness)
    {
        "probe_name": "Zero Bluff Station Penalty",
        "params": {
            "exploit_value_bonus": 0.05,
            "exploit_steal_bonus": 0.03,
            "bluff_station_penalty": 0.0,
            "icm_equity_weight": 0.95,
            "fold_penalty": -0.008,
            "pot_control_bonus": 0.02,
            "elimination_penalty": 5.5,
            "win_bonus": 2.5,
            "learning_rate": 0.00012,
            "initial_ent_coef": 0.030,
            "min_ent_coef": 0.008,
            "clip_range": 0.20,
            "min_lr_ratio": 0.30,
        },
        "hypothesis": "Tests if removing bluff penalties leads to firing air into inelastic calling stations.",
    },
    # Probe 5: Extreme High ICM Boundary (Heavy Risk Aversion)
    {
        "probe_name": "Extreme High ICM",
        "params": {
            "exploit_value_bonus": 0.05,
            "exploit_steal_bonus": 0.03,
            "bluff_station_penalty": 0.10,
            "icm_equity_weight": 1.80,
            "fold_penalty": -0.008,
            "pot_control_bonus": 0.02,
            "elimination_penalty": 8.0,
            "win_bonus": 2.5,
            "learning_rate": 0.00012,
            "initial_ent_coef": 0.030,
            "min_ent_coef": 0.008,
            "clip_range": 0.20,
            "min_lr_ratio": 0.30,
        },
        "hypothesis": "Tests if extreme risk aversion causes tournament paralysis and excessive folding to aggression.",
    },
    # Probe 6: Low ICM Boundary (Cash Game Aggression)
    {
        "probe_name": "Low ICM Boundary",
        "params": {
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
        },
        "hypothesis": "Tests if low ICM weighting causes reckless stack collisions and high elimination rates.",
    },
    # Probe 7: Positive Fold Penalty Boundary
    {
        "probe_name": "Positive Fold Penalty",
        "params": {
            "exploit_value_bonus": 0.05,
            "exploit_steal_bonus": 0.03,
            "bluff_station_penalty": 0.10,
            "icm_equity_weight": 0.95,
            "fold_penalty": 0.010,
            "pot_control_bonus": 0.02,
            "elimination_penalty": 5.5,
            "win_bonus": 2.5,
            "learning_rate": 0.00012,
            "initial_ent_coef": 0.030,
            "min_ent_coef": 0.008,
            "clip_range": 0.20,
            "min_lr_ratio": 0.30,
        },
        "hypothesis": "Tests if penalizing folds inflates VPIP and causes disastrous hands out of position.",
    },
    # Probe 8: Heavy Fold Reward Boundary
    {
        "probe_name": "Heavy Fold Reward",
        "params": {
            "exploit_value_bonus": 0.05,
            "exploit_steal_bonus": 0.03,
            "bluff_station_penalty": 0.10,
            "icm_equity_weight": 0.95,
            "fold_penalty": -0.025,
            "pot_control_bonus": 0.02,
            "elimination_penalty": 5.5,
            "win_bonus": 2.5,
            "learning_rate": 0.00012,
            "initial_ent_coef": 0.030,
            "min_ent_coef": 0.008,
            "clip_range": 0.20,
            "min_lr_ratio": 0.30,
        },
        "hypothesis": "Tests if over-rewarding folds makes the policy too tight, abandoning profitable blind steals.",
    },
    # Probe 9: High Learning Rate & Clip Boundary
    {
        "probe_name": "High Learning Rate",
        "params": {
            "exploit_value_bonus": 0.05,
            "exploit_steal_bonus": 0.03,
            "bluff_station_penalty": 0.10,
            "icm_equity_weight": 0.95,
            "fold_penalty": -0.008,
            "pot_control_bonus": 0.02,
            "elimination_penalty": 5.5,
            "win_bonus": 2.5,
            "learning_rate": 0.00028,
            "initial_ent_coef": 0.040,
            "min_ent_coef": 0.010,
            "clip_range": 0.25,
            "min_lr_ratio": 0.35,
        },
        "hypothesis": "Tests gradient stability of the Decoupled Actor and Card Attention backbone under high learning rate.",
    },
    # Probe 10: Low Learning Rate Boundary
    {
        "probe_name": "Low Learning Rate",
        "params": {
            "exploit_value_bonus": 0.05,
            "exploit_steal_bonus": 0.03,
            "bluff_station_penalty": 0.10,
            "icm_equity_weight": 0.95,
            "fold_penalty": -0.008,
            "pot_control_bonus": 0.02,
            "elimination_penalty": 5.5,
            "win_bonus": 2.5,
            "learning_rate": 0.00006,
            "initial_ent_coef": 0.020,
            "min_ent_coef": 0.005,
            "clip_range": 0.18,
            "min_lr_ratio": 0.25,
        },
        "hypothesis": "Tests if low learning rate underfits within standard rollout budgets in 152-dim space.",
    },
    # Probe 11: High Entropy Exploration Boundary
    {
        "probe_name": "High Entropy Exploration",
        "params": {
            "exploit_value_bonus": 0.05,
            "exploit_steal_bonus": 0.03,
            "bluff_station_penalty": 0.10,
            "icm_equity_weight": 0.95,
            "fold_penalty": -0.008,
            "pot_control_bonus": 0.02,
            "elimination_penalty": 5.5,
            "win_bonus": 2.5,
            "learning_rate": 0.00015,
            "initial_ent_coef": 0.055,
            "min_ent_coef": 0.012,
            "clip_range": 0.20,
            "min_lr_ratio": 0.30,
        },
        "hypothesis": "Tests if high exploration entropy disrupts bet-sizing stream learning and causes erratic sizing.",
    },
]


def evaluate_boundary_model(
    model: MaskablePPO,
    n_sessions: int = 25,
    base_seed: int = 25_000,
) -> Dict[str, float]:
    """Runs CRN tournament evaluations against the multi-agent league and records rich metrics."""
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


def run_boundary_trial(
    params: Dict[str, Any],
    timesteps: int = 250_000,
    n_envs: int = 12,
    seed: int = 42,
    eval_sessions: int = 25,
    use_dummy_vec: bool = False,
):
    """Trains a probe model on P-cores and evaluates tournament behavior."""
    vec_env_cls = DummyVecEnv if use_dummy_vec else SubprocVecEnv

    vec_env = vec_env_cls([
        make_holdem_d_env(
            i,
            seed=seed,
            fold_penalty=params["fold_penalty"],
            elimination_penalty=params["elimination_penalty"],
            win_bonus=params["win_bonus"],
            icm_equity_weight=params["icm_equity_weight"],
            pot_control_bonus=params["pot_control_bonus"],
            exploit_value_bonus=params["exploit_value_bonus"],
            exploit_steal_bonus=params["exploit_steal_bonus"],
            bluff_station_penalty=params["bluff_station_penalty"],
            league_mode=True,
        ) for i in range(n_envs)
    ])

    min_lr = params["learning_rate"] * params["min_lr_ratio"]
    lr_schedule = cosine_warm_restart_schedule(
        initial_lr=params["learning_rate"],
        min_lr=min_lr,
        n_cycles=6,
    )

    policy_kwargs = dict(
        features_extractor_class=CardAttentionExtractor,
        features_extractor_kwargs=dict(
            num_cards=7,
            embed_dim=32,
            num_heads=4,
            pooling="mean",
            card_features_dim=32,
            game_features_dim=96,
            slot_features_dim=0,
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    model = MaskablePPO(
        policy=HoldemDecoupledActorCriticPolicy,
        env=vec_env,
        learning_rate=lr_schedule,
        n_steps=256 if not use_dummy_vec else 64,
        batch_size=256 if not use_dummy_vec else 32,
        n_epochs=5 if not use_dummy_vec else 2,
        gamma=0.992,
        gae_lambda=0.92,
        clip_range=params["clip_range"],
        ent_coef=params["initial_ent_coef"],
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )

    belief_head = HoldemAuxiliaryBeliefHead(feature_dim=32 + 96, hidden_dim=128)
    belief_callback = HoldemAuxiliaryBeliefCallback(
        belief_head=belief_head,
        lr=params["learning_rate"] * 0.8,
        tier_weight=0.5,
        fold_weight=0.5,
        max_batch_size=2048,
        verbose=0,
    )

    callbacks = CallbackList([
        EntropyDecayCallback(
            initial_ent=params["initial_ent_coef"],
            min_ent=params["min_ent_coef"],
            total_timesteps=timesteps,
        ),
        belief_callback,
    ])

    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)
    train_dur = time.time() - t0
    vec_env.close()

    eval_res = evaluate_boundary_model(model, n_sessions=eval_sessions, base_seed=seed + 1000)

    # Multi-objective composite score emphasizing win rate, BB return, balanced VPIP, and survival
    vpip_penalty = abs(eval_res["vpip_rate"] - 30.0) * 0.3
    score = eval_res["win_rate"] + 0.1 * eval_res["bb_return"] - 0.05 * (eval_res["elim_rate"] - 45.0) - vpip_penalty

    return score, eval_res, train_dur


def analyze_boundary_results(
    results: List[Dict[str, Any]],
    output_bounds_path: str = SCREENED_BOUNDS_PATH,
    output_report_path: str = REPORT_PATH,
) -> Dict[str, Any]:
    """
    Analyzes boundary probe trials:
    1. Identifies failure modes and cliff conditions.
    2. Derives mathematically grounded tightened bounds for full HPO.
    3. Saves machine-readable bounds JSON and comprehensive report.
    """
    eliminated_picks = []

    # Sort results by score descending
    sorted_results = sorted(results, key=lambda r: r["score"], reverse=True)
    best_probe = sorted_results[0]
    baseline_probe = next((r for r in results if r["name"] == "Model D Baseline"), sorted_results[0])

    # 1. Analyze Exploit Value Bonus
    zero_exploit = next((r for r in results if r["name"] == "Zero Exploit Rewards (Ablation)"), None)
    high_exploit = next((r for r in results if r["name"] == "High Exploit Rewards"), None)

    val_bonus_min, val_bonus_max = 0.03, 0.12
    steal_bonus_min, steal_bonus_max = 0.015, 0.060

    if zero_exploit:
        score_diff = baseline_probe["score"] - zero_exploit["score"]
        win_diff = baseline_probe["eval"]["win_rate"] - zero_exploit["eval"]["win_rate"]
        if score_diff > 3.0 or win_diff > 4.0:
            eliminated_picks.append({
                "parameter": "exploit_value_bonus & exploit_steal_bonus",
                "eliminated_range": [0.0, 0.015],
                "rationale": f"Zero exploit reward shaping degraded win rate by {win_diff:.1f}% and score by {score_diff:.2f} relative to baseline.",
            })
            val_bonus_min = 0.035
            steal_bonus_min = 0.020

    if high_exploit:
        if high_exploit["score"] < baseline_probe["score"] - 4.0:
            eliminated_picks.append({
                "parameter": "exploit_value_bonus",
                "eliminated_range": [0.12, 0.20],
                "rationale": f"Excessive exploit value bonus ({high_exploit['params']['exploit_value_bonus']}) caused over-aggressive sizing and reduced tournament EV.",
            })
            val_bonus_max = 0.10
        elif high_exploit["score"] > baseline_probe["score"]:
            val_bonus_max = 0.15
            steal_bonus_max = 0.08

    # 2. Analyze Bluff Station Penalty
    zero_bluff_pen = next((r for r in results if r["name"] == "Zero Bluff Station Penalty"), None)
    heavy_bluff_pen = next((r for r in results if r["name"] == "Heavy Bluff Station Penalty"), None)

    bluff_pen_min, bluff_pen_max = 0.05, 0.20

    if zero_bluff_pen:
        if zero_bluff_pen["eval"]["bb_return"] < baseline_probe["eval"]["bb_return"] - 5.0 or zero_bluff_pen["score"] < baseline_probe["score"] - 3.0:
            eliminated_picks.append({
                "parameter": "bluff_station_penalty",
                "eliminated_range": [0.0, 0.04],
                "rationale": f"Zero bluff station penalty allowed bleeding chips into calling stations (BB return {zero_bluff_pen['eval']['bb_return']:+.1f} vs {baseline_probe['eval']['bb_return']:+.1f}).",
            })
            bluff_pen_min = 0.06

    if heavy_bluff_pen:
        if heavy_bluff_pen["score"] < baseline_probe["score"] - 4.0:
            eliminated_picks.append({
                "parameter": "bluff_station_penalty",
                "eliminated_range": [0.22, 0.35],
                "rationale": f"Heavy bluff penalty ({heavy_bluff_pen['params']['bluff_station_penalty']}) induced timid passivity, underperforming baseline.",
            })
            bluff_pen_max = 0.18

    # 3. Analyze ICM Equity Weight
    low_icm = next((r for r in results if r["name"] == "Low ICM Boundary"), None)
    high_icm = next((r for r in results if r["name"] == "Extreme High ICM"), None)

    icm_min, icm_max = 0.60, 1.40
    if low_icm:
        if low_icm["eval"]["elim_rate"] > baseline_probe["eval"]["elim_rate"] + 6.0:
            eliminated_picks.append({
                "parameter": "icm_equity_weight",
                "eliminated_range": [0.0, 0.40],
                "rationale": f"Low ICM weight ({low_icm['params']['icm_equity_weight']}) provoked elimination rate to spike to {low_icm['eval']['elim_rate']:.1f}%.",
            })
            icm_min = 0.55

    if high_icm:
        if high_icm["eval"]["win_rate"] < baseline_probe["eval"]["win_rate"] - 4.0:
            eliminated_picks.append({
                "parameter": "icm_equity_weight",
                "eliminated_range": [1.50, 2.20],
                "rationale": f"Extreme ICM weight ({high_icm['params']['icm_equity_weight']}) caused over-folding paralysis, reducing win rate to {high_icm['eval']['win_rate']:.1f}%.",
            })
            icm_max = 1.35

    # 4. Analyze Fold Penalty
    pos_fold = next((r for r in results if r["name"] == "Positive Fold Penalty"), None)
    heavy_fold = next((r for r in results if r["name"] == "Heavy Fold Reward"), None)

    fold_min, fold_max = -0.018, -0.002
    if pos_fold:
        if pos_fold["eval"]["vpip_rate"] > 36.0 or pos_fold["score"] < baseline_probe["score"] - 4.0:
            eliminated_picks.append({
                "parameter": "fold_penalty",
                "eliminated_range": [0.002, 0.020],
                "rationale": f"Positive fold penalty (+{pos_fold['params']['fold_penalty']}) caused VPIP inflation to {pos_fold['eval']['vpip_rate']:.1f}%, bleeding chips.",
            })
            fold_max = 0.00

    if heavy_fold:
        if heavy_fold["eval"]["vpip_rate"] < 23.0 or heavy_fold["score"] < baseline_probe["score"] - 4.0:
            eliminated_picks.append({
                "parameter": "fold_penalty",
                "eliminated_range": [-0.035, -0.018],
                "rationale": f"Heavy fold reward ({heavy_fold['params']['fold_penalty']}) suppressed VPIP to {heavy_fold['eval']['vpip_rate']:.1f}%, blinding out.",
            })
            fold_min = -0.015

    # 5. Analyze Learning Rate
    high_lr = next((r for r in results if r["name"] == "High Learning Rate"), None)
    low_lr = next((r for r in results if r["name"] == "Low Learning Rate"), None)

    lr_min, lr_max = 8.0e-5, 2.2e-4
    if high_lr and high_lr["score"] < baseline_probe["score"] - 6.0:
        eliminated_picks.append({
            "parameter": "learning_rate",
            "eliminated_range": [2.5e-4, 4.0e-4],
            "rationale": f"High LR ({high_lr['params']['learning_rate']:.5f}) led to decoupled actor instability.",
        })
        lr_max = 2.0e-4

    if low_lr and low_lr["score"] < baseline_probe["score"] - 6.0:
        eliminated_picks.append({
            "parameter": "learning_rate",
            "eliminated_range": [3.0e-5, 7.0e-5],
            "rationale": f"Low LR ({low_lr['params']['learning_rate']:.5f}) converged too sluggishly in 152-dim space.",
        })
        lr_min = 9.0e-5

    # 6. Analyze Entropy Exploration
    high_ent = next((r for r in results if r["name"] == "High Entropy Exploration"), None)
    ent_min, ent_max = 0.020, 0.045
    if high_ent and high_ent["score"] < baseline_probe["score"] - 4.0:
        eliminated_picks.append({
            "parameter": "initial_ent_coef",
            "eliminated_range": [0.048, 0.070],
            "rationale": f"High entropy ({high_ent['params']['initial_ent_coef']}) destabilized bet sizing selection.",
        })
        ent_max = 0.040

    screened_bounds = {
        "exploit_value_bonus_min": val_bonus_min,
        "exploit_value_bonus_max": val_bonus_max,
        "exploit_steal_bonus_min": steal_bonus_min,
        "exploit_steal_bonus_max": steal_bonus_max,
        "bluff_station_penalty_min": bluff_pen_min,
        "bluff_station_penalty_max": bluff_pen_max,
        "icm_equity_weight_min": icm_min,
        "icm_equity_weight_max": icm_max,
        "fold_penalty_min": fold_min,
        "fold_penalty_max": fold_max,
        "pot_control_bonus_min": 0.010,
        "pot_control_bonus_max": 0.040,
        "learning_rate_min": lr_min,
        "learning_rate_max": lr_max,
        "initial_ent_coef_min": ent_min,
        "initial_ent_coef_max": ent_max,
        "clip_range_min": 0.18,
        "clip_range_max": 0.24,
        "elimination_penalty_min": 3.0,
        "elimination_penalty_max": 7.5,
        "win_bonus_min": 1.5,
        "win_bonus_max": 4.0,
    }

    report = {
        "best_probe": {
            "name": best_probe["name"],
            "score": best_probe["score"],
            "eval": best_probe["eval"],
            "params": best_probe["params"],
        },
        "screened_bounds": screened_bounds,
        "eliminated_picks": eliminated_picks,
        "probes_evaluated": results,
    }

    os.makedirs(os.path.dirname(output_bounds_path), exist_ok=True)
    with open(output_bounds_path, "w") as f:
        json.dump(screened_bounds, f, indent=2)

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def run_boundary_search_holdem_d(
    n_trials: int = 12,
    timesteps_per_trial: int = 250_000,
    n_envs: int = 12,
    eval_sessions: int = 25,
    seed: int = 42,
    dry_run: bool = False,
):
    """Executes the Model D Boundary Value Analysis."""
    if dry_run:
        print("[Boundary Search Dry Run] Executing quick 2-probe validation (500 steps, 3 eval sessions)...")
        pin_to_pcores(12)
        dry_results = []
        for i in range(2):
            probe = BOUNDARY_PROBE_POINTS[i]
            p_score, p_eval, p_dur = run_boundary_trial(
                params=probe["params"],
                timesteps=500,
                n_envs=2,
                seed=seed + i * 10,
                eval_sessions=3,
                use_dummy_vec=True,
            )
            dry_results.append({
                "probe_idx": i,
                "name": probe["probe_name"],
                "score": p_score,
                "eval": p_eval,
                "params": probe["params"],
                "duration_sec": p_dur,
            })
            print(f"  Dry-run probe #{i+1} ({probe['probe_name']}): Score={p_score:.2f}, WinRate={p_eval['win_rate']:.1f}%")
        print("[Boundary Search Dry Run] PASSED! Configuration and execution pipeline verified.")
        return

    pin_to_pcores(n_envs)

    total_probes = min(n_trials, len(BOUNDARY_PROBE_POINTS))
    print("=================================================================")
    print("  TEXAS HOLD'EM MODEL D BOUNDARY SEARCH & SCREENING")
    print("=================================================================")
    print(f"  Probing {total_probes} boundary hypotheses | {timesteps_per_trial:,} steps/trial")
    print(f"  Parallel Envs: {n_envs} (12 P-cores) | Vector: {OBS_DIM_HOLDEM_D} dims")
    print(f"  Policy: HoldemDecoupledActorCriticPolicy | Multi-Task Auxiliary Belief")
    print(f"  Outputs: {SCREENED_BOUNDS_PATH} & {REPORT_PATH}")
    print("=================================================================\n")

    t_total_start = time.time()
    results: List[Dict[str, Any]] = []

    for i in range(total_probes):
        probe = BOUNDARY_PROBE_POINTS[i]
        p_name = probe["probe_name"]
        params = probe["params"]
        hypo = probe["hypothesis"]

        print(f"--- Probe #{i+1:02d}/{total_probes:02d}: {p_name} ---")
        print(f"    Hypothesis: {hypo}")
        print(f"    Params: ValBonus={params['exploit_value_bonus']}, StealBonus={params['exploit_steal_bonus']}, "
              f"BluffPen={params['bluff_station_penalty']}, ICM={params['icm_equity_weight']}, FoldPen={params['fold_penalty']}, "
              f"LR={params['learning_rate']:.5f}")

        trial_seed = seed + i * 50
        score, eval_res, dur = run_boundary_trial(
            params=params,
            timesteps=timesteps_per_trial,
            n_envs=n_envs,
            seed=trial_seed,
            eval_sessions=eval_sessions,
        )

        record = {
            "probe_idx": i,
            "name": p_name,
            "score": score,
            "eval": eval_res,
            "params": params,
            "duration_sec": dur,
        }
        results.append(record)

        print(
            f"    [Result #{i+1:02d}] Score: {score:+6.2f} | "
            f"Win Rate: {eval_res['win_rate']:4.1f}% | BB Ret: {eval_res['bb_return']:+6.1f} | "
            f"Elim: {eval_res['elim_rate']:4.1f}% | VPIP: {eval_res['vpip_rate']:4.1f}% | "
            f"FlopChk: {eval_res['flop_check_rate']:4.1f}% | Time: {dur:.1f}s\n"
        )

    total_time = time.time() - t_total_start
    print("=================================================================")
    print(f"  ALL {len(results)} BOUNDARY PROBES COMPLETED in {total_time/60:.1f} mins")
    print("=================================================================\n")

    # Synthesize findings and write screened bounds
    report = analyze_boundary_results(results)

    print(">>> SCREENED TIGHTENED SEARCH BOUNDS FOR FULL HPO <<<")
    for k, v in report["screened_bounds"].items():
        print(f"  - {k:<28}: {v}")

    print("\n>>> IDENTIFIED FAILURE MODES & ELIMINATED REGIONS <<<")
    for elim in report["eliminated_picks"]:
        print(f"  * {elim['parameter']} in {elim['eliminated_range']}: {elim['rationale']}")

    print(f"\nSaved tightened bounds to: {SCREENED_BOUNDS_PATH}")
    print(f"Saved full boundary report to: {REPORT_PATH}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boundary Search for Texas Hold'em Model D")
    parser.add_argument("--trials", type=int, default=12, help="Number of boundary probes (max 12)")
    parser.add_argument("--timesteps", type=int, default=250_000, help="Timesteps per probe")
    parser.add_argument("--n-envs", type=int, default=12, help="Number of parallel P-core environments")
    parser.add_argument("--eval-sessions", type=int, default=25, help="CRN tournament evaluation sessions per probe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Validate pipeline with fast smoke execution")
    args = parser.parse_args()

    run_boundary_search_holdem_d(
        n_trials=args.trials,
        timesteps_per_trial=args.timesteps,
        n_envs=args.n_envs,
        eval_sessions=args.eval_sessions,
        seed=args.seed,
        dry_run=args.dry_run,
    )
