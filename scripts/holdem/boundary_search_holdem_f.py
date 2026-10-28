"""
scripts/holdem/boundary_search_holdem_f.py — Boundary Value & Screening Search for Texas Hold'em Model F.
The Superhuman Closer.

Probes the extreme edges and critical transition points of Model F:
1. uncontested_pot_bonus: [0.0 (ablation) vs 0.015 (baseline) vs 0.045 (hyper-closer)]
2. scaffold_steps: [60k (fast transition) vs 110k (baseline) vs 160k (extended scaffolding)]
3. min_shaping_mult: [0.0 (pure chip-EV) vs 0.02 (residual shaping prior)]
4. exploit_value_bonus: [0.04 (mild) vs 0.105 (baseline) vs 0.16 (hyper-aggressive value)]
5. exploit_steal_bonus: [0.02 (mild) vs 0.038 (baseline) vs 0.065 (aggressive stealing)]
6. bluff_station_penalty: [0.05 (mild) vs 0.093 (baseline) vs 0.20 (heavy punting penalty)]
7. icm_equity_weight: [0.25 (cash aggression) vs 0.67 (baseline) vs 1.10 (high risk aversion)]
8. elimination_penalty: [1.0 (shallow) vs 1.76 (baseline) vs 4.0 (deep survival)]
9. win_bonus: [1.0 vs 1.15 vs 2.5]
10. learning_rate: [4.8e-5 (low) vs 9.1e-5 (baseline) vs 1.5e-4 (high)]
11. initial_ent_coef: [0.018 (disciplined) vs 0.0266 (baseline) vs 0.035 (high exploration)]
12. min_ent_coef: [0.003 (deep exploitation) vs 0.0046 (baseline) vs 0.008 (fluid)]
13. clip_range: [0.18 vs 0.21 vs 0.25]

Runs 12 curated boundary probe trials (250,000 timesteps each via 12 P-core SubprocVecEnvs),
evaluates each trial across 25 Common Random Numbers (CRN) tournament sessions against
the HoldemLeaguePool (Model E Champion + Model D Snapshots + Jittered Archetype Bots), and synthesizes:
- configs/holdem/screened_bounds_holdem_f.json (tightened search space for full HPO)
- results/holdem/model_f_boundary_report.json (empirical failure analysis & trade-off synthesis)
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

from games.holdem.gym_wrapper import HoldemPokerGymEnv, OBS_DIM_HOLDEM_F
from games.holdem.action_abstraction import (
    NUM_ACTIONS_HOLDEM_F,
    A_F_FOLD,
    A_F_CHECK_CALL,
    A_F_MIN_RAISE,
    A_F_THIRD_POT,
    A_F_HALF_POT,
    A_F_POT,
    A_F_OVERBET,
    A_F_ALL_IN,
    STREET_PREFLOP,
    STREET_FLOP,
)
from rl.holdem_decoupled_actor import HoldemDecoupledActorCriticPolicy
from rl.holdem_auxiliary_belief import HoldemAuxiliaryBeliefHead, HoldemAuxiliaryBeliefCallback
from rl.holdem_league import HoldemLeaguePool
from rl.holdem_curriculum import HoldemCurriculumRewardCallback
from scripts.holdem.train_holdem_f import (
    make_holdem_f_env,
    linear_warmup_cosine_decay,
    HoldemEntropyDecayCallback,
    pin_to_pcores,
)

SCREENED_BOUNDS_PATH = os.path.join(PROJECT_ROOT, "configs/holdem/screened_bounds_holdem_f.json")
REPORT_PATH = os.path.join(PROJECT_ROOT, "results/holdem/model_f_boundary_report.json")


# 12 Curated Boundary Value Probes targeting specific behavioral and mathematical hypotheses for Model F
BOUNDARY_PROBE_POINTS: List[Dict[str, Any]] = [
    # Probe 0: Model F Baseline (Model E Best Params + Default Steal Bonus)
    {
        "probe_name": "Model F Baseline (Model E Params)",
        "params": {
            "scaffold_steps": 110_000,
            "min_shaping_mult": 0.002,
            "exploit_value_bonus": 0.105,
            "exploit_steal_bonus": 0.038,
            "bluff_station_penalty": 0.093,
            "icm_equity_weight": 0.67,
            "fold_penalty": -0.006,
            "pot_control_bonus": 0.035,
            "elimination_penalty": 1.76,
            "win_bonus": 1.15,
            "uncontested_pot_bonus": 0.015,
            "learning_rate": 9.1e-5,
            "initial_ent_coef": 0.0266,
            "min_ent_coef": 0.0046,
            "clip_range": 0.21,
        },
        "hypothesis": "Baseline transfer of Model E champion parameters to Model F architecture.",
    },
    # Probe 1: High Steal Bonus (Hyper-Aggressive Closer)
    {
        "probe_name": "High Steal Bonus (0.045)",
        "params": {
            "scaffold_steps": 110_000,
            "min_shaping_mult": 0.002,
            "exploit_value_bonus": 0.105,
            "exploit_steal_bonus": 0.055,
            "bluff_station_penalty": 0.093,
            "icm_equity_weight": 0.67,
            "fold_penalty": -0.006,
            "pot_control_bonus": 0.035,
            "elimination_penalty": 1.76,
            "win_bonus": 1.15,
            "uncontested_pot_bonus": 0.045,
            "learning_rate": 9.1e-5,
            "initial_ent_coef": 0.0266,
            "min_ent_coef": 0.0046,
            "clip_range": 0.21,
        },
        "hypothesis": "Tests if aggressive uncontested pot stealing closes tournaments faster and avoids showdown variance.",
    },
    # Probe 2: Zero Steal Bonus (Ablation)
    {
        "probe_name": "Zero Steal Bonus (0.0 Ablation)",
        "params": {
            "scaffold_steps": 110_000,
            "min_shaping_mult": 0.002,
            "exploit_value_bonus": 0.105,
            "exploit_steal_bonus": 0.038,
            "bluff_station_penalty": 0.093,
            "icm_equity_weight": 0.67,
            "fold_penalty": -0.006,
            "pot_control_bonus": 0.035,
            "elimination_penalty": 1.76,
            "win_bonus": 1.15,
            "uncontested_pot_bonus": 0.0,
            "learning_rate": 9.1e-5,
            "initial_ent_coef": 0.0266,
            "min_ent_coef": 0.0046,
            "clip_range": 0.21,
        },
        "hypothesis": "Ablation test: determines performance drop when uncontested showdown-avoidance incentive is removed.",
    },
    # Probe 3: Fast Curriculum Annealing (60k Steps)
    {
        "probe_name": "Fast Curriculum Anneal (60k Steps)",
        "params": {
            "scaffold_steps": 60_000,
            "min_shaping_mult": 0.0,
            "exploit_value_bonus": 0.105,
            "exploit_steal_bonus": 0.038,
            "bluff_station_penalty": 0.093,
            "icm_equity_weight": 0.67,
            "fold_penalty": -0.006,
            "pot_control_bonus": 0.035,
            "elimination_penalty": 1.76,
            "win_bonus": 1.15,
            "uncontested_pot_bonus": 0.015,
            "learning_rate": 9.1e-5,
            "initial_ent_coef": 0.0266,
            "min_ent_coef": 0.0046,
            "clip_range": 0.21,
        },
        "hypothesis": "Tests whether rapid transition to pure chip EV improves endgame closing without shaping distortion.",
    },
    # Probe 4: Extended Scaffolding (160k Steps)
    {
        "probe_name": "Extended Scaffolding (160k Steps)",
        "params": {
            "scaffold_steps": 160_000,
            "min_shaping_mult": 0.015,
            "exploit_value_bonus": 0.105,
            "exploit_steal_bonus": 0.038,
            "bluff_station_penalty": 0.093,
            "icm_equity_weight": 0.67,
            "fold_penalty": -0.006,
            "pot_control_bonus": 0.035,
            "elimination_penalty": 1.76,
            "win_bonus": 1.15,
            "uncontested_pot_bonus": 0.015,
            "learning_rate": 9.1e-5,
            "initial_ent_coef": 0.0266,
            "min_ent_coef": 0.0046,
            "clip_range": 0.21,
        },
        "hypothesis": "Tests if maintaining shaping guidance across 64% of training stabilizes policy in self-play.",
    },
    # Probe 5: High Learning Rate (1.5e-4)
    {
        "probe_name": "High Learning Rate (1.5e-4)",
        "params": {
            "scaffold_steps": 110_000,
            "min_shaping_mult": 0.002,
            "exploit_value_bonus": 0.105,
            "exploit_steal_bonus": 0.038,
            "bluff_station_penalty": 0.093,
            "icm_equity_weight": 0.67,
            "fold_penalty": -0.006,
            "pot_control_bonus": 0.035,
            "elimination_penalty": 1.76,
            "win_bonus": 1.15,
            "uncontested_pot_bonus": 0.015,
            "learning_rate": 1.5e-4,
            "initial_ent_coef": 0.030,
            "min_ent_coef": 0.006,
            "clip_range": 0.24,
        },
        "hypothesis": "Tests if faster learning rate accelerates policy adaptation across the 8-action betting space.",
    },
    # Probe 6: Low Learning Rate (4.8e-5)
    {
        "probe_name": "Low Learning Rate (4.8e-5)",
        "params": {
            "scaffold_steps": 110_000,
            "min_shaping_mult": 0.002,
            "exploit_value_bonus": 0.105,
            "exploit_steal_bonus": 0.038,
            "bluff_station_penalty": 0.093,
            "icm_equity_weight": 0.67,
            "fold_penalty": -0.006,
            "pot_control_bonus": 0.035,
            "elimination_penalty": 1.76,
            "win_bonus": 1.15,
            "uncontested_pot_bonus": 0.015,
            "learning_rate": 4.8e-5,
            "initial_ent_coef": 0.022,
            "min_ent_coef": 0.0035,
            "clip_range": 0.18,
        },
        "hypothesis": "Tests if conservative gradient steps prevent policy divergence against neural sparring opponents.",
    },
    # Probe 7: Hyper-Aggressive Station Punisher
    {
        "probe_name": "Hyper-Exploit Punisher (0.16 Val / 0.06 Steal)",
        "params": {
            "scaffold_steps": 110_000,
            "min_shaping_mult": 0.002,
            "exploit_value_bonus": 0.16,
            "exploit_steal_bonus": 0.06,
            "bluff_station_penalty": 0.18,
            "icm_equity_weight": 0.67,
            "fold_penalty": -0.006,
            "pot_control_bonus": 0.035,
            "elimination_penalty": 1.76,
            "win_bonus": 1.15,
            "uncontested_pot_bonus": 0.025,
            "learning_rate": 9.1e-5,
            "initial_ent_coef": 0.0266,
            "min_ent_coef": 0.0046,
            "clip_range": 0.21,
        },
        "hypothesis": "Tests if maximum extraction from CallingStations and punishing TAG over-folding yields highest win rates.",
    },
    # Probe 8: Low ICM Cash Aggression
    {
        "probe_name": "Low ICM Boundary (0.25 Weight)",
        "params": {
            "scaffold_steps": 110_000,
            "min_shaping_mult": 0.002,
            "exploit_value_bonus": 0.105,
            "exploit_steal_bonus": 0.038,
            "bluff_station_penalty": 0.093,
            "icm_equity_weight": 0.25,
            "fold_penalty": -0.006,
            "pot_control_bonus": 0.025,
            "elimination_penalty": 1.0,
            "win_bonus": 1.0,
            "uncontested_pot_bonus": 0.020,
            "learning_rate": 9.1e-5,
            "initial_ent_coef": 0.0266,
            "min_ent_coef": 0.0046,
            "clip_range": 0.21,
        },
        "hypothesis": "Tests if low risk aversion unlocks dominant stack accumulation needed to enter heads-up with 70%+ chips.",
    },
    # Probe 9: High ICM Deep Survival
    {
        "probe_name": "High ICM Boundary (1.10 Weight)",
        "params": {
            "scaffold_steps": 110_000,
            "min_shaping_mult": 0.002,
            "exploit_value_bonus": 0.105,
            "exploit_steal_bonus": 0.038,
            "bluff_station_penalty": 0.093,
            "icm_equity_weight": 1.10,
            "fold_penalty": -0.006,
            "pot_control_bonus": 0.035,
            "elimination_penalty": 4.0,
            "win_bonus": 2.20,
            "uncontested_pot_bonus": 0.015,
            "learning_rate": 9.1e-5,
            "initial_ent_coef": 0.0266,
            "min_ent_coef": 0.0046,
            "clip_range": 0.21,
        },
        "hypothesis": "Tests if high survival pressure prevents unnecessary busts and reliably guarantees Top 2 finishes.",
    },
    # Probe 10: Low Entropy Precision Sizing
    {
        "probe_name": "Low Entropy / Precision Sizing (0.018 Init)",
        "params": {
            "scaffold_steps": 110_000,
            "min_shaping_mult": 0.002,
            "exploit_value_bonus": 0.105,
            "exploit_steal_bonus": 0.038,
            "bluff_station_penalty": 0.093,
            "icm_equity_weight": 0.67,
            "fold_penalty": -0.006,
            "pot_control_bonus": 0.035,
            "elimination_penalty": 1.76,
            "win_bonus": 1.15,
            "uncontested_pot_bonus": 0.015,
            "learning_rate": 9.1e-5,
            "initial_ent_coef": 0.018,
            "min_ent_coef": 0.003,
            "clip_range": 0.20,
        },
        "hypothesis": "Tests if lower entropy prevents random bet-sizing fuzziness and locks into optimal sizing faster.",
    },
    # Probe 11: High Entropy Exploration
    {
        "probe_name": "High Entropy Exploration (0.035 Init)",
        "params": {
            "scaffold_steps": 110_000,
            "min_shaping_mult": 0.002,
            "exploit_value_bonus": 0.105,
            "exploit_steal_bonus": 0.038,
            "bluff_station_penalty": 0.093,
            "icm_equity_weight": 0.67,
            "fold_penalty": -0.006,
            "pot_control_bonus": 0.035,
            "elimination_penalty": 1.76,
            "win_bonus": 1.15,
            "uncontested_pot_bonus": 0.015,
            "learning_rate": 9.1e-5,
            "initial_ent_coef": 0.035,
            "min_ent_coef": 0.008,
            "clip_range": 0.22,
        },
        "hypothesis": "Tests if sustained high exploration discovers unconventional high-EV overbets and 3-bet squeezes.",
    },
]


def evaluate_probe_model(
    model: MaskablePPO,
    n_sessions: int = 25,
    base_seed: int = 1042,
    league_pool: Optional[HoldemLeaguePool] = None,
) -> Dict[str, float]:
    """Evaluates a trained Model F probe model over Common Random Number (CRN) tournament sessions."""
    wins = 0
    eliminations = 0
    total_chips_delta = 0
    vpip_count = 0
    total_hands = 0
    total_actions = 0
    uncontested_wins = 0

    action_counts = {a: 0 for a in range(NUM_ACTIONS_HOLDEM_F)}

    for s_idx in range(n_sessions):
        session_seed = base_seed + s_idx * 7919
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
            obs_version="f",
            league_mode=True,
            league_pool=league_pool,
            rng_seed=session_seed,
        )

        obs, info = env.reset()
        done = False
        start_chips = info.get("hero_chips", 200)

        while not done:
            mask = env.action_masks()
            act, _ = model.predict(obs, action_masks=mask, deterministic=True)
            action = int(act)

            total_actions += 1
            if action in action_counts:
                action_counts[action] += 1

            if env.env.street == STREET_PREFLOP:
                if action >= 2 or (
                    action == A_F_CHECK_CALL and env.env.seats[0].total_invested > env.env.big_blind
                ):
                    vpip_count += 1
                total_hands += 1

            obs, rew, term, trunc, info = env.step(action)
            done = term or trunc

        if info.get("is_winner", False):
            wins += 1
        if env.env.last_won_uncontested and env.env.last_winner_seat == 0:
            uncontested_wins += 1
        if env.env.seats[0].chips <= 0:
            eliminations += 1
        total_chips_delta += (env.env.seats[0].chips - start_chips)

    win_rate = (wins / float(n_sessions)) * 100.0
    bb_return = (total_chips_delta / 2.0) / float(n_sessions)
    elim_rate = (eliminations / float(n_sessions)) * 100.0
    vpip_rate = (vpip_count / max(1, total_hands)) * 100.0

    act_denom = max(1, total_actions)
    third_pot_freq = (action_counts[A_F_THIRD_POT] / act_denom) * 100.0
    half_pot_freq = (action_counts[A_F_HALF_POT] / act_denom) * 100.0
    pot_freq = (action_counts[A_F_POT] / act_denom) * 100.0
    overbet_freq = (action_counts[A_F_OVERBET] / act_denom) * 100.0
    all_in_freq = (action_counts[A_F_ALL_IN] / act_denom) * 100.0

    return {
        "win_rate": win_rate,
        "bb_return": bb_return,
        "elim_rate": elim_rate,
        "vpip_rate": vpip_rate,
        "third_pot_freq": third_pot_freq,
        "half_pot_freq": half_pot_freq,
        "pot_freq": pot_freq,
        "overbet_freq": overbet_freq,
        "all_in_freq": all_in_freq,
        "uncontested_wins": uncontested_wins,
    }


def run_boundary_trial(
    params: Dict[str, Any],
    timesteps: int = 250_000,
    n_envs: int = 12,
    seed: int = 42,
    eval_sessions: int = 25,
    use_dummy_vec: bool = False,
):
    """Trains a Model F probe model on P-cores and evaluates tournament behavior."""
    vec_env_cls = DummyVecEnv if use_dummy_vec else SubprocVecEnv

    league_pool = HoldemLeaguePool(
        league_dir="models/holdem/league",
        anchor_model_path="models/holdem/model_holdem_e_best.zip",
    )

    vec_env = vec_env_cls([
        make_holdem_f_env(
            i,
            seed=seed,
            league_pool=league_pool,
            fold_penalty=params["fold_penalty"],
            elimination_penalty=params["elimination_penalty"],
            win_bonus=params["win_bonus"],
            icm_equity_weight=params["icm_equity_weight"],
            pot_control_bonus=params["pot_control_bonus"],
            exploit_value_bonus=params["exploit_value_bonus"],
            exploit_steal_bonus=params["exploit_steal_bonus"],
            bluff_station_penalty=params["bluff_station_penalty"],
            uncontested_pot_bonus=params.get("uncontested_pot_bonus", 0.015),
        ) for i in range(n_envs)
    ])

    lr_schedule = linear_warmup_cosine_decay(
        initial_lr=params["learning_rate"],
        min_lr=params["learning_rate"] * 0.20,
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
            game_features_dim=104,  # 168 - 64 = 104
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
        clip_range=params["clip_range"],
        ent_coef=params["initial_ent_coef"],
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )

    belief_head = HoldemAuxiliaryBeliefHead(feature_dim=64 + 104, hidden_dim=128)
    belief_callback = HoldemAuxiliaryBeliefCallback(
        belief_head=belief_head,
        lr=params["learning_rate"] * 0.8,
        tier_weight=0.5,
        fold_weight=0.5,
        max_batch_size=2048,
        verbose=0,
    )

    curriculum_callback = HoldemCurriculumRewardCallback(
        scaffold_steps=params["scaffold_steps"],
        total_timesteps=timesteps,
        min_shaping_mult=params["min_shaping_mult"],
        log_freq=50_000,
        verbose=0,
    )

    entropy_callback = HoldemEntropyDecayCallback(
        initial_ent=params["initial_ent_coef"],
        min_ent=params["min_ent_coef"],
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

    eval_res = evaluate_probe_model(model, n_sessions=eval_sessions, base_seed=seed + 1000, league_pool=league_pool)

    # Multi-objective composite score emphasizing win rate, BB return, and survival
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
    1. Identifies failure modes and transition points.
    2. Derives mathematically grounded tightened bounds for full HPO.
    3. Saves machine-readable bounds JSON and report.
    """
    sorted_results = sorted(results, key=lambda r: r["score"], reverse=True)
    best_probe = sorted_results[0]
    baseline_probe = next((r for r in results if r["name"] == "Model F Baseline (Model E Params)"), sorted_results[0])

    # Default screened bounds for Model F
    bounds = {
        "scaffold_steps_min": 60_000,
        "scaffold_steps_max": 160_000,
        "min_shaping_mult_min": 0.0,
        "min_shaping_mult_max": 0.02,
        "uncontested_pot_bonus_min": 0.010,
        "uncontested_pot_bonus_max": 0.050,
        "exploit_value_bonus_min": 0.05,
        "exploit_value_bonus_max": 0.16,
        "exploit_steal_bonus_min": 0.025,
        "exploit_steal_bonus_max": 0.065,
        "bluff_station_penalty_min": 0.06,
        "bluff_station_penalty_max": 0.18,
        "icm_equity_weight_min": 0.30,
        "icm_equity_weight_max": 0.95,
        "fold_penalty_min": -0.015,
        "fold_penalty_max": -0.002,
        "pot_control_bonus_min": 0.015,
        "pot_control_bonus_max": 0.045,
        "learning_rate_min": 6.0e-5,
        "learning_rate_max": 1.4e-4,
        "initial_ent_coef_min": 0.020,
        "initial_ent_coef_max": 0.035,
        "min_ent_coef_min": 0.003,
        "min_ent_coef_max": 0.007,
        "clip_range_min": 0.18,
        "clip_range_max": 0.24,
        "elimination_penalty_min": 1.2,
        "elimination_penalty_max": 3.5,
        "win_bonus_min": 1.0,
        "win_bonus_max": 2.2,
    }

    # Dynamic adjustments based on probe performance
    high_steal = next((r for r in results if "High Steal" in r["name"]), None)
    zero_steal = next((r for r in results if "Zero Steal" in r["name"]), None)
    if high_steal and zero_steal:
        if high_steal["score"] > zero_steal["score"]:
            bounds["uncontested_pot_bonus_min"] = 0.020
            bounds["uncontested_pot_bonus_max"] = 0.060
        else:
            bounds["uncontested_pot_bonus_min"] = 0.005
            bounds["uncontested_pot_bonus_max"] = 0.030

    fast_anneal = next((r for r in results if "Fast Curriculum" in r["name"]), None)
    ext_scaffold = next((r for r in results if "Extended Scaffolding" in r["name"]), None)
    if fast_anneal and ext_scaffold:
        if fast_anneal["score"] > ext_scaffold["score"]:
            bounds["scaffold_steps_min"] = 50_000
            bounds["scaffold_steps_max"] = 120_000
        else:
            bounds["scaffold_steps_min"] = 90_000
            bounds["scaffold_steps_max"] = 180_000

    low_icm = next((r for r in results if "Low ICM" in r["name"]), None)
    high_icm = next((r for r in results if "High ICM" in r["name"]), None)
    if low_icm and high_icm:
        if low_icm["eval"]["win_rate"] > high_icm["eval"]["win_rate"]:
            bounds["icm_equity_weight_min"] = 0.20
            bounds["icm_equity_weight_max"] = 0.70
            bounds["elimination_penalty_min"] = 1.0
            bounds["elimination_penalty_max"] = 2.5

    os.makedirs(os.path.dirname(output_bounds_path), exist_ok=True)
    with open(output_bounds_path, "w") as f:
        json.dump(bounds, f, indent=2)

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    report = {
        "best_probe": best_probe,
        "baseline_probe": baseline_probe,
        "all_probes": sorted_results,
        "screened_bounds": bounds,
    }
    with open(output_report_path, "w") as f:
        json.dump(report, f, indent=2)

    return bounds


def run_boundary_search_holdem_f(
    timesteps: int = 250_000,
    n_envs: int = 12,
    seed: int = 42,
    eval_sessions: int = 25,
):
    pin_to_pcores(n_envs)
    print("\n" + "=" * 80)
    print(f"      CARDSHARK-RL: TEXAS HOLD'EM MODEL F BOUNDARY SEARCH ({len(BOUNDARY_PROBE_POINTS)} PROBES)")
    print("=" * 80)
    print(f"  Probe Horizon:             {timesteps:,} steps per probe")
    print(f"  Parallel Workers:          {n_envs} P-cores")
    print(f"  Evaluation Sessions:       {eval_sessions} CRN tournament sessions")
    print(f"  Total Probes:              {len(BOUNDARY_PROBE_POINTS)}")
    print("=" * 80 + "\n")

    results = []
    t_start = time.time()

    for idx, probe in enumerate(BOUNDARY_PROBE_POINTS):
        name = probe["probe_name"]
        params = probe["params"]
        print(f"\n[{idx+1}/{len(BOUNDARY_PROBE_POINTS)}] Running Probe: {name}")
        print(f"  Hypothesis: {probe['hypothesis']}")

        score, eval_res, dur = run_boundary_trial(
            params=params,
            timesteps=timesteps,
            n_envs=n_envs,
            seed=seed + idx * 100,
            eval_sessions=eval_sessions,
        )

        res_entry = {
            "probe_idx": idx,
            "name": name,
            "score": score,
            "eval": eval_res,
            "params": params,
            "duration_sec": dur,
        }
        results.append(res_entry)

        print(
            f"  Result -> Win Rate: {eval_res['win_rate']:5.1f}% | "
            f"BB Return: {eval_res['bb_return']:+6.1f} | "
            f"Elim Rate: {eval_res['elim_rate']:4.1f}% | "
            f"33% Sizing: {eval_res['third_pot_freq']:4.1f}% | "
            f"Overbet: {eval_res['overbet_freq']:4.1f}% | "
            f"Score: {score:6.2f} (Duration: {dur:.1f}s)"
        )

    total_dur = time.time() - t_start
    print(f"\n[Completed] {len(BOUNDARY_PROBE_POINTS)} Boundary Probes finished in {total_dur/60:.2f} mins.")

    bounds = analyze_boundary_results(results)
    print(f"\n[Screened Bounds Synthesized] Saved to -> {SCREENED_BOUNDS_PATH}")
    print(f"[Boundary Report Synthesized] Saved to -> {REPORT_PATH}")
    return bounds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boundary Search for Texas Hold'em Model F")
    parser.add_argument("--timesteps", type=int, default=250_000, help="Steps per probe")
    parser.add_argument("--n-envs", type=int, default=12, help="Number of worker envs")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--eval-sessions", type=int, default=25, help="CRN evaluation sessions per probe")

    args = parser.parse_args()
    run_boundary_search_holdem_f(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        eval_sessions=args.eval_sessions,
    )
