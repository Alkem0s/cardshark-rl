"""
scripts/holdem/boundary_search_holdem_e.py — Boundary Value & Screening Search for Texas Hold'em Model E.
The Co-Evolutionary Champion.

Probes the extreme edges and critical transition points of Model E:
1. scaffold_steps: [100k (fast transition) vs 200k (extended scaffolding)]
2. min_shaping_mult: [0.0 (pure chip-EV) vs 0.20 (residual shaping prior)]
3. exploit_value_bonus: [0.03 (mild) vs 0.07 (mid) vs 0.12 (hyper-aggressive value)]
4. exploit_steal_bonus: [0.02 (mild) vs 0.04 (mid) vs 0.07 (aggressive stealing)]
5. bluff_station_penalty: [0.06 (mild) vs 0.14 (mid) vs 0.25 (heavy punting penalty)]
6. icm_equity_weight: [0.20 (cash game aggression) vs 0.75 (mid) vs 1.20 (high risk aversion)]
7. elimination_penalty: [1.5 (shallow) vs 3.14 (mid) vs 5.5 (deep survival)]
8. win_bonus: [1.0 vs 1.5 vs 3.0]
9. learning_rate: [4.0e-5 (low) vs 7.5e-5 (mid) vs 1.4e-4 (high)]
10. initial_ent_coef: [0.018 (disciplined) vs 0.025 (mid) vs 0.040 (high exploration)]
11. min_ent_coef: [0.004 (deep exploitation) vs 0.006 (mid) vs 0.010 (fluid)]
12. clip_range: [0.18 vs 0.20 vs 0.24]

Runs 12 curated boundary probe trials (250,000 timesteps each via 12 P-core SubprocVecEnvs),
evaluates each trial across 25 Common Random Numbers (CRN) tournament sessions against
the HoldemLeaguePool (Model D Champion + Model E Snapshots + Jittered Archetype Bots), and synthesizes:
- configs/holdem/screened_bounds_holdem_e.json (tightened search space for full HPO)
- results/holdem/model_e_boundary_report.json (empirical failure analysis & trade-off synthesis)
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
REPORT_PATH = os.path.join(PROJECT_ROOT, "results/holdem/model_e_boundary_report.json")


# 12 Curated Boundary Value Probes targeting specific behavioral and mathematical hypotheses for Model E
BOUNDARY_PROBE_POINTS: List[Dict[str, Any]] = [
    # Probe 0: Balanced Model E Baseline
    {
        "probe_name": "Model E Baseline",
        "params": {
            "scaffold_steps": 125_000,
            "min_shaping_mult": 0.0,
            "exploit_value_bonus": 0.07,
            "exploit_steal_bonus": 0.04,
            "bluff_station_penalty": 0.14,
            "icm_equity_weight": 0.75,
            "fold_penalty": -0.015,
            "pot_control_bonus": 0.025,
            "elimination_penalty": 3.14,
            "win_bonus": 1.50,
            "learning_rate": 7.5e-5,
            "initial_ent_coef": 0.025,
            "min_ent_coef": 0.006,
            "clip_range": 0.20,
        },
        "hypothesis": "Balanced Model E baseline with two-stage curriculum and self-play league sparring.",
    },
    # Probe 1: Fast Curriculum Annealing (Rapid Pure Equity Transition)
    {
        "probe_name": "Fast Curriculum Anneal (75k Steps)",
        "params": {
            "scaffold_steps": 75_000,
            "min_shaping_mult": 0.0,
            "exploit_value_bonus": 0.07,
            "exploit_steal_bonus": 0.04,
            "bluff_station_penalty": 0.14,
            "icm_equity_weight": 0.75,
            "fold_penalty": -0.015,
            "pot_control_bonus": 0.025,
            "elimination_penalty": 3.14,
            "win_bonus": 1.50,
            "learning_rate": 7.5e-5,
            "initial_ent_coef": 0.025,
            "min_ent_coef": 0.006,
            "clip_range": 0.20,
        },
        "hypothesis": "Tests whether early transition (75k steps) to pure chip EV produces higher unconstrained performance.",
    },
    # Probe 2: Extended Scaffolding (Conservative Guidance)
    {
        "probe_name": "Extended Scaffolding (180k Steps)",
        "params": {
            "scaffold_steps": 180_000,
            "min_shaping_mult": 0.0,
            "exploit_value_bonus": 0.07,
            "exploit_steal_bonus": 0.04,
            "bluff_station_penalty": 0.14,
            "icm_equity_weight": 0.75,
            "fold_penalty": -0.015,
            "pot_control_bonus": 0.025,
            "elimination_penalty": 3.14,
            "win_bonus": 1.50,
            "learning_rate": 7.5e-5,
            "initial_ent_coef": 0.025,
            "min_ent_coef": 0.006,
            "clip_range": 0.20,
        },
        "hypothesis": "Tests if retaining reward shaping throughout 72% of the 250k horizon stabilizes fundamentals.",
    },
    # Probe 3: Non-Zero Residual Shaping Multiplier
    {
        "probe_name": "Residual Shaping Floor (0.20 Min Mult)",
        "params": {
            "scaffold_steps": 125_000,
            "min_shaping_mult": 0.20,
            "exploit_value_bonus": 0.07,
            "exploit_steal_bonus": 0.04,
            "bluff_station_penalty": 0.14,
            "icm_equity_weight": 0.75,
            "fold_penalty": -0.015,
            "pot_control_bonus": 0.025,
            "elimination_penalty": 3.14,
            "win_bonus": 1.50,
            "learning_rate": 7.5e-5,
            "initial_ent_coef": 0.025,
            "min_ent_coef": 0.006,
            "clip_range": 0.20,
        },
        "hypothesis": "Tests whether keeping a 20% residual shaping anchor prevents policy drift in self-play.",
    },
    # Probe 4: Low ICM Aggression Boundary
    {
        "probe_name": "Low ICM Boundary (0.20 Weight)",
        "params": {
            "scaffold_steps": 125_000,
            "min_shaping_mult": 0.0,
            "exploit_value_bonus": 0.07,
            "exploit_steal_bonus": 0.04,
            "bluff_station_penalty": 0.14,
            "icm_equity_weight": 0.20,
            "fold_penalty": -0.015,
            "pot_control_bonus": 0.025,
            "elimination_penalty": 1.50,
            "win_bonus": 1.50,
            "learning_rate": 7.5e-5,
            "initial_ent_coef": 0.025,
            "min_ent_coef": 0.006,
            "clip_range": 0.20,
        },
        "hypothesis": "Tests if low tournament risk aversion unlocks dominant aggressive stack accumulation like in Model D.",
    },
    # Probe 5: High ICM Risk Aversion Boundary
    {
        "probe_name": "High ICM Boundary (1.20 Weight)",
        "params": {
            "scaffold_steps": 125_000,
            "min_shaping_mult": 0.0,
            "exploit_value_bonus": 0.07,
            "exploit_steal_bonus": 0.04,
            "bluff_station_penalty": 0.14,
            "icm_equity_weight": 1.20,
            "fold_penalty": -0.015,
            "pot_control_bonus": 0.025,
            "elimination_penalty": 5.50,
            "win_bonus": 2.50,
            "learning_rate": 7.5e-5,
            "initial_ent_coef": 0.025,
            "min_ent_coef": 0.006,
            "clip_range": 0.20,
        },
        "hypothesis": "Tests if high ICM and survival penalty induce timid passivity vs neural opponents.",
    },
    # Probe 6: Hyper-Aggressive Exploit & Steal Boundary
    {
        "probe_name": "High Exploit & Steal (0.12 / 0.07)",
        "params": {
            "scaffold_steps": 125_000,
            "min_shaping_mult": 0.0,
            "exploit_value_bonus": 0.12,
            "exploit_steal_bonus": 0.07,
            "bluff_station_penalty": 0.08,
            "icm_equity_weight": 0.75,
            "fold_penalty": -0.015,
            "pot_control_bonus": 0.025,
            "elimination_penalty": 3.14,
            "win_bonus": 1.50,
            "learning_rate": 7.5e-5,
            "initial_ent_coef": 0.025,
            "min_ent_coef": 0.006,
            "clip_range": 0.20,
        },
        "hypothesis": "Tests if high value extraction and blind stealing accelerate early tournament leads.",
    },
    # Probe 7: Heavy Anti-Punt Penalty Boundary
    {
        "probe_name": "Heavy Bluff Station Penalty (0.25)",
        "params": {
            "scaffold_steps": 125_000,
            "min_shaping_mult": 0.0,
            "exploit_value_bonus": 0.05,
            "exploit_steal_bonus": 0.03,
            "bluff_station_penalty": 0.25,
            "icm_equity_weight": 0.75,
            "fold_penalty": -0.015,
            "pot_control_bonus": 0.025,
            "elimination_penalty": 3.14,
            "win_bonus": 1.50,
            "learning_rate": 7.5e-5,
            "initial_ent_coef": 0.025,
            "min_ent_coef": 0.006,
            "clip_range": 0.20,
        },
        "hypothesis": "Tests if heavy bluff penalty eliminates punting into stations or causes under-bluffing against TAGs.",
    },
    # Probe 8: Low Learning Rate Boundary
    {
        "probe_name": "Low Learning Rate (4.0e-5)",
        "params": {
            "scaffold_steps": 125_000,
            "min_shaping_mult": 0.0,
            "exploit_value_bonus": 0.07,
            "exploit_steal_bonus": 0.04,
            "bluff_station_penalty": 0.14,
            "icm_equity_weight": 0.75,
            "fold_penalty": -0.015,
            "pot_control_bonus": 0.025,
            "elimination_penalty": 3.14,
            "win_bonus": 1.50,
            "learning_rate": 4.0e-5,
            "initial_ent_coef": 0.020,
            "min_ent_coef": 0.005,
            "clip_range": 0.18,
        },
        "hypothesis": "Tests if conservative gradient updates prevent policy divergence in self-play.",
    },
    # Probe 9: High Learning Rate Boundary
    {
        "probe_name": "High Learning Rate (1.4e-4)",
        "params": {
            "scaffold_steps": 125_000,
            "min_shaping_mult": 0.0,
            "exploit_value_bonus": 0.07,
            "exploit_steal_bonus": 0.04,
            "bluff_station_penalty": 0.14,
            "icm_equity_weight": 0.75,
            "fold_penalty": -0.015,
            "pot_control_bonus": 0.025,
            "elimination_penalty": 3.14,
            "win_bonus": 1.50,
            "learning_rate": 1.4e-4,
            "initial_ent_coef": 0.030,
            "min_ent_coef": 0.008,
            "clip_range": 0.24,
        },
        "hypothesis": "Tests if aggressive gradient updates accelerate co-evolution or cause policy instability.",
    },
    # Probe 10: High Entropy Exploration Boundary
    {
        "probe_name": "High Entropy Exploration (0.040 Init)",
        "params": {
            "scaffold_steps": 125_000,
            "min_shaping_mult": 0.0,
            "exploit_value_bonus": 0.07,
            "exploit_steal_bonus": 0.04,
            "bluff_station_penalty": 0.14,
            "icm_equity_weight": 0.75,
            "fold_penalty": -0.015,
            "pot_control_bonus": 0.025,
            "elimination_penalty": 3.14,
            "win_bonus": 1.50,
            "learning_rate": 7.5e-5,
            "initial_ent_coef": 0.040,
            "min_ent_coef": 0.010,
            "clip_range": 0.20,
        },
        "hypothesis": "Tests if wide action exploration discovers novel mixed strategies or increases variance.",
    },
    # Probe 11: Tight Disciplined Entropy Boundary
    {
        "probe_name": "Tight Disciplined Entropy (0.018 Init / 0.004 Min)",
        "params": {
            "scaffold_steps": 125_000,
            "min_shaping_mult": 0.0,
            "exploit_value_bonus": 0.07,
            "exploit_steal_bonus": 0.04,
            "bluff_station_penalty": 0.14,
            "icm_equity_weight": 0.75,
            "fold_penalty": -0.015,
            "pot_control_bonus": 0.025,
            "elimination_penalty": 3.14,
            "win_bonus": 1.50,
            "learning_rate": 6.5e-5,
            "initial_ent_coef": 0.018,
            "min_ent_coef": 0.004,
            "clip_range": 0.18,
        },
        "hypothesis": "Tests if razor-sharp deterministic play maximizes exploitation against known opponents.",
    },
]


def evaluate_probe_model(
    model: MaskablePPO,
    n_sessions: int = 25,
    base_seed: int = 20_000,
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


def run_boundary_trial(
    params: Dict[str, Any],
    timesteps: int = 250_000,
    n_envs: int = 12,
    seed: int = 42,
    eval_sessions: int = 25,
    use_dummy_vec: bool = False,
):
    """Trains a Model E probe model on P-cores and evaluates tournament behavior."""
    vec_env_cls = DummyVecEnv if use_dummy_vec else SubprocVecEnv

    league_pool = HoldemLeaguePool(
        league_dir="models/holdem/league",
        anchor_model_path="models/holdem/model_holdem_d_best.zip",
    )

    vec_env = vec_env_cls([
        make_holdem_e_env(
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
        clip_range=params["clip_range"],
        ent_coef=params["initial_ent_coef"],
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )

    belief_head = HoldemAuxiliaryBeliefHead(feature_dim=64 + 94, hidden_dim=128)
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
    baseline_probe = next((r for r in results if r["name"] == "Model E Baseline"), sorted_results[0])

    # Default screened bounds
    bounds = {
        "scaffold_steps_min": 75_000,
        "scaffold_steps_max": 180_000,
        "min_shaping_mult_min": 0.0,
        "min_shaping_mult_max": 0.15,
        "exploit_value_bonus_min": 0.04,
        "exploit_value_bonus_max": 0.11,
        "exploit_steal_bonus_min": 0.02,
        "exploit_steal_bonus_max": 0.06,
        "bluff_station_penalty_min": 0.07,
        "bluff_station_penalty_max": 0.22,
        "icm_equity_weight_min": 0.20,
        "icm_equity_weight_max": 0.95,
        "fold_penalty_min": -0.025,
        "fold_penalty_max": -0.005,
        "pot_control_bonus_min": 0.015,
        "pot_control_bonus_max": 0.040,
        "learning_rate_min": 5.0e-5,
        "learning_rate_max": 1.2e-4,
        "initial_ent_coef_min": 0.020,
        "initial_ent_coef_max": 0.035,
        "min_ent_coef_min": 0.004,
        "min_ent_coef_max": 0.009,
        "clip_range_min": 0.18,
        "clip_range_max": 0.24,
        "elimination_penalty_min": 1.5,
        "elimination_penalty_max": 4.5,
        "win_bonus_min": 1.0,
        "win_bonus_max": 2.5,
    }

    # Dynamic adjustments based on probe performance
    fast_anneal = next((r for r in results if r["name"] == "Fast Curriculum Anneal (75k Steps)"), None)
    ext_scaffold = next((r for r in results if r["name"] == "Extended Scaffolding (180k Steps)"), None)
    if fast_anneal and ext_scaffold:
        if fast_anneal["score"] > ext_scaffold["score"]:
            bounds["scaffold_steps_min"] = 60_000
            bounds["scaffold_steps_max"] = 140_000
        else:
            bounds["scaffold_steps_min"] = 100_000
            bounds["scaffold_steps_max"] = 200_000

    low_icm = next((r for r in results if r["name"] == "Low ICM Boundary (0.20 Weight)"), None)
    high_icm = next((r for r in results if r["name"] == "High ICM Boundary (1.20 Weight)"), None)
    if low_icm and high_icm:
        if low_icm["eval"]["win_rate"] > high_icm["eval"]["win_rate"]:
            bounds["icm_equity_weight_min"] = 0.15
            bounds["icm_equity_weight_max"] = 0.70
            bounds["elimination_penalty_min"] = 1.0
            bounds["elimination_penalty_max"] = 3.5

    os.makedirs(os.path.dirname(output_bounds_path), exist_ok=True)
    with open(output_bounds_path, "w") as f:
        json.dump(bounds, f, indent=2)

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    report = {
        "best_probe": best_probe,
        "all_probes": sorted_results,
        "screened_bounds": bounds,
    }
    with open(output_report_path, "w") as f:
        json.dump(report, f, indent=2)

    return bounds


def run_boundary_search_holdem_e(
    timesteps: int = 250_000,
    n_envs: int = 12,
    seed: int = 42,
    eval_sessions: int = 25,
):
    pin_to_pcores(n_envs)
    print("\n" + "=" * 80)
    print("      CARDSHARK-RL: TEXAS HOLD'EM MODEL E BOUNDARY SEARCH (12 PROBES)")
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
            f"Score: {score:6.2f} (Duration: {dur:.1f}s)"
        )

    total_dur = time.time() - t_start
    print(f"\n[Completed] 12 Boundary Probes finished in {total_dur/60:.2f} mins.")

    bounds = analyze_boundary_results(results)
    print(f"\n[Screened Bounds Synthesized] Saved to -> {SCREENED_BOUNDS_PATH}")
    print(f"[Boundary Report Synthesized] Saved to -> {REPORT_PATH}")
    return bounds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boundary Search for Texas Hold'em Model E")
    parser.add_argument("--timesteps", type=int, default=250_000, help="Steps per probe")
    parser.add_argument("--n-envs", type=int, default=12, help="Number of worker envs")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--eval-sessions", type=int, default=25, help="CRN evaluation sessions per probe")

    args = parser.parse_args()
    run_boundary_search_holdem_e(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        eval_sessions=args.eval_sessions,
    )
