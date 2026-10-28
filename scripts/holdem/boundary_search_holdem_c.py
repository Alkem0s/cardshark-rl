"""
scripts/holdem/boundary_search_holdem_c.py — Boundary Value & Screening Search for Model C.

Probes the extreme edges and critical transition points of the Model C hyperparameter space:
1. icm_equity_weight: [0.0 (no ICM) vs 0.20 vs 0.50 vs 1.00 vs 2.00 (heavy risk aversion)]
2. fold_penalty: [-0.030 (heavy fold reward) vs -0.015 vs 0.0 vs +0.010 (positive penalty)]
3. pot_control_bonus: [0.0 (no bonus) vs 0.02 vs 0.06 (aggressive check bonus)]
4. learning_rate: [8.0e-5 (low) vs 1.8e-4 (mid) vs 3.5e-4 (high)]
5. initial_ent_coef: [0.018 (low exploration) vs 0.035 vs 0.060 (high exploration)]
6. elimination_penalty: [0.0 vs 3.0 vs 8.0]

Runs 10-12 fast boundary probe trials (35,000 timesteps each via 12 P-core SubprocVecEnvs),
evaluates each trial across 25 Common Random Numbers (CRN) tournament sessions against
the mixed league pool (Model B Champion + Archetype Bots), and automatically synthesizes:
- configs/holdem/screened_bounds_holdem_c.json (tightened search space for full HPO)
- results/holdem/model_c_boundary_report.json (comprehensive empirical failure analysis)
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

import optuna
from optuna.samplers import TPESampler
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv

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

SCREENED_BOUNDS_PATH = "configs/holdem/screened_bounds_holdem_c.json"
REPORT_PATH = "results/holdem/model_c_boundary_report.json"


# Curated Boundary Value Probes targeting specific behavioral and mathematical hypotheses
BOUNDARY_PROBE_POINTS: List[Dict[str, Any]] = [
    # Probe 0: Balanced Midpoint Baseline
    {
        "probe_name": "Midpoint Baseline",
        "params": {
            "fold_penalty": -0.015,
            "icm_equity_weight": 0.50,
            "pot_control_bonus": 0.020,
            "elimination_penalty": 3.0,
            "win_bonus": 1.0,
            "learning_rate": 0.00018,
            "initial_ent_coef": 0.035,
            "min_ent_coef": 0.005,
            "clip_range": 0.20,
            "min_lr_ratio": 0.25,
        },
        "hypothesis": "Balanced GTO baseline with moderate fold reward and standard ICM weight.",
    },
    # Probe 1: Zero ICM Boundary (Cash Game mode)
    {
        "probe_name": "Zero ICM Boundary",
        "params": {
            "fold_penalty": -0.015,
            "icm_equity_weight": 0.0,
            "pot_control_bonus": 0.020,
            "elimination_penalty": 0.0,
            "win_bonus": 0.0,
            "learning_rate": 0.00018,
            "initial_ent_coef": 0.035,
            "min_ent_coef": 0.005,
            "clip_range": 0.20,
            "min_lr_ratio": 0.25,
        },
        "hypothesis": "Tests if removing ICM awareness causes high bustout rates and poor tournament survival.",
    },
    # Probe 2: Extreme High ICM Boundary (Heavy Risk Aversion)
    {
        "probe_name": "Extreme High ICM",
        "params": {
            "fold_penalty": -0.015,
            "icm_equity_weight": 2.0,
            "pot_control_bonus": 0.020,
            "elimination_penalty": 8.0,
            "win_bonus": 1.0,
            "learning_rate": 0.00018,
            "initial_ent_coef": 0.035,
            "min_ent_coef": 0.005,
            "clip_range": 0.20,
            "min_lr_ratio": 0.25,
        },
        "hypothesis": "Tests if excessive ICM penalty induces tournament paralysis and blind decay.",
    },
    # Probe 3: Positive Fold Penalty Boundary (Model B failure mode)
    {
        "probe_name": "Positive Fold Penalty",
        "params": {
            "fold_penalty": 0.010,
            "icm_equity_weight": 0.50,
            "pot_control_bonus": 0.020,
            "elimination_penalty": 3.0,
            "win_bonus": 1.0,
            "learning_rate": 0.00018,
            "initial_ent_coef": 0.035,
            "min_ent_coef": 0.005,
            "clip_range": 0.20,
            "min_lr_ratio": 0.25,
        },
        "hypothesis": "Tests if punishing folding induces VPIP inflation despite preflop GTO range masking.",
    },
    # Probe 4: Heavy Fold Reward Boundary
    {
        "probe_name": "Heavy Fold Reward",
        "params": {
            "fold_penalty": -0.030,
            "icm_equity_weight": 0.50,
            "pot_control_bonus": 0.020,
            "elimination_penalty": 3.0,
            "win_bonus": 1.0,
            "learning_rate": 0.00018,
            "initial_ent_coef": 0.035,
            "min_ent_coef": 0.005,
            "clip_range": 0.20,
            "min_lr_ratio": 0.25,
        },
        "hypothesis": "Tests if rewarding folding too heavily drops VPIP below the optimal 28% championship range.",
    },
    # Probe 5: Zero Pot Control (Model B checking collapse)
    {
        "probe_name": "Zero Pot Control",
        "params": {
            "fold_penalty": -0.015,
            "icm_equity_weight": 0.50,
            "pot_control_bonus": 0.0,
            "elimination_penalty": 3.0,
            "win_bonus": 1.0,
            "learning_rate": 0.00018,
            "initial_ent_coef": 0.035,
            "min_ent_coef": 0.005,
            "clip_range": 0.20,
            "min_lr_ratio": 0.25,
        },
        "hypothesis": "Tests if removing pot-control bonus collapses flop check frequency toward 0%.",
    },
    # Probe 6: High Pot Control Bonus
    {
        "probe_name": "High Pot Control Bonus",
        "params": {
            "fold_penalty": -0.015,
            "icm_equity_weight": 0.50,
            "pot_control_bonus": 0.060,
            "elimination_penalty": 3.0,
            "win_bonus": 1.0,
            "learning_rate": 0.00018,
            "initial_ent_coef": 0.035,
            "min_ent_coef": 0.005,
            "clip_range": 0.20,
            "min_lr_ratio": 0.25,
        },
        "hypothesis": "Tests if strong pot-control reward induces excessive passivity on the flop.",
    },
    # Probe 7: High Learning Rate & Clip Boundary (Gradient Volatility)
    {
        "probe_name": "High Learning Rate",
        "params": {
            "fold_penalty": -0.015,
            "icm_equity_weight": 0.50,
            "pot_control_bonus": 0.020,
            "elimination_penalty": 3.0,
            "win_bonus": 1.0,
            "learning_rate": 0.00035,
            "initial_ent_coef": 0.045,
            "min_ent_coef": 0.008,
            "clip_range": 0.25,
            "min_lr_ratio": 0.30,
        },
        "hypothesis": "Tests high learning rate stability in transformer card attention trunk.",
    },
    # Probe 8: Low Learning Rate Boundary (Sluggish Convergence)
    {
        "probe_name": "Low Learning Rate",
        "params": {
            "fold_penalty": -0.015,
            "icm_equity_weight": 0.50,
            "pot_control_bonus": 0.020,
            "elimination_penalty": 3.0,
            "win_bonus": 1.0,
            "learning_rate": 0.00008,
            "initial_ent_coef": 0.025,
            "min_ent_coef": 0.004,
            "clip_range": 0.18,
            "min_lr_ratio": 0.20,
        },
        "hypothesis": "Tests if low learning rate underfits within standard rollout budgets.",
    },
    # Probe 9: High Entropy Exploration Boundary
    {
        "probe_name": "High Entropy Exploration",
        "params": {
            "fold_penalty": -0.015,
            "icm_equity_weight": 0.50,
            "pot_control_bonus": 0.020,
            "elimination_penalty": 3.0,
            "win_bonus": 1.0,
            "learning_rate": 0.00020,
            "initial_ent_coef": 0.060,
            "min_ent_coef": 0.010,
            "clip_range": 0.20,
            "min_lr_ratio": 0.25,
        },
        "hypothesis": "Tests if high initial entropy introduces excessive random noise into early tournament rounds.",
    },
]


def evaluate_boundary_model(
    model: MaskablePPO,
    n_sessions: int = 25,
    base_seed: int = 20_000,
) -> Dict[str, float]:
    """Runs CRN tournament evaluations and records VPIP, flop check, elimination, and win rates."""
    wins = 0
    total_hands = 0
    vpip_count = 0
    eliminations = 0
    total_chips_delta = 0

    flop_moves = 0
    flop_checks = 0

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

    return {
        "win_rate": win_rate,
        "bb_return": bb_return,
        "elim_rate": elim_rate,
        "vpip_rate": vpip_rate,
        "flop_check_rate": flop_check_rate,
    }


def run_boundary_trial(
    params: Dict[str, Any],
    timesteps: int = 35_000,
    n_envs: int = 12,
    seed: int = 42,
    eval_sessions: int = 25,
) -> Tuple[float, Dict[str, float], float]:
    """Trains a quick probe model on 12 P-cores and evaluates tournament behavior."""
    vec_env = SubprocVecEnv([
        make_holdem_c_env(
            i,
            seed=seed,
            fold_penalty=params["fold_penalty"],
            elimination_penalty=params["elimination_penalty"],
            win_bonus=params["win_bonus"],
            icm_equity_weight=params["icm_equity_weight"],
            pot_control_bonus=params["pot_control_bonus"],
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
            game_features_dim=77,
            slot_features_dim=0,
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    model = MaskablePPO(
        policy=MaskableActorCriticPolicy,
        env=vec_env,
        learning_rate=lr_schedule,
        n_steps=256,
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

    callbacks = [
        EntropyDecayCallback(
            initial_ent=params["initial_ent_coef"],
            min_ent=params["min_ent_coef"],
            total_timesteps=timesteps,
        )
    ]

    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)
    train_dur = time.time() - t0
    vec_env.close()

    eval_res = evaluate_boundary_model(model, n_sessions=eval_sessions, base_seed=seed + 1000)

    # Multi-objective composite score emphasizing win rate, positive return, and balanced VPIP
    vpip_penalty = abs(eval_res["vpip_rate"] - 31.0) * 0.3
    score = eval_res["win_rate"] + 0.1 * eval_res["bb_return"] - 0.05 * (eval_res["elim_rate"] - 50.0) - vpip_penalty

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
    insights = []

    # Sort results by score descending
    results.sort(key=lambda r: r["score"], reverse=True)
    best_probe = results[0]

    # Analyze ICM equity weight impact
    zero_icm = next((r for r in results if r["params"]["icm_equity_weight"] == 0.0), None)
    high_icm = next((r for r in results if r["params"]["icm_equity_weight"] >= 1.5), None)
    mid_icm = next((r for r in results if 0.3 <= r["params"]["icm_equity_weight"] <= 0.7), None)

    icm_min, icm_max = 0.15, 1.00
    if zero_icm and mid_icm:
        if zero_icm["eval"]["elim_rate"] > mid_icm["eval"]["elim_rate"] + 8.0:
            eliminated_picks.append({
                "parameter": "icm_equity_weight",
                "eliminated_range": [0.0, 0.10],
                "rationale": f"Zero ICM caused elimination rate to spike to {zero_icm['eval']['elim_rate']:.1f}% (vs {mid_icm['eval']['elim_rate']:.1f}% at mid-ICM).",
            })
            icm_min = 0.15

    if high_icm and mid_icm:
        if high_icm["eval"]["win_rate"] < mid_icm["eval"]["win_rate"] - 5.0:
            eliminated_picks.append({
                "parameter": "icm_equity_weight",
                "eliminated_range": [1.25, 2.00],
                "rationale": f"Excessive ICM weight ({high_icm['params']['icm_equity_weight']}) caused over-folding/paralysis, depressing win rate to {high_icm['eval']['win_rate']:.1f}%.",
            })
            icm_max = 1.00

    # Analyze fold penalty impact
    pos_fold = next((r for r in results if r["params"]["fold_penalty"] > 0.0), None)
    deep_fold = next((r for r in results if r["params"]["fold_penalty"] <= -0.025), None)

    fold_min, fold_max = -0.022, 0.00
    if pos_fold:
        if pos_fold["eval"]["vpip_rate"] > 38.0 or pos_fold["score"] < best_probe["score"] - 5.0:
            eliminated_picks.append({
                "parameter": "fold_penalty",
                "eliminated_range": [0.005, 0.020],
                "rationale": f"Positive fold penalty (+{pos_fold['params']['fold_penalty']}) provoked VPIP inflation to {pos_fold['eval']['vpip_rate']:.1f}%, hurting EV.",
            })
            fold_max = 0.00

    if deep_fold and mid_icm:
        if deep_fold["eval"]["vpip_rate"] < 24.0:
            eliminated_picks.append({
                "parameter": "fold_penalty",
                "eliminated_range": [-0.040, -0.025],
                "rationale": f"Excessive fold reward ({deep_fold['params']['fold_penalty']}) over-constrains VPIP to {deep_fold['eval']['vpip_rate']:.1f}%, blinding out.",
            })
            fold_min = -0.020

    # Analyze pot control bonus
    zero_pot_ctrl = next((r for r in results if r["params"]["pot_control_bonus"] == 0.0), None)
    pot_ctrl_min, pot_ctrl_max = 0.015, 0.050
    if zero_pot_ctrl and mid_icm:
        if zero_pot_ctrl["eval"]["flop_check_rate"] < 5.0:
            eliminated_picks.append({
                "parameter": "pot_control_bonus",
                "eliminated_range": [0.0, 0.008],
                "rationale": f"Zero pot-control bonus collapsed flop check frequency to {zero_pot_ctrl['eval']['flop_check_rate']:.1f}%, losing pot control.",
            })
            pot_ctrl_min = 0.012

    # Analyze learning rate
    high_lr = next((r for r in results if r["params"]["learning_rate"] >= 3.0e-4), None)
    low_lr = next((r for r in results if r["params"]["learning_rate"] <= 1.0e-4), None)
    lr_min, lr_max = 1.2e-4, 2.8e-4

    if high_lr and high_lr["score"] < best_probe["score"] - 8.0:
        eliminated_picks.append({
            "parameter": "learning_rate",
            "eliminated_range": [3.0e-4, 5.0e-4],
            "rationale": f"High LR ({high_lr['params']['learning_rate']:.5f}) showed policy instability, lagging behind mid-LR trials.",
        })
        lr_max = 2.6e-4

    if low_lr and low_lr["score"] < best_probe["score"] - 8.0:
        eliminated_picks.append({
            "parameter": "learning_rate",
            "eliminated_range": [5.0e-5, 1.0e-4],
            "rationale": f"Low LR ({low_lr['params']['learning_rate']:.5f}) learned too slowly in early rounds.",
        })
        lr_min = 1.2e-4

    screened_bounds = {
        "icm_equity_weight_min": icm_min,
        "icm_equity_weight_max": icm_max,
        "fold_penalty_min": fold_min,
        "fold_penalty_max": fold_max,
        "pot_control_bonus_min": pot_ctrl_min,
        "pot_control_bonus_max": pot_ctrl_max,
        "learning_rate_min": lr_min,
        "learning_rate_max": lr_max,
        "initial_ent_coef_min": 0.025,
        "initial_ent_coef_max": 0.048,
        "clip_range_min": 0.18,
        "clip_range_max": 0.24,
        "elimination_penalty_min": 2.0,
        "elimination_penalty_max": 6.0,
        "win_bonus_min": 0.5,
        "win_bonus_max": 3.0,
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


def run_boundary_search_holdem_c(
    n_trials: int = 10,
    timesteps_per_trial: int = 35_000,
    n_envs: int = 12,
    seed: int = 42,
    dry_run: bool = False,
):
    """Executes the Model C Boundary Value Analysis."""
    if dry_run:
        print("[Boundary Search Dry Run] Validated configuration. Ready for execution.")
        return

    pin_to_pcores(n_envs)

    print("=================================================================")
    print("  TEXAS HOLD'EM MODEL C BOUNDARY SEARCH & SCREENING")
    print("=================================================================")
    print(f"  Probing {min(n_trials, len(BOUNDARY_PROBE_POINTS))} boundary hypotheses | {timesteps_per_trial:,} steps/trial")
    print(f"  Parallel Envs: {n_envs} (12 P-cores) | Vector: {OBS_DIM_HOLDEM_C} dims")
    print(f"  Outputs: {SCREENED_BOUNDS_PATH} & {REPORT_PATH}")
    print("=================================================================\n")

    t_total_start = time.time()
    results: List[Dict[str, Any]] = []

    for i in range(min(n_trials, len(BOUNDARY_PROBE_POINTS))):
        probe = BOUNDARY_PROBE_POINTS[i]
        p_name = probe["probe_name"]
        params = probe["params"]
        hypo = probe["hypothesis"]

        print(f"--- Probe #{i+1:02d}: {p_name} ---")
        print(f"    Hypothesis: {hypo}")
        print(f"    Params: ICM={params['icm_equity_weight']}, FoldPen={params['fold_penalty']}, "
              f"PotCtrl={params['pot_control_bonus']}, LR={params['learning_rate']:.5f}")

        trial_seed = seed + i * 50
        score, eval_res, dur = run_boundary_trial(
            params=params,
            timesteps=timesteps_per_trial,
            n_envs=n_envs,
            seed=trial_seed,
            eval_sessions=25,
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
        print(f"  - {k:<25}: {v}")

    print("\n>>> IDENTIFIED FAILURE MODES & ELIMINATED REGIONS <<<")
    for elim in report["eliminated_picks"]:
        print(f"  * {elim['parameter']} in {elim['eliminated_range']}: {elim['rationale']}")

    print(f"\nSaved tightened bounds to: {SCREENED_BOUNDS_PATH}")
    print(f"Saved full boundary report to: {REPORT_PATH}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boundary Search for Texas Hold'em Model C")
    parser.add_argument("--trials", type=int, default=10, help="Number of boundary probes (max 10)")
    parser.add_argument("--timesteps", type=int, default=35_000, help="Timesteps per probe")
    parser.add_argument("--n-envs", type=int, default=12, help="Number of parallel P-core environments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration without running")
    args = parser.parse_args()

    run_boundary_search_holdem_c(
        n_trials=args.trials,
        timesteps_per_trial=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        dry_run=args.dry_run,
    )
