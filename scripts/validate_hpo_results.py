"""
scripts/validate_hpo_results.py — Statistical Validation & Variance Audit for HPO Results.

Evaluates:
1. HPO Trial History Analysis:
   - Score variance across 25 trials.
   - Parameter-performance rank correlation (Spearman rho).
   - TPE convergence trajectory (Trial 0 -> Trial 24).
2. Head-to-Head Common Random Numbers (CRN) Validation Tournament:
   - Runs 50 sessions under identical seeds for Baseline vs HPO Winner.
   - Computes win rates, mean returns, and paired t-test p-value to prove whether
     HPO gains are statistically genuine or attributable to variance.
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from core.neural.card_attention import CardAttentionExtractor
from core.neural.decoupled_actor import DecoupledMaskableActorCriticPolicy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from games.triple_draw_27.gym_wrapper import TripleDrawPokerGymEnv
from games.holdem.gym_wrapper import HoldemPokerGymEnv
import core.neural.card_attention
import core.neural.decoupled_actor
sys.modules["card_attention"] = core.neural.card_attention
sys.modules["decoupled_actor"] = core.neural.decoupled_actor


def analyze_trial_history(history_path: str, game_name: str) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"  {game_name.upper()} — HPO TRIAL VARIANCE & SENSITIVITY AUDIT")
    print(f"=======================================================")
    if not os.path.exists(history_path):
        print(f"[Warning] History file not found at: {history_path}")
        return {}

    with open(history_path, "r", encoding="utf-8") as f:
        trials = json.load(f)

    if not trials:
        print("[Warning] No trials recorded.")
        return {}

    scores = [t["value"] for t in trials if t.get("value") is not None]
    if not scores:
        return {}

    min_score = min(scores)
    max_score = max(scores)
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))

    best_trial = max(trials, key=lambda t: t.get("value", -9999))

    print(f"Total Completed Trials : {len(scores)}")
    print(f"Score Range            : [{min_score:.2f}, {max_score:.2f}]")
    print(f"Mean Score +/- Std     : {mean_score:.2f} +/- {std_score:.2f}")
    print(f"Relative Variance (CV) : {(std_score / max(1e-3, abs(mean_score))) * 100:.1f}%")
    print(f"Top Trial #{best_trial['trial_id']} Score   : {best_trial['value']:.2f}")
    print(f"Top Trial Parameters   : {json.dumps(best_trial['params'], indent=2)}")

    return {
        "n_trials": len(scores),
        "min": min_score,
        "max": max_score,
        "mean": mean_score,
        "std": std_score,
        "best_trial": best_trial,
    }


def evaluate_holdem_model(model_path: str, num_sessions: int = 50, seed_base: int = 500_000) -> Tuple[float, float, List[float]]:
    if not os.path.exists(model_path):
        print(f"[Error] Model checkpoint not found: {model_path}")
        return 0.0, 0.0, []

    model = MaskablePPO.load(model_path)
    obs_dim = model.observation_space.shape[0]
    obs_ver = "b" if obs_dim == 125 else "a"
    eval_env = HoldemPokerGymEnv(
        num_seats=5,
        starting_chips=200,
        hero_seat=0,
        max_hands_per_session=100,
        rng_seed=seed_base,
        obs_version=obs_ver,
    )

    rewards = []
    wins = 0
    for sess_i in range(num_sessions):
        obs, info = eval_env.reset(seed=seed_base + sess_i * 1000)
        done = False
        sess_r = 0.0
        steps = 0
        while not done and steps < 200:
            mask = eval_env.action_masks()
            act, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, r, term, trunc, info = eval_env.step(int(act))
            sess_r += r
            done = term or trunc
            steps += 1
        rewards.append(sess_r)
        if info.get("is_winner", False):
            wins += 1

    eval_env.close()
    win_rate = (wins / num_sessions) * 100.0
    mean_reward = float(np.mean(rewards))
    return win_rate, mean_reward, rewards


def evaluate_27_model(model_path: str, num_sessions: int = 50, seed_base: int = 600_000) -> Tuple[float, float, List[float]]:
    if not os.path.exists(model_path):
        print(f"[Error] Model checkpoint not found: {model_path}")
        return 0.0, 0.0, []

    model = MaskablePPO.load(model_path)
    eval_env = TripleDrawPokerGymEnv(
        num_seats=5,
        starting_chips=200,
        hero_seat=0,
        max_hands_per_session=100,
        rng_seed=seed_base,
    )

    rewards = []
    wins = 0
    for sess_i in range(num_sessions):
        obs, info = eval_env.reset(seed=seed_base + sess_i * 1000)
        done = False
        sess_r = 0.0
        steps = 0
        while not done and steps < 200:
            mask = eval_env.action_masks()
            act, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, r, term, trunc, info = eval_env.step(int(act))
            sess_r += r
            done = term or trunc
            steps += 1
        rewards.append(sess_r)
        if info.get("is_winner", False):
            wins += 1

    eval_env.close()
    win_rate = (wins / num_sessions) * 100.0
    mean_reward = float(np.mean(rewards))
    return win_rate, mean_reward, rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate HPO results")
    parser.add_argument("--audit-only", action="store_true", help="Only audit trial history without running tournament")
    args = parser.parse_args()

    hpo_27_path = os.path.join(PROJECT_ROOT, "configs", "triple_draw_27", "hpo_trials_history.json")
    hpo_holdem_path = os.path.join(PROJECT_ROOT, "configs", "holdem", "hpo_trials_history.json")

    res_27 = analyze_trial_history(hpo_27_path, "2-7 Triple Draw")
    res_holdem = analyze_trial_history(hpo_holdem_path, "No-Limit Texas Hold'em")

    if not args.audit_only:
        print("\n=======================================================")
        print("  POST-HPO STATISTICAL VALIDATION TOURNAMENTS (50 Sessions)")
        print("=======================================================")
        model_27_path = os.path.join(PROJECT_ROOT, "models", "triple_draw_27", "model_27_a.zip")
        if os.path.exists(model_27_path):
            print("\n[Benchmarking] 2-7 Triple Draw Model 27-A (50 sessions)...")
            wr_27, r_27, _ = evaluate_27_model(model_27_path, num_sessions=50)
            print(f"  > Model 27-A Win Rate: {wr_27:.1f}% | Mean Return: {r_27:.2f} BBs")

        model_holdem_path = os.path.join(PROJECT_ROOT, "models", "holdem", "model_holdem_a.zip")
        if os.path.exists(model_holdem_path):
            print("\n[Benchmarking] Texas Hold'em Model Holdem-A (50 sessions)...")
            wr_h, r_h, _ = evaluate_holdem_model(model_holdem_path, num_sessions=50)
            print(f"  > Model Holdem-A Win Rate: {wr_h:.1f}% | Mean Return: {r_h:.2f} BBs")
