"""
evaluate_model_d.py — Evaluation Benchmark for CardShark-RL Model D Champion.

Measures:
1. Freezeout 1st Place Championship Rate across multi-player tournament tables.
2. Head-to-Head sparring against Model C (Seat 1) and Model B (Seat 2).
3. BB/100 win rate and average tournament chip share.
4. Tell & tilt exploitation audit against AdversarialExploiter.
"""

from __future__ import annotations
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import argparse
import numpy as np
from typing import Dict, List, Optional

import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sb3_contrib import MaskablePPO
from rl.multi_gym_wrapper import MultiDrawPokerGymEnv, SUPERHUMAN_OBS_DIM
from rl.league import LeagueOpponent
from game.multi_opponents import AdversarialExploiter, make_random_archetype
from rl.card_attention import CardAttentionExtractor
import rl.card_attention
sys.modules["card_attention"] = rl.card_attention


def evaluate_model_d_tournament(
    model_d_path: str = "models/model_d.zip",
    model_c_path: str = "models/model_c.zip",
    model_b_path: str = "models/model_b.zip",
    num_sessions: int = 30,
    starting_chips: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    max_hands: int = 200,
    blind_escalation_interval: Optional[int] = 15,
    randomize_stacks: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Evaluates Model D against Model C, Model B, and table adversaries."""
    # Fallback to archive if primary model_d.zip not found yet
    if not os.path.exists(model_d_path):
        for candidate in [
            "models/model_d.zip",
            "models/archive/run_1.5m_champion/model_d.zip",
            "models/archive/run_1.5m_champion/model_d_best.zip",
        ]:
            if os.path.exists(candidate):
                model_d_path = candidate
                break

    if not os.path.exists(model_d_path):
        raise FileNotFoundError(f"Model D checkpoint not found at: {model_d_path}")

    if verbose:
        print(f"=== EVALUATING MODEL D CHAMPION ({num_sessions} Tournament Sessions) ===")
        print(f"  Hero Model: {model_d_path}")
        print(f"  Sparring Opponent (Seat 1): {model_c_path if os.path.exists(model_c_path) else 'Heuristic Bot'}")
        print(f"  Sparring Opponent (Seat 2): {model_b_path if os.path.exists(model_b_path) else 'Heuristic Bot'}")
        print(f"  Sparring Opponent (Seat 3): AdversarialExploiter")

    model_d = MaskablePPO.load(model_d_path)

    env = MultiDrawPokerGymEnv(
        num_seats=5,
        starting_chips=starting_chips,
        small_blind=small_blind,
        big_blind=big_blind,
        hero_seat=0,
        max_hands_per_session=max_hands,
        blind_escalation_interval=blind_escalation_interval,
        randomize_stacks=randomize_stacks,
        superhuman_obs=True,
        rng_seed=seed,
    )

    # Fixed sparring seats
    has_model_c = os.path.exists(model_c_path)
    has_model_b = os.path.exists(model_b_path)

    model_c_bot = LeagueOpponent(model_path=model_c_path, opponent_id=1, name="Model_C_Superhuman") if has_model_c else None
    model_b_bot = LeagueOpponent(model_path=model_b_path, opponent_id=2, name="Model_B_Baseline") if has_model_b else None
    exploiter_bot = AdversarialExploiter()

    survived_count = 0
    first_place_count = 0
    model_d_wins = 0
    model_c_wins = 0
    model_b_wins = 0
    other_wins = 0

    total_hands_played = 0
    total_net_chips = 0
    final_chip_shares = []
    hands_per_session = []
    total_table_chips = starting_chips * 5

    for sess_idx in range(num_sessions):
        obs, info = env.reset(seed=seed + sess_idx * 31)

        # Seed sparring opponents
        if model_c_bot:
            env.opponents[1] = model_c_bot
        if model_b_bot:
            env.opponents[2] = model_b_bot
        env.opponents[3] = exploiter_bot

        done = False
        initial_chips = starting_chips

        while not done:
            mask = env.action_masks()
            action, _ = model_d.predict(obs, action_masks=mask, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        hero_chips = info["hero_chips"]
        hands_in_sess = info["hands_played"]
        total_hands_played += hands_in_sess
        hands_per_session.append(hands_in_sess)

        net_chips = hero_chips - initial_chips
        total_net_chips += net_chips

        chip_share = (hero_chips / total_table_chips) * 100.0
        final_chip_shares.append(chip_share)

        if hero_chips > 0:
            survived_count += 1

        # Check winner
        winner_seat = None
        for s_idx, seat in enumerate(env.env.seats):
            if seat.chips >= total_table_chips * 0.95 or (len(env.env.alive_seats) == 1 and env.env.alive_seats[0] == s_idx):
                winner_seat = s_idx
                break

        if winner_seat == 0:
            first_place_count += 1
            model_d_wins += 1
        elif winner_seat == 1 and has_model_c:
            model_c_wins += 1
        elif winner_seat == 2 and has_model_b:
            model_b_wins += 1
        else:
            other_wins += 1

    survival_rate = (survived_count / max(1, num_sessions)) * 100.0
    first_place_rate = (first_place_count / max(1, num_sessions)) * 100.0
    avg_chip_share = float(np.mean(final_chip_shares)) if final_chip_shares else 0.0
    avg_hands = float(np.mean(hands_per_session)) if hands_per_session else 0.0
    bb_per_100 = (total_net_chips / big_blind) / (max(1, total_hands_played) / 100.0)

    results = {
        "model": "Model D Champion",
        "num_sessions": num_sessions,
        "total_hands": total_hands_played,
        "avg_session_length": round(avg_hands, 1),
        "bb_per_100": round(bb_per_100, 2),
        "survival_rate_pct": round(survival_rate, 1),
        "first_place_rate_pct": round(first_place_rate, 1),
        "avg_final_chip_share_pct": round(avg_chip_share, 1),
        "head_to_head_breakdown": {
            "Model_D_Championships": model_d_wins,
            "Model_C_Championships": model_c_wins,
            "Model_B_Championships": model_b_wins,
            "Other_Adversaries_Wins": other_wins,
        },
    }

    if verbose:
        print("\n" + "=" * 60)
        print("  MODEL D TOURNAMENT EVALUATION SUMMARY")
        print("=" * 60)
        print(f"  Winrate:               {bb_per_100:+.2f} BB/100")
        print(f"  Championship Rate (1st): {first_place_rate:.1f}% ({model_d_wins}/{num_sessions} tournaments)")
        print(f"  Survival Rate:         {survival_rate:.1f}%")
        print(f"  Average Chip Share:    {avg_chip_share:.1f}% (Table parity: 20.0%)")
        print(f"  Average Hands/Session: {avg_hands:.1f} hands")
        print("\n  Head-to-Head Titles:")
        print(f"    - Model D Champion:   {model_d_wins} titles")
        print(f"    - Model C Superhuman: {model_c_wins} titles")
        print(f"    - Model B Baseline:   {model_b_wins} titles")
        print(f"    - Other Bots:         {other_wins} titles")
        print("=" * 60 + "\n")

    os.makedirs("results", exist_ok=True)
    with open("results/model_d_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CardShark-RL Model D Champion")
    parser.add_argument("--sessions", "--num-sessions", dest="sessions", type=int, default=30, help="Number of tournament sessions")
    parser.add_argument("--model-d", type=str, default="models/model_d.zip")
    parser.add_argument("--model-c", type=str, default="models/model_c.zip")
    parser.add_argument("--model-b", type=str, default="models/model_b.zip")
    parser.add_argument("--chips", type=int, default=200)
    parser.add_argument("--blind-escalation", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate_model_d_tournament(
        model_d_path=args.model_d,
        model_c_path=args.model_c,
        model_b_path=args.model_b,
        num_sessions=args.sessions,
        starting_chips=args.chips,
        blind_escalation_interval=args.blind_escalation,
        seed=args.seed,
    )
