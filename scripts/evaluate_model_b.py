"""
evaluate_multi.py — Evaluation and Tell-Analysis for Multi-Player CardShark-RL.

Evaluates the trained Model B agent across 5-seat tournament sessions.
Features:
- Tournament Survival Rate & 1st Place Win Rate.
- BB/100 win rate calculation across multi-player hands.
- Scale-Invariance Verification across different buy-in depths (e.g. 100 vs 1,000 vs 5,000 chips).
- Behavioral Tell Analysis: Hero's action matrix mapped against opponent draw counts and aggression.
"""

from __future__ import annotations
import os
import argparse
import numpy as np
from typing import Dict, List, Optional

import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sb3_contrib import MaskablePPO
from rl.multi_gym_wrapper import MultiDrawPokerGymEnv, multi_mask_fn
from game.multi_draw_poker_env import (
    A_FOLD, A_CALL, A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN, A_DRAW_START
)
from game.multi_opponents import ARCHETYPE_CLASSES, NUM_ARCHETYPES


def evaluate_tournament_sessions(
    model_path: Optional[str] = None,
    num_sessions: int = 50,
    starting_chips: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    max_hands: int = 100,
    blind_escalation_interval: Optional[int] = None,
    randomize_stacks: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Evaluates an agent over multiple tournament sessions."""
    env = MultiDrawPokerGymEnv(
        num_seats=5,
        starting_chips=starting_chips,
        small_blind=small_blind,
        big_blind=big_blind,
        max_hands_per_session=max_hands,
        blind_escalation_interval=blind_escalation_interval,
        randomize_stacks=randomize_stacks,
        rng_seed=seed,
    )

    model = None
    if isinstance(model_path, MaskablePPO):
        model = model_path
    elif isinstance(model_path, str) and os.path.exists(model_path):
        if verbose:
            print(f"Loading policy from: {model_path}")
        model = MaskablePPO.load(model_path)
    else:
        if verbose:
            print("No model provided or not found — running Random Legal Baseline policy.")

    survived_count = 0
    first_place_count = 0
    total_hands_played = 0
    total_net_chips = 0
    final_chip_shares = []
    hands_per_session = []

    # Telemetry: Hero actions when facing different relative opponent draw counts
    # {draw_count: {"fold": N, "call": N, "raise": N}}
    draw_reaction_matrix = {
        d: {"fold": 0, "call": 0, "raise": 0} for d in range(6)
    }

    total_table_chips = starting_chips * 5

    for sess_idx in range(num_sessions):
        obs, info = env.reset(seed=seed + sess_idx * 17)
        done = False
        initial_chips = starting_chips

        while not done:
            mask = env.action_masks()
            legal_actions = np.where(mask == 1)[0]

            if model is not None:
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                action = int(action)
            else:
                action = int(np.random.choice(legal_actions))

            # Record telemetry if post-draw
            if env.env.phase == "post_draw":
                # Check draw count of the player who drew to Hero's right (last actor before Hero)
                right_seat = (env.hero_seat - 1) % env.num_seats
                opp_draw = env.env.seats[right_seat].draw_count
                if 0 <= opp_draw <= 5:
                    if action == A_FOLD:
                        draw_reaction_matrix[opp_draw]["fold"] += 1
                    elif action == A_CALL:
                        draw_reaction_matrix[opp_draw]["call"] += 1
                    elif action in (A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN):
                        draw_reaction_matrix[opp_draw]["raise"] += 1

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        hero_final_chips = info["hero_chips"]
        hands_in_sess = info["hands_played"]
        total_hands_played += hands_in_sess
        hands_per_session.append(hands_in_sess)

        net_chips = hero_final_chips - initial_chips
        total_net_chips += net_chips

        chip_share = (hero_final_chips / total_table_chips) * 100.0
        final_chip_shares.append(chip_share)

        if hero_final_chips > 0:
            survived_count += 1
        if hero_final_chips >= total_table_chips * 0.95 or info.get("is_winner", False):
            first_place_count += 1

    survival_rate = (survived_count / num_sessions) * 100.0
    first_place_rate = (first_place_count / num_sessions) * 100.0
    avg_chip_share = float(np.mean(final_chip_shares))
    avg_hands = float(np.mean(hands_per_session))

    # BB/100: (net_chips / big_blind) / (total_hands / 100)
    bb_per_100 = (total_net_chips / big_blind) / (max(1, total_hands_played) / 100.0)

    results = {
        "num_sessions": num_sessions,
        "total_hands": total_hands_played,
        "survival_rate_pct": survival_rate,
        "first_place_rate_pct": first_place_rate,
        "avg_final_chip_share_pct": avg_chip_share,
        "avg_hands_survived": avg_hands,
        "bb_per_100": bb_per_100,
        "draw_reaction_matrix": draw_reaction_matrix,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"  EVALUATION RESULTS ({num_sessions} Sessions | Starting chips: {starting_chips})")
        print(f"{'='*60}")
        print(f"  Total Hands Played:       {total_hands_played:,}")
        print(f"  Tournament Survival Rate: {survival_rate:.1f}%")
        print(f"  1st Place Win Rate:       {first_place_rate:.1f}%")
        print(f"  Avg Final Chip Share:     {avg_chip_share:.1f}% of table chips")
        print(f"  Avg Hands Survived:       {avg_hands:.1f} hands")
        print(f"  Winrate:                  {bb_per_100:+.2f} BB/100")
        print(f"{'='*60}\n")

    return results


def run_scale_invariance_test(model_path: Optional[str] = None):
    """Verifies that the agent achieves consistent win rates across different buy-in scales."""
    print("=== SCALE-INVARIANCE BENCHMARK ===")
    chip_configs = [
        {"name": "Micro (100 chips)", "chips": 100, "sb": 1, "bb": 2},
        {"name": "Standard (1,000 chips)", "chips": 1000, "sb": 10, "bb": 20},
        {"name": "High Roller (10,000 chips)", "chips": 10000, "sb": 100, "bb": 200},
    ]

    for cfg in chip_configs:
        print(f"\n--- Testing on {cfg['name']} ---")
        res = evaluate_tournament_sessions(
            model_path=model_path,
            num_sessions=20,
            starting_chips=cfg["chips"],
            small_blind=cfg["sb"],
            big_blind=cfg["bb"],
            max_hands=50,
            verbose=False,
        )
        print(
            f"  {cfg['name']}: Survival: {res['survival_rate_pct']:.1f}% | "
            f"Avg Chip Share: {res['avg_final_chip_share_pct']:.1f}% | "
            f"BB/100: {res['bb_per_100']:+.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Multi-Player CardShark-RL Agent")
    parser.add_argument("--model-path", type=str, default="models/model_b_multiplayer.zip")
    parser.add_argument("--sessions", type=int, default=30)
    parser.add_argument("--chips", type=int, default=200)
    parser.add_argument("--freezeout", action="store_true", help="Escalate blinds to guarantee full elimination down to 1 winner")
    parser.add_argument("--randomize-stacks", action="store_true", help="Start sessions with randomized stack depths")
    parser.add_argument("--test-scale", action="store_true", help="Run scale invariance across buy-in sizes")
    args = parser.parse_args()

    evaluate_tournament_sessions(
        model_path=args.model_path if os.path.exists(args.model_path) else None,
        num_sessions=args.sessions,
        starting_chips=args.chips,
        blind_escalation_interval=12 if args.freezeout else None,
        randomize_stacks=args.randomize_stacks,
        max_hands=250 if args.freezeout else 100,
    )

    if args.test_scale:
        run_scale_invariance_test(
            model_path=args.model_path if os.path.exists(args.model_path) else None
        )
