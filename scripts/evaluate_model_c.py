"""
evaluate_superhuman.py — Comprehensive Benchmark Suite for Model C (Superhuman Autonomous Agent).

Capabilities:
1. Head-to-Head Sparring: Pits Model C directly against Model B at the same table.
2. Freezeout Championship Rate: Measures 1st place tournament knockout rate and survival.
3. Comparative Benchmark Table: Side-by-side performance metrics (BB/100, 1st place %, chip share).
4. Dynamic Tilt & Tell Exploitation Audit: Measures Model C's counter-actions when opponents tilt.
"""

from __future__ import annotations
import os
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple

import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sb3_contrib import MaskablePPO
from rl.multi_gym_wrapper import MultiDrawPokerGymEnv, make_superhuman_multi_env, SUPERHUMAN_OBS_DIM, OBS_DIM
from game.multi_draw_poker_env import (
    A_FOLD, A_CALL, A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN,
)
from rl.league import LeagueOpponent
from game.multi_opponents import make_random_archetype, make_opponent_by_id


def evaluate_superhuman_sessions(
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
    spar_against_model_b: bool = True,
    verbose: bool = True,
) -> dict:
    """Evaluates Model C in 5-seat tables, optionally sparring directly against Model B in Seat 1."""

    # 1. Load Model C policy
    if not os.path.exists(model_c_path):
        raise FileNotFoundError(f"Model C checkpoint not found at: {model_c_path}")

    model_c = MaskablePPO.load(model_c_path)
    obs_dim = model_c.observation_space.shape[0]
    is_superhuman = (obs_dim == SUPERHUMAN_OBS_DIM)

    # 2. Setup Environment
    env = MultiDrawPokerGymEnv(
        num_seats=5,
        starting_chips=starting_chips,
        small_blind=small_blind,
        big_blind=big_blind,
        hero_seat=0, # Model C at Seat 0
        max_hands_per_session=max_hands,
        blind_escalation_interval=blind_escalation_interval,
        randomize_stacks=randomize_stacks,
        superhuman_obs=is_superhuman,
        rng_seed=seed,
    )

    # 3. Setup Model B as fixed opponent at Seat 1 if requested
    has_model_b_sparring = spar_against_model_b and os.path.exists(model_b_path)
    model_b_bot = None
    if has_model_b_sparring:
        model_b_bot = LeagueOpponent(
            model_path=model_b_path,
            opponent_id=1,
            name="Model_B_SparringPartner",
            deterministic=True,
        )

    survived_count = 0
    first_place_count = 0
    total_hands_played = 0
    total_net_chips = 0
    final_chip_shares = []
    hands_per_session = []

    # Model B head-to-head tracking
    model_b_wins = 0
    model_c_wins = 0

    # Tilt audit tracking: {tilt_active: {"fold": N, "call": N, "raise": N}}
    tilt_reaction = {
        "tilt_detected": {"fold": 0, "call": 0, "raise": 0},
        "normal_play":   {"fold": 0, "call": 0, "raise": 0},
    }

    total_table_chips = starting_chips * 5

    for sess_idx in range(num_sessions):
        obs, info = env.reset(seed=seed + sess_idx * 31)

        # Inject Model B at Seat 1
        if has_model_b_sparring and model_b_bot is not None:
            env.opponents[1] = model_b_bot

        done = False
        initial_chips = starting_chips

        while not done:
            mask = env.action_masks()
            action, _ = model_c.predict(obs, action_masks=mask, deterministic=True)
            action = int(action)

            # Audit Hero reaction when facing an opponent who is actively tilting
            any_opponent_tilting = any(
                env.tracker.get_seat_recency_delta(s) > 0.25
                for s in range(1, env.num_seats)
                if env.env.seats[s].is_alive and not env.env.seats[s].folded
            )
            key = "tilt_detected" if any_opponent_tilting else "normal_play"
            if action == A_FOLD:
                tilt_reaction[key]["fold"] += 1
            elif action == A_CALL:
                tilt_reaction[key]["call"] += 1
            elif action in (A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN):
                tilt_reaction[key]["raise"] += 1

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
        if info.get("is_winner", False) or hero_chips >= total_table_chips * 0.95:
            first_place_count += 1
            model_c_wins += 1
        elif has_model_b_sparring:
            # Check if Model B won
            seat_1_chips = env.env.seats[1].chips
            if seat_1_chips >= total_table_chips * 0.95:
                model_b_wins += 1

    survival_rate = (survived_count / num_sessions) * 100.0
    first_place_rate = (first_place_count / num_sessions) * 100.0
    avg_chip_share = float(np.mean(final_chip_shares))
    avg_hands = float(np.mean(hands_per_session))
    bb_per_100 = (total_net_chips / big_blind) / (max(1, total_hands_played) / 100.0)

    results = {
        "model_name": os.path.basename(model_c_path),
        "obs_dim": obs_dim,
        "num_sessions": num_sessions,
        "total_hands": total_hands_played,
        "survival_rate_pct": survival_rate,
        "first_place_rate_pct": first_place_rate,
        "avg_final_chip_share_pct": avg_chip_share,
        "avg_hands_survived": avg_hands,
        "bb_per_100": bb_per_100,
        "model_c_wins": model_c_wins,
        "model_b_wins": model_b_wins,
        "tilt_reaction": tilt_reaction,
    }

    if verbose:
        print("\n" + "=" * 68)
        print(f"  MODEL C SUPERHUMAN BENCHMARK ({num_sessions} Sessions | Starting Chips: {starting_chips})")
        print("=" * 68)
        print(f"  Observation Dimension:        {obs_dim} (Superhuman Architecture)")
        print(f"  Total Hands Played:           {total_hands_played:,}")
        print(f"  Tournament Survival Rate:     {survival_rate:.1f}%")
        print(f"  1st Place Win Rate:           {first_place_rate:.1f}% (Table parity is 20.0%)")
        print(f"  Average Chip Share:           {avg_chip_share:.1f}% of table chips")
        print(f"  Average Hands Survived:       {avg_hands:.1f} hands")
        print(f"  Winrate:                      {bb_per_100:+.2f} BB/100")
        if has_model_b_sparring:
            print(f"  Head-to-Head vs Model B:      Model C [{model_c_wins}] - [{model_b_wins}] Model B")
        print("-" * 68)
        print("  Tilt Adaptation Audit:")
        for state, counts in tilt_reaction.items():
            tot = max(1, sum(counts.values()))
            f_pct = counts["fold"] / tot * 100.0
            c_pct = counts["call"] / tot * 100.0
            r_pct = counts["raise"] / tot * 100.0
            print(f"    {state:<16} Fold: {f_pct:4.1f}% | Call: {c_pct:4.1f}% | Raise: {r_pct:4.1f}%")
        print("=" * 68 + "\n")

    return results


def run_comparative_table(model_c_path: str, model_b_path: str, sessions: int = 25):
    """Executes side-by-side benchmark comparing Model C and Model B."""
    print("\n" + "=" * 76)
    print("  RUNNING SIDE-BY-SIDE BENCHMARK: MODEL C (Superhuman) vs MODEL B (Production)")
    print("=" * 76)

    res_b = None
    if os.path.exists(model_b_path):
        print(f"1. Evaluating Model B baseline ({model_b_path})...")
        from evaluate_multi import evaluate_tournament_sessions
        res_b = evaluate_tournament_sessions(
            model_path=model_b_path,
            num_sessions=sessions,
            blind_escalation_interval=15,
            randomize_stacks=True,
            max_hands=200,
            verbose=False,
        )

    res_c = None
    if os.path.exists(model_c_path):
        print(f"2. Evaluating Model C Superhuman ({model_c_path})...")
        res_c = evaluate_superhuman_sessions(
            model_c_path=model_c_path,
            model_b_path=model_b_path,
            num_sessions=sessions,
            blind_escalation_interval=15,
            randomize_stacks=True,
            max_hands=200,
            spar_against_model_b=True,
            verbose=False,
        )

    # Print Comparative Table
    print("\n" + "-" * 76)
    print(f"  {'Metric':<28} | {'Model B (Baseline)':<20} | {'Model C (Superhuman)':<20}")
    print("-" * 76)
    if res_b and res_c:
        print(f"  {'Observation Dimension':<28} | {'63 dims':<20} | {'87 dims':<20}")
        print(f"  {'Winrate (BB/100)':<28} | {res_b['bb_per_100']:>+17.2f}  | {res_c['bb_per_100']:>+17.2f} ")
        print(f"  {'1st Place Win Rate':<28} | {res_b['first_place_rate_pct']:>19.1f}% | {res_c['first_place_rate_pct']:>19.1f}%")
        print(f"  {'Survival Rate':<28} | {res_b['survival_rate_pct']:>19.1f}% | {res_c['survival_rate_pct']:>19.1f}%")
        print(f"  {'Table Chip Share':<28} | {res_b['avg_final_chip_share_pct']:>19.1f}% | {res_c['avg_final_chip_share_pct']:>19.1f}%")
        print(f"  {'Hands per Session':<28} | {res_b['avg_hands_survived']:>19.1f}  | {res_c['avg_hands_survived']:>19.1f} ")
    print("-" * 76 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Model C Superhuman Poker Agent")
    parser.add_argument("--model-c", type=str, default="models/model_c.zip")
    parser.add_argument("--model-b", type=str, default="models/model_b.zip")
    parser.add_argument("--sessions", type=int, default=30)
    parser.add_argument("--chips", type=int, default=200)
    parser.add_argument("--compare", action="store_true", help="Run head-to-head comparison table")
    parser.add_argument("--smoke-test", action="store_true", help="Run 5 sessions for quick verification")
    args = parser.parse_args()

    sessions = 5 if args.smoke_test else args.sessions

    if args.compare:
        run_comparative_table(args.model_c, args.model_b, sessions=sessions)
    else:
        evaluate_superhuman_sessions(
            model_c_path=args.model_c,
            model_b_path=args.model_b,
            num_sessions=sessions,
            starting_chips=args.chips,
            blind_escalation_interval=15,
            randomize_stacks=True,
            max_hands=200,
        )
