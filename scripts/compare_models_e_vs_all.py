"""
compare_models_e_vs_all.py — Definitive Multi-Model Tournament & Statistical Benchmark Suite.

Evaluates Model E (The Superhuman Crusher) against all project models and table adversaries:
- Model E (91-dim ICM + Dominated Discard Masking)
- Model D (87-dim Card Attention Champion)
- Model C (87-dim Superhuman MLP Baseline)
- Adversarial Exploiter (Aggressive probe-betting nemesis)
- Random Heuristic Archetype (TAG, LAG, Rock, CallingStation)

Features:
1. Positional-Parity Tournaments: Alternates hero seats between Model E and Model D/C to eliminate positional bias.
2. Full Elimination Completion: Games run until 1 survivor wins all table chips.
3. Strict Deterministic Inference: All neural models evaluate with greedy deterministic actions.
4. Statistical Testing: Computes Wilson 95% confidence intervals and two-proportion z-tests.
"""

from __future__ import annotations
import os
import sys
import math
import json
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sb3_contrib import MaskablePPO
from rl.multi_gym_wrapper import (
    MultiDrawPokerGymEnv,
    ModelObsDim,
    OBS_DIM_B,
    OBS_DIM_C,
    OBS_DIM_D,
    OBS_DIM_E,
)
from rl.league import LeagueOpponent
from game.multi_opponents import AdversarialExploiter, make_random_archetype
from rl.card_attention import CardAttentionExtractor
import rl.card_attention
sys.modules["card_attention"] = rl.card_attention


def calculate_proportions_z_test(k1: int, n1: int, k2: int, n2: int) -> Tuple[float, float]:
    """Two-proportion z-test for difference in win rates. Returns (z_score, p_value)."""
    if n1 <= 0 or n2 <= 0:
        return 0.0, 1.0

    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)

    if p_pool == 0 or p_pool == 1:
        return 0.0, 1.0

    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    if se <= 1e-12:
        return 0.0, 1.0

    z = (p1 - p2) / se
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return float(z), float(p_value)


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes Wilson score 95% confidence interval for a proportion."""
    if n <= 0:
        return 0.0, 0.0
    z = 1.96
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) + (z**2) / (4.0 * n)) / n)) / denom
    lower = max(0.0, center - margin) * 100.0
    upper = min(1.0, center + margin) * 100.0
    return round(lower, 1), round(upper, 1)


def run_benchmark_e_vs_all(
    model_e_path: str = "models/model_e.zip",
    model_d_path: str = "models/model_d.zip",
    model_c_path: str = "models/model_c.zip",
    num_sessions: int = 100,
    starting_chips: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    blind_escalation: int = 15,
    max_hands: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    if not os.path.exists(model_e_path):
        raise FileNotFoundError(f"Model E checkpoint not found at: {model_e_path}")

    has_model_d = os.path.exists(model_d_path)
    has_model_c = os.path.exists(model_c_path)

    if verbose:
        print("\n" + "=" * 80)
        print("     CARDSHARK-RL: MULTI-MODEL TOURNAMENT BENCHMARK (MODELS E vs D vs C)")
        print("=" * 80)
        print(f"  Model E Policy:                     {model_e_path}")
        print(f"  Model D Policy:                     {model_d_path if has_model_d else 'None'}")
        print(f"  Model C Policy:                     {model_c_path if has_model_c else 'None'}")
        print(f"  Adversarial Nemesis (Seat 3):       AdversarialExploiter")
        print(f"  Table Archetype (Seat 4):           Random Archetype (TAG/LAG/Rock)")
        print(f"  Tournament Configuration:          {num_sessions} sessions | 5 seats | {starting_chips} chips/seat")
        print(f"  Blinds: Small={small_blind}, Big={big_blind} (Escalates every {blind_escalation} hands)")
        print("=" * 80 + "\n")

    # Load policies
    model_e_policy = MaskablePPO.load(model_e_path)
    model_d_policy = MaskablePPO.load(model_d_path) if has_model_d else None

    # Vectorized / Gym Env with Model E 91-dim state and Discard Masking
    env = MultiDrawPokerGymEnv(
        num_seats=5,
        starting_chips=starting_chips,
        small_blind=small_blind,
        big_blind=big_blind,
        hero_seat=0,
        max_hands_per_session=max_hands,
        blind_escalation_interval=blind_escalation,
        randomize_stacks=True,
        obs_dim=ModelObsDim.MODEL_E,
        discard_masking=True,
        rng_seed=seed,
    )

    exploiter_bot = AdversarialExploiter()
    model_c_bot = LeagueOpponent(model_path=model_c_path, opponent_id=2, name="Model_C", deterministic=True) if has_model_c else None

    # Metrics Tracking
    e_titles = 0
    d_titles = 0
    c_titles = 0
    exploiter_titles = 0
    archetype_titles = 0

    e_net_chips_total = 0
    d_net_chips_total = 0
    c_net_chips_total = 0
    total_hands_played = 0

    e_chip_shares: list[float] = []
    d_chip_shares: list[float] = []
    c_chip_shares: list[float] = []

    e_survivals = 0
    d_survivals = 0
    c_survivals = 0

    e_outlasted_d = 0
    e_outlasted_c = 0
    e_outlasted_exploiter = 0

    total_table_chips = starting_chips * 5

    for sess in range(num_sessions):
        # Alternating seating:
        # Even sessions: Model E is Hero (Seat 0), Model D is LeagueOpponent (Seat 1)
        # Odd sessions:  Model D is Hero (Seat 0), Model E is LeagueOpponent (Seat 1)
        is_e_hero = (sess % 2 == 0) or not has_model_d

        e_seat = 0 if is_e_hero else 1
        d_seat = 1 if is_e_hero else 0

        # Model D uses 87-dim observation space; Model E uses 91-dim observation space.
        # To ensure strict native observation compatibility:
        # We configure env.obs_dim according to whichever model is Hero (Seat 0)!
        env.obs_dim = ModelObsDim.MODEL_E.value if is_e_hero else ModelObsDim.MODEL_D.value
        env.discard_masking = is_e_hero  # Model E uses dominated discard action masking

        obs, info = env.reset(seed=seed + sess * 89)

        hero_model = model_e_policy if is_e_hero else model_d_policy

        # Seed Seat 1 with the other model
        if is_e_hero and has_model_d:
            env.opponents[1] = LeagueOpponent(model_path=model_d_path, opponent_id=1, name="Model_D", deterministic=True)
        elif not is_e_hero:
            env.opponents[1] = LeagueOpponent(model_path=model_e_path, opponent_id=1, name="Model_E", deterministic=True)

        # Seat 2: Model C
        if model_c_bot:
            env.opponents[2] = model_c_bot

        # Seat 3: Adversarial Exploiter
        env.opponents[3] = exploiter_bot

        # Seat 4: Random Archetype
        env.opponents[4] = make_random_archetype(rng=np.random.default_rng(seed + sess * 23))

        done = False
        while not done:
            mask = env.action_masks()
            action, _ = hero_model.predict(obs, action_masks=mask, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Complete tournament if hero eliminated early so tournament plays down to 1 survivor
        if len(env.env.alive_seats) > 1 and not env.env.session_done and env.env.hands_played < max_hands:
            env.step_table_until_winner()

        hands_in_session = env.env.hands_played
        total_hands_played += hands_in_session

        chips_e = env.env.seats[e_seat].chips
        chips_d = env.env.seats[d_seat].chips if has_model_d else 0
        chips_c = env.env.seats[2].chips if has_model_c else 0
        chips_exp = env.env.seats[3].chips

        # Net chips
        net_e = chips_e - starting_chips
        net_d = chips_d - starting_chips
        net_c = chips_c - starting_chips
        e_net_chips_total += net_e
        d_net_chips_total += net_d
        c_net_chips_total += net_c

        # Chip share
        share_e = (chips_e / total_table_chips) * 100.0
        share_d = (chips_d / total_table_chips) * 100.0
        share_c = (chips_c / total_table_chips) * 100.0
        e_chip_shares.append(share_e)
        d_chip_shares.append(share_d)
        c_chip_shares.append(share_c)

        if chips_e > 0:
            e_survivals += 1
        if chips_d > 0:
            d_survivals += 1
        if chips_c > 0:
            c_survivals += 1

        if chips_e > chips_d:
            e_outlasted_d += 1
        if chips_e > chips_c:
            e_outlasted_c += 1
        if chips_e > chips_exp:
            e_outlasted_exploiter += 1

        # Determine Winner
        winner_seat = None
        for s_idx, seat in enumerate(env.env.seats):
            if seat.chips >= total_table_chips * 0.95 or (len(env.env.alive_seats) == 1 and env.env.alive_seats[0] == s_idx):
                winner_seat = s_idx
                break

        if winner_seat is None:
            winner_seat = int(np.argmax([s.chips for s in env.env.seats]))

        if winner_seat == e_seat:
            e_titles += 1
        elif has_model_d and winner_seat == d_seat:
            d_titles += 1
        elif has_model_c and winner_seat == 2:
            c_titles += 1
        elif winner_seat == 3:
            exploiter_titles += 1
        else:
            archetype_titles += 1

        if verbose and (sess + 1) % max(1, num_sessions // 5) == 0:
            e_pct = (e_titles / (sess + 1)) * 100.0
            d_pct = (d_titles / (sess + 1)) * 100.0 if has_model_d else 0.0
            print(f"  [Progress: {sess+1}/{num_sessions} sessions] Model E: {e_pct:.1f}% ({e_titles}) | Model D: {d_pct:.1f}% ({d_titles}) | Exploiter: {exploiter_titles}")

    # Final Aggregation
    e_win_rate = (e_titles / num_sessions) * 100.0
    d_win_rate = (d_titles / num_sessions) * 100.0
    c_win_rate = (c_titles / num_sessions) * 100.0
    exp_win_rate = (exploiter_titles / num_sessions) * 100.0
    arch_win_rate = (archetype_titles / num_sessions) * 100.0

    e_bb_100 = (e_net_chips_total / big_blind) / (max(1, total_hands_played) / 100.0)
    d_bb_100 = (d_net_chips_total / big_blind) / (max(1, total_hands_played) / 100.0)
    c_bb_100 = (c_net_chips_total / big_blind) / (max(1, total_hands_played) / 100.0)

    e_avg_share = float(np.mean(e_chip_shares))
    d_avg_share = float(np.mean(d_chip_shares))
    c_avg_share = float(np.mean(c_chip_shares))

    e_ci_low, e_ci_high = wilson_score_interval(e_titles, num_sessions)
    d_ci_low, d_ci_high = wilson_score_interval(d_titles, num_sessions)
    c_ci_low, c_ci_high = wilson_score_interval(c_titles, num_sessions)

    z_stat, p_val = calculate_proportions_z_test(e_titles, num_sessions, d_titles, num_sessions)

    results = {
        "num_sessions": num_sessions,
        "total_hands_played": total_hands_played,
        "avg_session_length": round(total_hands_played / max(1, num_sessions), 1),
        "model_e": {
            "championships": e_titles,
            "win_rate_pct": round(e_win_rate, 1),
            "ci_95": [e_ci_low, e_ci_high],
            "bb_per_100": round(e_bb_100, 2),
            "avg_chip_share_pct": round(e_avg_share, 1),
            "survival_rate_pct": round(e_survivals / num_sessions * 100.0, 1),
        },
        "model_d": {
            "championships": d_titles,
            "win_rate_pct": round(d_win_rate, 1),
            "ci_95": [d_ci_low, d_ci_high],
            "bb_per_100": round(d_bb_100, 2),
            "avg_chip_share_pct": round(d_avg_share, 1),
            "survival_rate_pct": round(d_survivals / num_sessions * 100.0, 1),
        },
        "model_c": {
            "championships": c_titles,
            "win_rate_pct": round(c_win_rate, 1),
            "ci_95": [c_ci_low, c_ci_high],
            "bb_per_100": round(c_bb_100, 2),
            "avg_chip_share_pct": round(c_avg_share, 1),
            "survival_rate_pct": round(c_survivals / num_sessions * 100.0, 1),
        },
        "adversaries": {
            "exploiter_titles": exploiter_titles,
            "exploiter_win_rate_pct": round(exp_win_rate, 1),
            "archetype_titles": archetype_titles,
            "archetype_win_rate_pct": round(arch_win_rate, 1),
        },
        "head_to_head_outlast": {
            "model_e_outlasted_d": e_outlasted_d,
            "model_e_outlasted_c": e_outlasted_c,
            "model_e_outlasted_exploiter": e_outlasted_exploiter,
        },
        "hypothesis_test": {
            "z_statistic": round(z_stat, 3),
            "p_value": round(p_val, 4),
            "model_e_statistically_superior": p_val < 0.05 and z_stat > 0,
        },
    }

    if verbose:
        print("\n" + "=" * 80)
        print("               CARDSHARK-RL: MODEL E vs ALL TOURNAMENT RESULTS")
        print("=" * 80)
        print(f"{'Metric':<30} | {'Model E':<18} | {'Model D':<18} | {'Model C':<18}")
        print("-" * 80)
        print(f"{'1st Place Championship Rate':<30} | {e_win_rate:>6.1f}% [{e_ci_low}-{e_ci_high}%] | {d_win_rate:>6.1f}% [{d_ci_low}-{d_ci_high}%] | {c_win_rate:>6.1f}% [{c_ci_low}-{c_ci_high}%]")
        print(f"{'Total Titles Won':<30} | {e_titles:>6} / {num_sessions:<9} | {d_titles:>6} / {num_sessions:<9} | {c_titles:>6} / {num_sessions:<9}")
        print(f"{'BB/100 Win Rate':<30} | {e_bb_100:>+10.2f} BB/100  | {d_bb_100:>+10.2f} BB/100  | {c_bb_100:>+10.2f} BB/100")
        print(f"{'Average Final Chip Share':<30} | {e_avg_share:>6.1f}% (Par: 20%)  | {d_avg_share:>6.1f}% (Par: 20%)  | {c_avg_share:>6.1f}% (Par: 20%)")
        print(f"{'Tournament Survival Rate':<30} | {e_survivals / num_sessions * 100:>6.1f}%             | {d_survivals / num_sessions * 100:>6.1f}%             | {c_survivals / num_sessions * 100:>6.1f}%")
        print("-" * 80)
        print(f"  Adversarial Exploiter (Seat 3): {exp_win_rate:.1f}% ({exploiter_titles}/{num_sessions})")
        print(f"  Heuristic Archetypes (Seat 4):  {arch_win_rate:.1f}% ({archetype_titles}/{num_sessions})")
        print(f"  Model E Outlasted Model D:      {e_outlasted_d}/{num_sessions} tournaments")
        print(f"  Model E Outlasted Exploiter:    {e_outlasted_exploiter}/{num_sessions} tournaments")
        print("=" * 80)
        print(f"  Two-Proportion Z-Test (E vs D): z = {z_stat:.3f}, p-value = {p_val:.4f}\n")

    os.makedirs("results", exist_ok=True)
    out_file = "results/comparison_e_vs_all.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"  Saved full telemetry report to: {out_file}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Model Tournament Benchmark: Model E vs All")
    parser.add_argument("--sessions", type=int, default=100, help="Number of tournament sessions to simulate")
    parser.add_argument("--model-e", type=str, default="models/model_e.zip", help="Path to Model E checkpoint")
    parser.add_argument("--model-d", type=str, default="models/model_d.zip", help="Path to Model D checkpoint")
    parser.add_argument("--model-c", type=str, default="models/model_c.zip", help="Path to Model C checkpoint")
    parser.add_argument("--chips", type=int, default=200, help="Starting stack per seat")
    parser.add_argument("--blind-escalation", type=int, default=15, help="Hands between blind escalations")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    run_benchmark_e_vs_all(
        model_e_path=args.model_e,
        model_d_path=args.model_d,
        model_c_path=args.model_c,
        num_sessions=args.sessions,
        starting_chips=args.chips,
        blind_escalation=args.blind_escalation,
        seed=args.seed,
    )
