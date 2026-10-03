"""
compare_models_c_vs_d.py — Head-to-Head Tournament & Statistical Comparison Suite.

Rigorous comparison between Model C (Superhuman MLP Baseline) and Model D (Card Attention Champion).

Features:
1. Positional-Parity Tournaments: Alternates/randomizes seat assignments across sessions
   to completely eliminate positional bias between Model C and Model D.
2. Direct Head-to-Head Tracking: Measures 1st place championship rates, direct knockout duels,
   average final chip shares, and BB/100 win rates for BOTH models simultaneously.
3. Rigorous Statistical Testing: Calculates Wilson score 95% confidence intervals and
   two-proportion z-test p-values to determine if Model D is statistically significantly superior.
4. Automated Actionable Diagnostics: Outputs clear guidance on whether to deploy or run HPO.

Usage:
    python scripts/compare_models_c_vs_d.py --sessions 50
    python scripts/compare_models_c_vs_d.py --sessions 100 --model-d models/model_d.zip --model-c models/model_c.zip
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
from rl.multi_gym_wrapper import MultiDrawPokerGymEnv, SUPERHUMAN_OBS_DIM
from rl.league import LeagueOpponent
from game.multi_opponents import AdversarialExploiter, make_random_archetype
from rl.card_attention import CardAttentionExtractor
import rl.card_attention
sys.modules["card_attention"] = rl.card_attention


def calculate_proportions_z_test(k1: int, n1: int, k2: int, n2: int) -> Tuple[float, float]:
    """
    Two-proportion z-test for difference in win rates.
    Returns: (z_score, two_tailed_p_value)
    """
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
    # Two-tailed p-value using normal error function approximation
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return float(z), float(p_value)


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes Wilson score 95% confidence interval for a proportion."""
    if n <= 0:
        return 0.0, 0.0
    z = 1.96  # 95% confidence
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) + (z**2) / (4.0 * n)) / n)) / denom
    lower = max(0.0, center - margin) * 100.0
    upper = min(1.0, center + margin) * 100.0
    return round(lower, 1), round(upper, 1)


def run_head_to_head_comparison(
    model_d_path: str = "models/model_d.zip",
    model_c_path: str = "models/model_c.zip",
    model_b_path: str = "models/model_b.zip",
    num_sessions: int = 50,
    starting_chips: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    blind_escalation: int = 15,
    max_hands: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Executes a balanced head-to-head tournament benchmark between Model C and Model D."""

    # 1. Resolve model checkpoints
    for candidate in [model_d_path, "models/archive/run_1.5m_champion/model_d.zip"]:
        if os.path.exists(candidate):
            model_d_path = candidate
            break

    if not os.path.exists(model_d_path):
        raise FileNotFoundError(f"Model D checkpoint not found: {model_d_path}")
    if not os.path.exists(model_c_path):
        raise FileNotFoundError(f"Model C checkpoint not found: {model_c_path}")

    if verbose:
        print("\n" + "=" * 76)
        print("   CARDSHARK-RL: MODEL C vs MODEL D TOURNAMENT BENCHMARK")
        print("=" * 76)
        print(f"  Model D (Card Attention Champion): {model_d_path}")
        print(f"  Model C (Superhuman MLP Baseline): {model_c_path}")
        print(f"  Model B (Benchmark Anchor):        {model_b_path if os.path.exists(model_b_path) else 'None'}")
        print(f"  Tournament Configuration:          {num_sessions} sessions | 5 seats | {starting_chips} chips/seat")
        print(f"  Blinds: Small={small_blind}, Big={big_blind} (Escalates every {blind_escalation} hands)")
        print("=" * 76 + "\n")

    # Load policies
    model_d_policy = MaskablePPO.load(model_d_path)
    model_c_policy = MaskablePPO.load(model_c_path)

    # Initialize gym env
    env = MultiDrawPokerGymEnv(
        num_seats=5,
        starting_chips=starting_chips,
        small_blind=small_blind,
        big_blind=big_blind,
        hero_seat=0,
        max_hands_per_session=max_hands,
        blind_escalation_interval=blind_escalation,
        randomize_stacks=True,
        superhuman_obs=True,
        rng_seed=seed,
    )

    # Fixed sparring bots
    has_model_b = os.path.exists(model_b_path)
    model_b_bot = LeagueOpponent(model_path=model_b_path, opponent_id=2, name="Model_B") if has_model_b else None
    exploiter_bot = AdversarialExploiter()

    # Metrics Tracking
    d_titles = 0
    c_titles = 0
    b_titles = 0
    other_titles = 0

    d_net_chips_total = 0
    c_net_chips_total = 0
    total_hands_played = 0

    d_chip_shares = []
    c_chip_shares = []
    session_hands_list = []

    d_survivals = 0
    c_survivals = 0

    # Direct duels: When it comes down to final 2 (Model D vs Model C), who wins?
    direct_duels_d_won = 0
    direct_duels_c_won = 0

    total_table_chips = starting_chips * 5

    for sess in range(num_sessions):
        # Alternate hero role between Model D and Model C to ensure zero bias
        # Even sessions: Model D is Hero (Seat 0), Model C is LeagueOpponent (Seat 1)
        # Odd sessions:  Model C is Hero (Seat 0), Model D is LeagueOpponent (Seat 1)
        is_d_hero = (sess % 2 == 0)

        hero_model = model_d_policy if is_d_hero else model_c_policy
        d_seat = 0 if is_d_hero else 1
        c_seat = 1 if is_d_hero else 0

        obs, info = env.reset(seed=seed + sess * 79)

        # Seat 1: The other neural model
        if is_d_hero:
            env.opponents[1] = LeagueOpponent(model_path=model_c_path, opponent_id=1, name="Model_C")
        else:
            env.opponents[1] = LeagueOpponent(model_path=model_d_path, opponent_id=1, name="Model_D")

        # Seat 2: Model B
        if model_b_bot:
            env.opponents[2] = model_b_bot
        # Seat 3: Adversarial Exploiter
        env.opponents[3] = exploiter_bot
        # Seat 4: Random Archetype
        env.opponents[4] = make_random_archetype(rng=np.random.default_rng(seed + sess * 13))

        done = False
        while not done:
            mask = env.action_masks()
            action, _ = hero_model.predict(obs, action_masks=mask, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Extract final table chips
        hands_in_session = info.get("hands_played", 1)
        total_hands_played += hands_in_session
        session_hands_list.append(hands_in_session)

        chips_d = env.env.seats[d_seat].chips
        chips_c = env.env.seats[c_seat].chips
        chips_b = env.env.seats[2].chips if has_model_b else 0

        # Net chips
        net_d = chips_d - starting_chips
        net_c = chips_c - starting_chips
        d_net_chips_total += net_d
        c_net_chips_total += net_c

        # Chip share
        share_d = (chips_d / total_table_chips) * 100.0
        share_c = (chips_c / total_table_chips) * 100.0
        d_chip_shares.append(share_d)
        c_chip_shares.append(share_c)

        if chips_d > 0:
            d_survivals += 1
        if chips_c > 0:
            c_survivals += 1

        # Determine Winner
        winner_seat = None
        for s_idx, seat in enumerate(env.env.seats):
            if seat.chips >= total_table_chips * 0.95 or (len(env.env.alive_seats) == 1 and env.env.alive_seats[0] == s_idx):
                winner_seat = s_idx
                break

        # Fallback if max hands hit without full elimination
        if winner_seat is None:
            winner_seat = int(np.argmax([s.chips for s in env.env.seats]))

        if winner_seat == d_seat:
            d_titles += 1
        elif winner_seat == c_seat:
            c_titles += 1
        elif winner_seat == 2 and has_model_b:
            b_titles += 1
        else:
            other_titles += 1

        # Direct heads-up duel check
        alive = env.env.alive_seats
        if len(alive) == 2 and d_seat in alive and c_seat in alive:
            if winner_seat == d_seat:
                direct_duels_d_won += 1
            elif winner_seat == c_seat:
                direct_duels_c_won += 1

        if verbose and (sess + 1) % max(1, num_sessions // 5) == 0:
            d_pct_now = (d_titles / (sess + 1)) * 100.0
            c_pct_now = (c_titles / (sess + 1)) * 100.0
            print(f"  [Progress: {sess+1}/{num_sessions} sessions] Model D: {d_pct_now:.1f}% ({d_titles}) | Model C: {c_pct_now:.1f}% ({c_titles})")

    # Final Aggregation
    d_win_rate = (d_titles / num_sessions) * 100.0
    c_win_rate = (c_titles / num_sessions) * 100.0
    b_win_rate = (b_titles / num_sessions) * 100.0
    other_win_rate = (other_titles / num_sessions) * 100.0

    d_bb_100 = (d_net_chips_total / big_blind) / (max(1, total_hands_played) / 100.0)
    c_bb_100 = (c_net_chips_total / big_blind) / (max(1, total_hands_played) / 100.0)

    d_avg_share = float(np.mean(d_chip_shares))
    c_avg_share = float(np.mean(c_chip_shares))

    d_surv_rate = (d_survivals / num_sessions) * 100.0
    c_surv_rate = (c_survivals / num_sessions) * 100.0

    # Statistical significance
    z_stat, p_val = calculate_proportions_z_test(d_titles, num_sessions, c_titles, num_sessions)
    d_ci_low, d_ci_high = wilson_score_interval(d_titles, num_sessions)
    c_ci_low, c_ci_high = wilson_score_interval(c_titles, num_sessions)

    # Diagnostic Assessment
    if d_win_rate > c_win_rate and p_val < 0.05:
        verdict = "STATISTICALLY SIGNIFICANT BREAKTHROUGH (p < 0.05). Model D decisively outperforms Model C!"
        recommendation = "Deploy Model D to production. Its CardAttention Transformer provides superior tournament equity."
    elif d_win_rate > c_win_rate:
        verdict = f"MODEL D ADVANTAGE DETECTED (+{d_win_rate - c_win_rate:.1f}%), but within statistical margin of error (p = {p_val:.3f})."
        recommendation = "Model D is ahead. To confirm statistical significance, run --sessions 150. If plateaued, execute HPO tuning."
    elif math.isclose(d_win_rate, c_win_rate, abs_tol=2.0):
        verdict = f"PARITY DETECTED ({d_win_rate:.1f}% vs {c_win_rate:.1f}%)."
        recommendation = "Model D has matched Model C's capability but not surpassed it. Recommended action: Run HPO search."
    else:
        verdict = f"MODEL C REMAINS SUPERIOR (-{c_win_rate - d_win_rate:.1f}%)."
        recommendation = "Model D hyperparameters need recalibration. Run `python scripts/hpo_model_d.py --mode staged`."

    results = {
        "num_sessions": num_sessions,
        "total_hands_played": total_hands_played,
        "avg_session_length": round(float(np.mean(session_hands_list)), 1),
        "model_d": {
            "championships": d_titles,
            "win_rate_pct": round(d_win_rate, 1),
            "ci_95": [d_ci_low, d_ci_high],
            "bb_per_100": round(d_bb_100, 2),
            "avg_chip_share_pct": round(d_avg_share, 1),
            "survival_rate_pct": round(d_surv_rate, 1),
        },
        "model_c": {
            "championships": c_titles,
            "win_rate_pct": round(c_win_rate, 1),
            "ci_95": [c_ci_low, c_ci_high],
            "bb_per_100": round(c_bb_100, 2),
            "avg_chip_share_pct": round(c_avg_share, 1),
            "survival_rate_pct": round(c_surv_rate, 1),
        },
        "other_opponents": {
            "model_b_titles": b_titles,
            "model_b_win_rate_pct": round(b_win_rate, 1),
            "adversaries_titles": other_titles,
            "adversaries_win_rate_pct": round(other_win_rate, 1),
        },
        "direct_duels": {
            "model_d_won": direct_duels_d_won,
            "model_c_won": direct_duels_c_won,
        },
        "hypothesis_test": {
            "z_statistic": round(z_stat, 3),
            "p_value": round(p_val, 4),
            "statistically_significant_05": p_val < 0.05,
        },
        "verdict": verdict,
        "recommendation": recommendation,
    }

    if verbose:
        print("\n" + "=" * 76)
        print("               CARDSHARK-RL: HEAD-TO-HEAD TOURNAMENT RESULTS")
        print("=" * 76)
        print(f"{'Metric':<32} | {'Model D (Attention)':<18} | {'Model C (Superhuman)':<18}")
        print("-" * 76)
        print(f"{'1st Place Championship Rate':<32} | {d_win_rate:>6.1f}% [{d_ci_low}-{d_ci_high}%]  | {c_win_rate:>6.1f}% [{c_ci_low}-{c_ci_high}%]")
        print(f"{'Total Titles Won':<32} | {d_titles:>6} / {num_sessions:<9} | {c_titles:>6} / {num_sessions:<9}")
        print(f"{'BB/100 Win Rate':<32} | {d_bb_100:>+10.2f} BB/100   | {c_bb_100:>+10.2f} BB/100")
        print(f"{'Average Final Chip Share':<32} | {d_avg_share:>6.1f}% (Par: 20%)   | {c_avg_share:>6.1f}% (Par: 20%)")
        print(f"{'Tournament Survival Rate':<32} | {d_surv_rate:>6.1f}%              | {c_surv_rate:>6.1f}%")
        if direct_duels_d_won + direct_duels_c_won > 0:
            print(f"{'Final-2 Duel Showdowns':<32} | {direct_duels_d_won:>6} wins             | {direct_duels_c_won:>6} wins")
        print("-" * 76)
        print(f"{'Model B (Baseline Anchor) Rate':<32} | {b_win_rate:>6.1f}% ({b_titles}/{num_sessions})")
        print(f"{'Adversarial Exploiter / Archetypes':<32} | {other_win_rate:>6.1f}% ({other_titles}/{num_sessions})")
        print("=" * 76)
        print(f"\n  Two-Proportion Z-Test: z = {z_stat:.3f}, p-value = {p_val:.4f}")
        print(f"  VERDICT:        {verdict}")
        print(f"  RECOMMENDATION: {recommendation}\n")
        print("=" * 76 + "\n")

    os.makedirs("results", exist_ok=True)
    out_file = "results/comparison_c_vs_d.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"  Saved full telemetry report to: {out_file}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Head-to-Head Tournament Comparison: Model C vs Model D")
    parser.add_argument("--sessions", type=int, default=50, help="Number of tournament sessions to simulate (default: 50)")
    parser.add_argument("--model-d", type=str, default="models/model_d.zip", help="Path to Model D checkpoint")
    parser.add_argument("--model-c", type=str, default="models/model_c.zip", help="Path to Model C checkpoint")
    parser.add_argument("--model-b", type=str, default="models/model_b.zip", help="Path to Model B checkpoint")
    parser.add_argument("--chips", type=int, default=200, help="Starting stack per seat")
    parser.add_argument("--blind-escalation", type=int, default=15, help="Hands between blind escalations")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    run_head_to_head_comparison(
        model_d_path=args.model_d,
        model_c_path=args.model_c,
        model_b_path=args.model_b,
        num_sessions=args.sessions,
        starting_chips=args.chips,
        blind_escalation=args.blind_escalation,
        seed=args.seed,
    )
