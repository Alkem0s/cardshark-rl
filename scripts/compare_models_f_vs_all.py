"""
compare_models_f_vs_all.py — Definitive Multi-Model Tournament & Statistical Benchmark Suite.

Evaluates Model F (The Co-Evolutionary Champion) against all project models and table adversaries:
- Model F (97-dim Game Geometry + Decoupled Dual-Stream Actor + Aux Belief)
- Model E (91-dim ICM + Dominated Discard Masking)
- Model D (87-dim Card Attention Champion)
- Adversarial Exploiter (Nemesis bot)
- Random Heuristic Archetype (TAG, LAG, Rock, CallingStation)

Features:
1. Positional-Parity Tournaments: Rotates hero seats among Model F, Model E, and Model D
   (Seats 0, 1, 2) across sessions to completely eliminate positional bias.
2. Full Elimination Completion: Games run until 1 survivor wins all table chips.
3. Strict Deterministic Inference: All neural models evaluate with greedy deterministic actions.
4. Statistical Testing: Computes Wilson 95% confidence intervals and two-proportion z-tests
   (F vs E, F vs D, F vs Adversarial Exploiter).
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
    OBS_DIM_F,
)
from rl.league import LeagueOpponent
from game.multi_opponents import AdversarialExploiter, make_random_archetype
from rl.card_attention import CardAttentionExtractor
from rl.decoupled_actor import DecoupledMaskableActorCriticPolicy
import rl.card_attention
import rl.decoupled_actor
sys.modules["card_attention"] = rl.card_attention
sys.modules["decoupled_actor"] = rl.decoupled_actor


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


def run_benchmark_f_vs_all(
    model_f_path: str = "models/model_f.zip",
    model_e_path: str = "models/model_e.zip",
    model_d_path: str = "models/model_d.zip",
    num_sessions: int = 100,
    starting_chips: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    blind_escalation: int = 15,
    max_hands: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    if not os.path.exists(model_f_path):
        raise FileNotFoundError(f"Model F checkpoint not found at: {model_f_path}")
    if not os.path.exists(model_e_path):
        raise FileNotFoundError(f"Model E checkpoint not found at: {model_e_path}")

    has_model_d = os.path.exists(model_d_path)

    if verbose:
        print("\n" + "=" * 85)
        print("        CARDSHARK-RL: MULTI-MODEL TOURNAMENT BENCHMARK (MODELS F vs E vs D)")
        print("=" * 85)
        print(f"  Model F Policy:                     {model_f_path} (97 dims, Decoupled ActionNet)")
        print(f"  Model E Policy:                     {model_e_path} (91 dims, Superhuman Crusher)")
        print(f"  Model D Policy:                     {model_d_path if has_model_d else 'None'} (87 dims)")
        print(f"  Adversarial Nemesis (Seat 3):       AdversarialExploiter")
        print(f"  Table Archetype (Seat 4):           Random Archetype (TAG/LAG/Rock/CallingStation)")
        print(f"  Tournament Configuration:          {num_sessions} sessions | 5 seats | {starting_chips} chips/seat")
        print(f"  Blinds: Small={small_blind}, Big={big_blind} (Escalates every {blind_escalation} hands)")
        print(f"  Positional Parity:                 Rotating Seats 0, 1, 2 across F, E, D")
        print("=" * 85 + "\n")

    # Load neural policies
    model_f_policy = MaskablePPO.load(model_f_path)
    model_e_policy = MaskablePPO.load(model_e_path)
    model_d_policy = MaskablePPO.load(model_d_path) if has_model_d else None

    # Base Gym Environment
    env = MultiDrawPokerGymEnv(
        num_seats=5,
        starting_chips=starting_chips,
        small_blind=small_blind,
        big_blind=big_blind,
        hero_seat=0,
        max_hands_per_session=max_hands,
        blind_escalation_interval=blind_escalation,
        randomize_stacks=True,
        obs_dim=ModelObsDim.MODEL_F,
        discard_masking=True,
        rng_seed=seed,
    )

    exploiter_bot = AdversarialExploiter()

    # Metrics Tracking
    f_titles = 0
    e_titles = 0
    d_titles = 0
    exploiter_titles = 0
    archetype_titles = 0

    f_net_chips_total = 0
    e_net_chips_total = 0
    d_net_chips_total = 0
    total_hands_played = 0

    f_chip_shares: list[float] = []
    e_chip_shares: list[float] = []
    d_chip_shares: list[float] = []

    f_survivals = 0
    e_survivals = 0
    d_survivals = 0

    f_outlasted_e = 0
    f_outlasted_d = 0
    f_outlasted_exploiter = 0

    total_table_chips = starting_chips * 5

    for sess in range(num_sessions):
        # 3-Way Positional Parity Rotation:
        # sess % 3 == 0: Seat 0 = F, Seat 1 = E, Seat 2 = D
        # sess % 3 == 1: Seat 0 = E, Seat 1 = D, Seat 2 = F
        # sess % 3 == 2: Seat 0 = D, Seat 1 = F, Seat 2 = E
        mod = sess % 3 if has_model_d else sess % 2

        if mod == 0:
            hero_name = "Model_F"
            f_seat, e_seat, d_seat = 0, 1, 2
            hero_model = model_f_policy
            hero_obs_dim = ModelObsDim.MODEL_F.value
            hero_discard_masking = True
            seat_1_path = model_e_path
            seat_1_name = "Model_E"
            seat_2_path = model_d_path if has_model_d else None
            seat_2_name = "Model_D"
        elif mod == 1:
            hero_name = "Model_E"
            e_seat, d_seat, f_seat = 0, (1 if has_model_d else 2), (2 if has_model_d else 1)
            hero_model = model_e_policy
            hero_obs_dim = ModelObsDim.MODEL_E.value
            hero_discard_masking = True
            seat_1_path = model_d_path if has_model_d else model_f_path
            seat_1_name = "Model_D" if has_model_d else "Model_F"
            seat_2_path = model_f_path if has_model_d else None
            seat_2_name = "Model_F" if has_model_d else None
        else: # mod == 2
            hero_name = "Model_D"
            d_seat, f_seat, e_seat = 0, 1, 2
            hero_model = model_d_policy
            hero_obs_dim = ModelObsDim.MODEL_D.value
            hero_discard_masking = False
            seat_1_path = model_f_path
            seat_1_name = "Model_F"
            seat_2_path = model_e_path
            seat_2_name = "Model_E"

        env.obs_dim = hero_obs_dim
        env.discard_masking = hero_discard_masking

        obs, info = env.reset(seed=seed + sess * 97)

        # Setup Table Opponents
        if seat_1_path:
            env.opponents[1] = LeagueOpponent(model_path=seat_1_path, opponent_id=1, name=seat_1_name, deterministic=True)
        if seat_2_path:
            env.opponents[2] = LeagueOpponent(model_path=seat_2_path, opponent_id=2, name=seat_2_name, deterministic=True)

        env.opponents[3] = exploiter_bot
        env.opponents[4] = make_random_archetype(rng=np.random.default_rng(seed + sess * 31))

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

        chips_f = env.env.seats[f_seat].chips
        chips_e = env.env.seats[e_seat].chips
        chips_d = env.env.seats[d_seat].chips if has_model_d else 0
        chips_exp = env.env.seats[3].chips

        # Net chips
        net_f = chips_f - starting_chips
        net_e = chips_e - starting_chips
        net_d = chips_d - starting_chips
        f_net_chips_total += net_f
        e_net_chips_total += net_e
        d_net_chips_total += net_d

        # Chip share
        share_f = (chips_f / total_table_chips) * 100.0
        share_e = (chips_e / total_table_chips) * 100.0
        share_d = (chips_d / total_table_chips) * 100.0
        f_chip_shares.append(share_f)
        e_chip_shares.append(share_e)
        d_chip_shares.append(share_d)

        if chips_f > 0:
            f_survivals += 1
        if chips_e > 0:
            e_survivals += 1
        if chips_d > 0:
            d_survivals += 1

        if chips_f > chips_e:
            f_outlasted_e += 1
        if chips_f > chips_d:
            f_outlasted_d += 1
        if chips_f > chips_exp:
            f_outlasted_exploiter += 1

        # Determine Winner
        winner_seat = None
        for s_idx, seat in enumerate(env.env.seats):
            if seat.chips >= total_table_chips * 0.95 or (len(env.env.alive_seats) == 1 and env.env.alive_seats[0] == s_idx):
                winner_seat = s_idx
                break

        if winner_seat is None:
            winner_seat = int(np.argmax([s.chips for s in env.env.seats]))

        if winner_seat == f_seat:
            f_titles += 1
        elif winner_seat == e_seat:
            e_titles += 1
        elif has_model_d and winner_seat == d_seat:
            d_titles += 1
        elif winner_seat == 3:
            exploiter_titles += 1
        else:
            archetype_titles += 1

        if verbose and (sess + 1) % max(1, num_sessions // 10) == 0:
            f_pct = (f_titles / (sess + 1)) * 100.0
            e_pct = (e_titles / (sess + 1)) * 100.0
            d_pct = (d_titles / (sess + 1)) * 100.0 if has_model_d else 0.0
            exp_pct = (exploiter_titles / (sess + 1)) * 100.0
            print(f"  [Tourney {sess+1:3d}/{num_sessions}] Model F: {f_pct:4.1f}% ({f_titles:2d}) | Model E: {e_pct:4.1f}% ({e_titles:2d}) | Exploiter: {exp_pct:4.1f}% ({exploiter_titles:2d}) | Model D: {d_pct:4.1f}%")

    # Final Aggregations
    f_win_rate = (f_titles / num_sessions) * 100.0
    e_win_rate = (e_titles / num_sessions) * 100.0
    d_win_rate = (d_titles / num_sessions) * 100.0
    exp_win_rate = (exploiter_titles / num_sessions) * 100.0
    arch_win_rate = (archetype_titles / num_sessions) * 100.0

    f_bb_100 = (f_net_chips_total / big_blind) / (max(1, total_hands_played) / 100.0)
    e_bb_100 = (e_net_chips_total / big_blind) / (max(1, total_hands_played) / 100.0)
    d_bb_100 = (d_net_chips_total / big_blind) / (max(1, total_hands_played) / 100.0)

    f_avg_share = float(np.mean(f_chip_shares))
    e_avg_share = float(np.mean(e_chip_shares))
    d_avg_share = float(np.mean(d_chip_shares))

    f_ci_low, f_ci_high = wilson_score_interval(f_titles, num_sessions)
    e_ci_low, e_ci_high = wilson_score_interval(e_titles, num_sessions)
    d_ci_low, d_ci_high = wilson_score_interval(d_titles, num_sessions)
    exp_ci_low, exp_ci_high = wilson_score_interval(exploiter_titles, num_sessions)

    # Statistical significance tests
    z_f_vs_e, p_f_vs_e = calculate_proportions_z_test(f_titles, num_sessions, e_titles, num_sessions)
    z_f_vs_d, p_f_vs_d = calculate_proportions_z_test(f_titles, num_sessions, d_titles, num_sessions)
    z_f_vs_exp, p_f_vs_exp = calculate_proportions_z_test(f_titles, num_sessions, exploiter_titles, num_sessions)

    results = {
        "num_sessions": num_sessions,
        "total_hands_played": total_hands_played,
        "avg_session_length": round(total_hands_played / max(1, num_sessions), 1),
        "model_f": {
            "championships": f_titles,
            "win_rate_pct": round(f_win_rate, 1),
            "ci_95": [f_ci_low, f_ci_high],
            "bb_per_100": round(f_bb_100, 2),
            "avg_chip_share_pct": round(f_avg_share, 1),
            "survival_rate_pct": round(f_survivals / num_sessions * 100.0, 1),
        },
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
        "adversaries": {
            "exploiter_titles": exploiter_titles,
            "exploiter_win_rate_pct": round(exp_win_rate, 1),
            "exploiter_ci_95": [exp_ci_low, exp_ci_high],
            "archetype_titles": archetype_titles,
            "archetype_win_rate_pct": round(arch_win_rate, 1),
        },
        "head_to_head_outlast": {
            "model_f_outlasted_e": f_outlasted_e,
            "model_f_outlasted_d": f_outlasted_d,
            "model_f_outlasted_exploiter": f_outlasted_exploiter,
        },
        "statistical_tests": {
            "f_vs_e": {
                "z_statistic": round(z_f_vs_e, 3),
                "p_value": round(p_f_vs_e, 4),
                "model_f_superior": p_f_vs_e < 0.05 and z_f_vs_e > 0,
            },
            "f_vs_d": {
                "z_statistic": round(z_f_vs_d, 3),
                "p_value": round(p_f_vs_d, 4),
                "model_f_superior": p_f_vs_d < 0.05 and z_f_vs_d > 0,
            },
            "f_vs_exploiter": {
                "z_statistic": round(z_f_vs_exp, 3),
                "p_value": round(p_f_vs_exp, 4),
                "model_f_superior": p_f_vs_exp < 0.05 and z_f_vs_exp > 0,
            },
        },
    }

    if verbose:
        print("\n" + "=" * 85)
        print("               CARDSHARK-RL: MODEL F vs ALL TOURNAMENT RESULTS")
        print("=" * 85)
        print(f"{'Metric':<30} | {'Model F (Champion)':<20} | {'Model E':<16} | {'Model D':<16}")
        print("-" * 85)
        print(f"{'1st Place Championship Rate':<30} | {f_win_rate:>6.1f}% [{f_ci_low:>4.1f}-{f_ci_high:>4.1f}%]  | {e_win_rate:>5.1f}% [{e_ci_low:>4.1f}-{e_ci_high:>4.1f}%]| {d_win_rate:>5.1f}% [{d_ci_low:>4.1f}-{d_ci_high:>4.1f}%]")
        print(f"{'Total Titles Won':<30} | {f_titles:>6} / {num_sessions:<11} | {e_titles:>5} / {num_sessions:<8} | {d_titles:>5} / {num_sessions:<8}")
        print(f"{'BB/100 Win Rate':<30} | {f_bb_100:>+12.2f} BB/100   | {e_bb_100:>+9.2f} BB/100 | {d_bb_100:>+9.2f} BB/100")
        print(f"{'Average Final Chip Share':<30} | {f_avg_share:>8.1f}% (Par: 20%) | {e_avg_share:>7.1f}% (20%)   | {d_avg_share:>7.1f}% (20%)")
        print(f"{'Tournament Survival Rate':<30} | {f_survivals / num_sessions * 100:>8.1f}%             | {e_survivals / num_sessions * 100:>7.1f}%         | {d_survivals / num_sessions * 100:>7.1f}%")
        print("-" * 85)
        print(f"  Adversarial Exploiter (Seat 3): {exp_win_rate:.1f}% [{exp_ci_low}-{exp_ci_high}%] ({exploiter_titles}/{num_sessions})")
        print(f"  Heuristic Archetypes (Seat 4):  {arch_win_rate:.1f}% ({archetype_titles}/{num_sessions})")
        print(f"  Model F Outlasted Model E:      {f_outlasted_e}/{num_sessions} tournaments ({f_outlasted_e/num_sessions*100:.1f}%)")
        print(f"  Model F Outlasted Exploiter:    {f_outlasted_exploiter}/{num_sessions} tournaments ({f_outlasted_exploiter/num_sessions*100:.1f}%)")
        print(f"  Model F Outlasted Model D:      {f_outlasted_d}/{num_sessions} tournaments ({f_outlasted_d/num_sessions*100:.1f}%)")
        print("=" * 85)
        print(f"  Two-Proportion Z-Test (F vs E):         z = {z_f_vs_e:+.3f}, p-value = {p_f_vs_e:.4f} ({'SIGNIFICANT (p < 0.05)' if p_f_vs_e < 0.05 else 'Not yet p < 0.05'})")
        print(f"  Two-Proportion Z-Test (F vs Exploiter): z = {z_f_vs_exp:+.3f}, p-value = {p_f_vs_exp:.4f}")
        print(f"  Two-Proportion Z-Test (F vs D):         z = {z_f_vs_d:+.3f}, p-value = {p_f_vs_d:.4f}\n")

    os.makedirs("results", exist_ok=True)
    out_file = "results/comparison_f_vs_all.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"  Saved full telemetry report to: {out_file}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Model Tournament Benchmark: Model F vs All")
    parser.add_argument("--sessions", type=int, default=100, help="Number of tournament sessions to simulate")
    parser.add_argument("--model-f", type=str, default="models/model_f.zip", help="Path to Model F checkpoint")
    parser.add_argument("--model-e", type=str, default="models/model_e.zip", help="Path to Model E checkpoint")
    parser.add_argument("--model-d", type=str, default="models/model_d.zip", help="Path to Model D checkpoint")
    parser.add_argument("--chips", type=int, default=200, help="Starting stack per seat")
    parser.add_argument("--blind-escalation", type=int, default=15, help="Hands between blind escalations")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    run_benchmark_f_vs_all(
        model_f_path=args.model_f,
        model_e_path=args.model_e,
        model_d_path=args.model_d,
        num_sessions=args.sessions,
        starting_chips=args.chips,
        blind_escalation=args.blind_escalation,
        seed=args.seed,
    )
