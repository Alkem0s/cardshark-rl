"""
scripts/holdem/duel_g_vs_f.py — Direct 1v1 Heads-Up Duel: Model G vs. Model F.

Pits Model G (1v1 Heads-Up Specialist) directly against Model F (Tournament Specialist)
in a dedicated 1v1 heads-up showdown across N matches.
Measures win rate, chip delta, match duration, and statistical significance.
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import core.neural.card_attention
sys.modules["card_attention"] = core.neural.card_attention
from sb3_contrib import MaskablePPO
from core.math_utils import wilson_score_interval, calculate_proportions_z_test
from games.holdem.gym_wrapper import HoldemPokerGymEnv
from rl.holdem_league import HoldemLeagueOpponent


def run_1v1_duel(
    model_g_path: str = "models/holdem/model_holdem_g_1v1_specialist.zip",
    model_f_path: str = "models/holdem/model_holdem_f_champion_2.5m.zip",
    num_matches: int = 50,
    starting_chips: int = 200,
    seed: int = 42,
) -> dict:
    print("=" * 80)
    print("       CARDSHARK-RL: DIRECT 1v1 HEADS-UP DUEL — MODEL G vs. MODEL F")
    print("=" * 80)
    print(f"  Hero (Seat 0):    Model G Specialist ({model_g_path})")
    print(f"  Rival (Seat 1):   Model F Champion   ({model_f_path})")
    print(f"  Matches:          {num_matches} heads-up duels | {starting_chips} starting chips")
    print("=" * 80 + "\n")

    model_g = MaskablePPO.load(model_g_path, device="cpu")

    env = HoldemPokerGymEnv(
        num_seats=5,
        starting_chips=starting_chips,
        small_blind=1,
        big_blind=2,
        hero_seat=0,
        max_hands_per_session=150,
        blind_escalation_interval=15,
        heads_up_mode=True,
        obs_version="f",
        rng_seed=seed,
    )

    g_wins = 0
    f_wins = 0
    g_chips = []
    hands_list = []

    for match in range(num_matches):
        m_seed = seed + match * 97
        obs, _ = env.reset(seed=m_seed)

        # Set Seat 1 to Model F
        env.opponents[1] = HoldemLeagueOpponent(
            model_path=model_f_path,
            opponent_id=1,
            name="Model_F",
            rng=np.random.default_rng(m_seed + 1),
            deterministic=True,
        )

        done = False
        while not done:
            mask = env.action_masks()
            action, _ = model_g.predict(obs, action_masks=mask, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

        hero_end_chips = env.env.seats[0].chips
        rival_end_chips = env.env.seats[1].chips
        g_chips.append(hero_end_chips - starting_chips)
        hands_list.append(env.env.hands_played)

        if hero_end_chips > rival_end_chips:
            g_wins += 1
        else:
            f_wins += 1

    win_rate_g = g_wins / num_matches
    win_rate_f = f_wins / num_matches
    ci_g = wilson_score_interval(g_wins, num_matches)
    ci_f = wilson_score_interval(f_wins, num_matches)
    z_stat, p_val = calculate_proportions_z_test(g_wins, num_matches, f_wins, num_matches)
    avg_chips = float(np.mean(g_chips))
    avg_hands = float(np.mean(hands_list))

    print(f"  Final Results ({num_matches} matches):")
    print(f"  Model G (1v1 Specialist) Wins: {g_wins}/{num_matches} ({win_rate_g*100:.1f}%) [95% CI: {ci_g[0]*100:.1f}% - {ci_g[1]*100:.1f}%]")
    print(f"  Model F (Tournament Core) Wins: {f_wins}/{num_matches} ({win_rate_f*100:.1f}%) [95% CI: {ci_f[0]*100:.1f}% - {ci_f[1]*100:.1f}%]")
    print(f"  Model G Avg Net Chips:          {avg_chips:+.1f} chips")
    print(f"  Avg Hands Per Match:            {avg_hands:.1f} hands")
    print(f"  Statistical Significance:       Z = {z_stat:.3f}, p-value = {p_val:.5f}")
    if p_val < 0.05:
        print("  ==> Statistically Significant Difference (p < 0.05)")
    else:
        print("  ==> Difference is within standard statistical bounds")
    print("=" * 80)

    results = {
        "num_matches": num_matches,
        "model_g": {
            "wins": g_wins,
            "win_rate": win_rate_g,
            "ci_95": list(ci_g),
            "avg_net_chips": avg_chips,
        },
        "model_f": {
            "wins": f_wins,
            "win_rate": win_rate_f,
            "ci_95": list(ci_f),
            "avg_net_chips": -avg_chips,
        },
        "avg_hands": avg_hands,
        "hypothesis_test": {
            "z_statistic": float(z_stat),
            "p_value": float(p_val),
            "significant_at_05": bool(p_val < 0.05),
        },
    }

    out_path = os.path.join(PROJECT_ROOT, "results", "holdem", "duel_holdem_g_vs_f.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Duel results saved to: {out_path}\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct 1v1 Duel: Model G vs Model F")
    parser.add_argument("--matches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_1v1_duel(num_matches=args.matches, seed=args.seed)

