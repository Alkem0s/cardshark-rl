"""
scripts/holdem/duel_e_vs_f.py — Head-to-Head Tournament Duel: Model F vs. Model E.

Simulates 5-player tournaments where Model F and Model E compete at the same table
alongside the standard benchmark bot field (TAGBot, LAGBot, CallingStationBot).
Evaluates head-to-head title capture, outlast rate, average finish rank, and statistical significance.
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
from games.holdem.opponents import TAGBot, LAGBot, CallingStationBot
from rl.holdem_league import HoldemLeagueOpponent
from games.holdem.action_abstraction import A_CHECK_CALL


def run_duel(
    model_f_path: str = "models/holdem/model_holdem_f_champion_2.5m.zip",
    model_e_path: str = "models/holdem/model_holdem_e_best.zip",
    num_sessions: int = 50,
    starting_chips: int = 200,
    blind_escalation: int = 15,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    print("=" * 80)
    print("       CARDSHARK-RL: HEAD-TO-HEAD DUEL — MODEL F vs. MODEL E")
    print("=" * 80)
    print(f"  Hero (Seat 0):    Model F ({model_f_path})")
    print(f"  Rival (Seat 1):   Model E ({model_e_path})")
    print(f"  Bot Field:        TAGBot (Seat 2), LAGBot (Seat 3), CallingStation (Seat 4)")
    print(f"  Tournaments:      {num_sessions} sessions | 5 seats | {starting_chips} starting chips")
    print("=" * 80 + "\n")

    model_f = MaskablePPO.load(model_f_path)

    env = HoldemPokerGymEnv(
        num_seats=5,
        starting_chips=starting_chips,
        small_blind=1,
        big_blind=2,
        hero_seat=0,
        max_hands_per_session=150,
        blind_escalation_interval=blind_escalation,
        randomize_stacks=True,
        pos_gto_masking=True,
        exploitative_masking=True,
        obs_version="f",
        rng_seed=seed,
    )

    titles = {"model_f": 0, "model_e": 0, "TAGBot": 0, "LAGBot": 0, "CallingStation": 0}
    f_ranks = []
    e_ranks = []
    f_outlasted_e = 0
    e_outlasted_f = 0
    total_hands = 0

    for sess in range(num_sessions):
        s_seed = seed + sess * 73
        obs, info = env.reset(seed=s_seed)

        # Setup table opponents
        env.opponents[1] = HoldemLeagueOpponent(
            model_path=model_e_path,
            opponent_id=1,
            name="Model_E",
            rng=np.random.default_rng(s_seed + 1),
            deterministic=True,
        )
        env.opponents[2] = TAGBot(opponent_id=2, rng=np.random.default_rng(s_seed + 2))
        env.opponents[3] = LAGBot(opponent_id=3, rng=np.random.default_rng(s_seed + 3))
        env.opponents[4] = CallingStationBot(opponent_id=4, rng=np.random.default_rng(s_seed + 4))

        done = False
        while not done:
            mask = env.action_masks()
            action, _ = model_f.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        # Complete tournament if hero eliminated early
        while not env.env.session_done:
            cur = env.env.current_seat
            legal = env.env.get_legal_actions(cur)
            if not legal:
                env.env._advance_betting()
                continue

            opp = env.opponents.get(cur)
            if opp:
                if hasattr(opp, "act_with_env"):
                    act = opp.act_with_env(env.env, cur, legal)
                else:
                    act = opp.act(
                        hole_cards=env.env.seats[cur].hand,
                        board=env.env.board,
                        pot=env.env.pot,
                        bet_to_call=env.env.seats[cur].bet_to_call,
                        street=env.env.street,
                        stack=env.env.seats[cur].chips,
                        legal_actions=legal,
                    )
            else:
                act = A_CHECK_CALL if A_CHECK_CALL in legal else legal[0]

            env.env.step_action(cur, act)

        total_hands += env.env.hands_played
        chips = [s.chips for s in env.env.seats]
        winner = int(np.argmax(chips))

        # Track title winner
        if winner == 0:
            titles["model_f"] += 1
        elif winner == 1:
            titles["model_e"] += 1
        elif winner in env.opponents:
            titles[env.opponents[winner].name] = titles.get(env.opponents[winner].name, 0) + 1

        # Track ranks
        sorted_seats = sorted(env.env.seats, key=lambda s: s.chips, reverse=True)
        r_map = {s.seat_idx: r + 1 for r, s in enumerate(sorted_seats)}
        f_rank = r_map[0]
        e_rank = r_map[1]
        f_ranks.append(f_rank)
        e_ranks.append(e_rank)

        if f_rank < e_rank:
            f_outlasted_e += 1
        elif e_rank < f_rank:
            e_outlasted_f += 1

        if verbose and (sess + 1) % max(1, num_sessions // 5) == 0:
            print(f"  [Tourney {sess+1:2d}/{num_sessions}] Model F Titles: {titles['model_f']} | Model E Titles: {titles['model_e']} | Bots: {sess+1 - titles['model_f'] - titles['model_e']}")

    f_win_pct = (titles["model_f"] / num_sessions) * 100.0
    e_win_pct = (titles["model_e"] / num_sessions) * 100.0
    f_ci = wilson_score_interval(titles["model_f"], num_sessions)
    e_ci = wilson_score_interval(titles["model_e"], num_sessions)

    # Statistical test between F and E
    z_stat, p_val = calculate_proportions_z_test(titles["model_f"], num_sessions, titles["model_e"], num_sessions)

    results = {
        "num_sessions": num_sessions,
        "total_hands": total_hands,
        "model_f": {
            "titles": titles["model_f"],
            "win_rate_pct": round(f_win_pct, 1),
            "ci_95": f_ci,
            "avg_rank": round(float(np.mean(f_ranks)), 2),
            "outlasted_rival": f_outlasted_e,
        },
        "model_e": {
            "titles": titles["model_e"],
            "win_rate_pct": round(e_win_pct, 1),
            "ci_95": e_ci,
            "avg_rank": round(float(np.mean(e_ranks)), 2),
            "outlasted_rival": e_outlasted_f,
        },
        "all_titles": titles,
        "statistical_test": {
            "z_statistic": round(z_stat, 3),
            "p_value": round(p_val, 4),
            "significant_at_05": p_val < 0.05,
        },
    }

    print("\n" + "=" * 80)
    print("                       HEAD-TO-HEAD DUEL SUMMARY")
    print("=" * 80)
    print(f"  Model F:  {titles['model_f']} Titles ({f_win_pct:.1f}%) | Avg Rank: {results['model_f']['avg_rank']} | Outlasted Model E: {f_outlasted_e}/{num_sessions}")
    print(f"  Model E:  {titles['model_e']} Titles ({e_win_pct:.1f}%) | Avg Rank: {results['model_e']['avg_rank']} | Outlasted Model F: {e_outlasted_f}/{num_sessions}")
    print(f"  Bots:     TAG: {titles.get('TAGBot', 0)} | LAG: {titles.get('LAGBot', 0)} | Station: {titles.get('CallingStation', 0)}")
    print(f"  Z-Test:   Z = {z_stat:.3f}, p = {p_val:.4f} (Significant at 0.05: {p_val < 0.05})")
    print("=" * 80 + "\n")

    os.makedirs("results/holdem", exist_ok=True)
    out_file = "results/holdem/duel_holdem_f_vs_e.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  [Saved] Duel results saved to: {out_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Texas Hold'em Duel: Model F vs Model E")
    parser.add_argument("--sessions", type=int, default=50)
    parser.add_argument("--model-f", type=str, default="models/holdem/model_holdem_f_champion_2.5m.zip")
    parser.add_argument("--model-e", type=str, default="models/holdem/model_holdem_e_best.zip")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_duel(
        model_f_path=args.model_f,
        model_e_path=args.model_e,
        num_sessions=args.sessions,
        seed=args.seed,
    )

