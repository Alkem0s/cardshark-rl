"""
scripts/holdem/compare_holdem.py — No-Limit Texas Hold'em Tournament Benchmark Suite.

Evaluates Model Holdem-A against table opponents:
- Seat 0: Model Holdem-A (Hero)
- Seat 1: TAGBot (Tight-Aggressive)
- Seat 2: LAGBot (Loose-Aggressive)
- Seat 3: CallingStationBot (Passive Station)
- Seat 4: RandomHoldemBot (Uniform Random)
"""

from __future__ import annotations
import os
import sys
import math
import json
import argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sb3_contrib import MaskablePPO
from core.math_utils import wilson_score_interval, calculate_proportions_z_test
from games.holdem.gym_wrapper import HoldemPokerGymEnv
from games.holdem.opponents import (
    TAGBot,
    LAGBot,
    CallingStationBot,
    RandomHoldemBot,
    make_random_holdem_opponent,
)
from games.holdem.action_abstraction import A_CHECK_CALL


def run_benchmark_holdem(
    model_path: str = "models/holdem/model_holdem_a.zip",
    num_sessions: int = 50,
    starting_chips: int = 200,
    blind_escalation: int = 15,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Hold'em model checkpoint not found: {model_path}")

    model = MaskablePPO.load(model_path)
    env = HoldemPokerGymEnv(
        num_seats=5,
        starting_chips=starting_chips,
        small_blind=1,
        big_blind=2,
        hero_seat=0,
        max_hands_per_session=150,
        blind_escalation_interval=blind_escalation,
        randomize_stacks=True,
        rng_seed=seed,
    )

    titles = {"model_holdem": 0, "tag_bot": 0, "lag_bot": 0, "station": 0, "random": 0}
    total_hands = 0

    if verbose:
        print("\n" + "=" * 80)
        print("          CARDSHARK-RL: TEXAS HOLD'EM TOURNAMENT BENCHMARK")
        print("=" * 80)
        print(f"  Model Policy:              {model_path}")
        print(f"  Table Field:               TAGBot, LAGBot, CallingStationBot, RandomBot")
        print(f"  Configuration:            {num_sessions} sessions | 5 seats | {starting_chips} chips")
        print("=" * 80 + "\n")

    for sess in range(num_sessions):
        obs, info = env.reset(seed=seed + sess * 73)
        env.opponents[1] = TAGBot(opponent_id=1, rng=np.random.default_rng(seed + sess * 11))
        env.opponents[2] = LAGBot(opponent_id=2, rng=np.random.default_rng(seed + sess * 23))
        env.opponents[3] = CallingStationBot(opponent_id=3, rng=np.random.default_rng(seed + sess * 37))
        env.opponents[4] = RandomHoldemBot(opponent_id=4, rng=np.random.default_rng(seed + sess * 51))

        done = False
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
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

        if winner == 0:
            titles["model_holdem"] += 1
        elif winner == 1:
            titles["tag_bot"] += 1
        elif winner == 2:
            titles["lag_bot"] += 1
        elif winner == 3:
            titles["station"] += 1
        else:
            titles["random"] += 1

        if verbose and (sess + 1) % max(1, num_sessions // 5) == 0:
            pct = (titles["model_holdem"] / (sess + 1)) * 100.0
            print(f"  [Tourney {sess+1:3d}/{num_sessions}] Model Hold'em: {pct:4.1f}% ({titles['model_holdem']}) | TAG: {titles['tag_bot']} | LAG: {titles['lag_bot']}")

    win_rate = (titles["model_holdem"] / num_sessions) * 100.0
    ci_low, ci_high = wilson_score_interval(titles["model_holdem"], num_sessions)

    results = {
        "num_sessions": num_sessions,
        "total_hands": total_hands,
        "model_holdem": {
            "titles": titles["model_holdem"],
            "win_rate_pct": round(win_rate, 1),
            "ci_95": [ci_low, ci_high],
        },
        "opponents": titles,
    }

    if verbose:
        print("\n" + "=" * 80)
        print(f"  Model Hold'em Win Rate: {win_rate:.1f}% [{ci_low:.1f}% - {ci_high:.1f}%] ({titles['model_holdem']}/{num_sessions})")
        print("=" * 80 + "\n")

    os.makedirs("results/holdem", exist_ok=True)
    out_file = "results/holdem/comparison_holdem_vs_all.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Texas Hold'em Tournament Benchmark")
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument("--model", type=str, default="models/holdem/model_holdem_a.zip")
    args = parser.parse_args()

    run_benchmark_holdem(model_path=args.model, num_sessions=args.sessions)
