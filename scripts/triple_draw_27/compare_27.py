"""
scripts/triple_draw_27/compare_27.py — 2-7 Triple Draw Tournament Benchmark Suite.

Evaluates Model 27-A against table opponents:
- Seat 0: Model 27-A (Hero)
- Seat 1: PatStandBot
- Seat 2: SmoothDrawBot
- Seat 3: LowballExploiter
- Seat 4: Random Lowball Archetype
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
from games.triple_draw_27.gym_wrapper import TripleDrawPokerGymEnv
from games.triple_draw_27.opponents import (
    PatStandBot,
    SmoothDrawBot,
    LowballExploiter,
    make_random_lowball_opponent,
)
import core.neural.card_attention
import core.neural.decoupled_actor
sys.modules["card_attention"] = core.neural.card_attention
sys.modules["decoupled_actor"] = core.neural.decoupled_actor


def run_benchmark_27(
    model_path: str = "models/triple_draw_27/model_27_a.zip",
    num_sessions: int = 100,
    starting_chips: int = 200,
    blind_escalation: int = 15,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"2-7 Lowball model checkpoint not found: {model_path}")

    model = MaskablePPO.load(model_path)
    env = TripleDrawPokerGymEnv(
        num_seats=5,
        starting_chips=starting_chips,
        small_blind=1,
        big_blind=2,
        hero_seat=0,
        max_hands_per_session=150,
        blind_escalation_interval=blind_escalation,
        randomize_stacks=True,
        discard_masking=True,
        rng_seed=seed,
    )

    titles = {"model_27": 0, "pat_bot": 0, "smooth_bot": 0, "exploiter": 0, "archetype": 0}
    total_hands = 0
    total_table_chips = starting_chips * 5

    if verbose:
        print("\n" + "=" * 80)
        print("          CARDSHARK-RL: 2-7 TRIPLE DRAW TOURNAMENT BENCHMARK")
        print("=" * 80)
        print(f"  Model Policy:              {model_path}")
        print(f"  Table Field:               PatStandBot, SmoothDrawBot, LowballExploiter, Random")
        print(f"  Configuration:            {num_sessions} sessions | 5 seats | {starting_chips} chips")
        print("=" * 80 + "\n")

    for sess in range(num_sessions):
        obs, info = env.reset(seed=seed + sess * 73)
        env.opponents[1] = PatStandBot(opponent_id=1, rng=np.random.default_rng(seed + sess * 11))
        env.opponents[2] = SmoothDrawBot(opponent_id=2, rng=np.random.default_rng(seed + sess * 23))
        env.opponents[3] = LowballExploiter(opponent_id=3, rng=np.random.default_rng(seed + sess * 37))
        env.opponents[4] = make_random_lowball_opponent(opp_id=4, rng=np.random.default_rng(seed + sess * 51))

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
                if env.env.phase in ("draw_1", "draw_2", "draw_3"):
                    env.env._advance_draw_queue()
                else:
                    env.env._advance_betting()
                continue

            opp = env.opponents.get(cur)
            if env.env.phase in ("bet_1", "bet_2", "bet_3", "bet_4"):
                act = opp.bet_action(
                    hand=env.env.seats[cur].hand,
                    pot=env.env.pot,
                    bet_to_call=env.env.seats[cur].bet_to_call,
                    phase=env.env.phase,
                    stack=env.env.seats[cur].chips,
                    legal_actions=legal,
                ) if opp else (1 if 1 in legal else legal[0])
            else:
                draw_round = 0 if env.env.phase == "draw_1" else (1 if env.env.phase == "draw_2" else 2)
                discards = opp.draw_action(env.env.seats[cur].hand, draw_round=draw_round) if opp else []
                act = 6 + sum(1 << d for d in discards)

            env.env.step_action(cur, act)

        total_hands += env.env.hands_played
        chips = [s.chips for s in env.env.seats]
        winner = int(np.argmax(chips))

        if winner == 0:
            titles["model_27"] += 1
        elif winner == 1:
            titles["pat_bot"] += 1
        elif winner == 2:
            titles["smooth_bot"] += 1
        elif winner == 3:
            titles["exploiter"] += 1
        else:
            titles["archetype"] += 1

        if verbose and (sess + 1) % max(1, num_sessions // 5) == 0:
            pct = (titles["model_27"] / (sess + 1)) * 100.0
            print(f"  [Tourney {sess+1:3d}/{num_sessions}] Model 27-A: {pct:4.1f}% ({titles['model_27']}) | Exploiter: {titles['exploiter']} | Pat: {titles['pat_bot']}")

    win_rate = (titles["model_27"] / num_sessions) * 100.0
    ci_low, ci_high = wilson_score_interval(titles["model_27"], num_sessions)

    results = {
        "num_sessions": num_sessions,
        "total_hands": total_hands,
        "model_27": {
            "titles": titles["model_27"],
            "win_rate_pct": round(win_rate, 1),
            "ci_95": [ci_low, ci_high],
        },
        "opponents": titles,
    }

    if verbose:
        print("\n" + "=" * 80)
        print(f"  Model 27-A Win Rate: {win_rate:.1f}% [{ci_low:.1f}% - {ci_high:.1f}%] ({titles['model_27']}/{num_sessions})")
        print("=" * 80 + "\n")

    os.makedirs("results/triple_draw_27", exist_ok=True)
    out_file = "results/triple_draw_27/comparison_27_vs_all.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-7 Triple Draw Tournament Benchmark")
    parser.add_argument("--sessions", type=int, default=100)
    parser.add_argument("--model", type=str, default="models/triple_draw_27/model_27_a.zip")
    args = parser.parse_args()

    run_benchmark_27(model_path=args.model, num_sessions=args.sessions)
