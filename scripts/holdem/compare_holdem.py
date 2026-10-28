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
    league_mode: bool = False,
    verbose: bool = True,
) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Hold'em model checkpoint not found: {model_path}")

    from core.neural.card_attention import CardAttentionExtractor
    import core.neural.card_attention
    sys.modules["card_attention"] = core.neural.card_attention
    from rl.holdem_decoupled_actor import HoldemDecoupledActorCriticPolicy

    model = MaskablePPO.load(model_path)
    obs_dim = getattr(model.observation_space, "shape", [119])[0]
    if obs_dim == 168:
        obs_version = "f"
    elif obs_dim == 158:
        obs_version = "e"
    elif obs_dim == 152:
        obs_version = "d"
    elif obs_dim in (123, 138):
        obs_version = "c"
    elif obs_dim in (119, 125):
        obs_version = "b"
    else:
        obs_version = "a"

    league_pool = None
    if league_mode:
        from rl.holdem_league import HoldemLeaguePool
        league_pool = HoldemLeaguePool(
            league_dir="models/holdem/league",
            anchor_model_path="models/holdem/model_holdem_d_best.zip",
        )

    env = HoldemPokerGymEnv(
        num_seats=5,
        starting_chips=starting_chips,
        small_blind=1,
        big_blind=2,
        hero_seat=0,
        max_hands_per_session=150,
        blind_escalation_interval=blind_escalation,
        randomize_stacks=True,
        dominated_masking=True,
        pos_gto_masking=(obs_version in ("c", "d", "e", "f")),
        exploitative_masking=(obs_version in ("d", "e", "f")),
        obs_version=obs_version,
        league_mode=league_mode,
        league_pool=league_pool,
        rng_seed=seed,
    )

    titles = {"model_holdem": 0, "TAGBot": 0, "LAGBot": 0, "CallingStation": 0, "RandomBot": 0}
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
        if not league_mode:
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

        if winner == 0:
            titles["model_holdem"] = titles.get("model_holdem", 0) + 1
        elif winner in env.opponents:
            opp_name = env.opponents[winner].name
            titles[opp_name] = titles.get(opp_name, 0) + 1
        else:
            titles["other"] = titles.get("other", 0) + 1

        if verbose and (sess + 1) % max(1, num_sessions // 5) == 0:
            pct = (titles["model_holdem"] / (sess + 1)) * 100.0
            print(f"  [Tourney {sess+1:3d}/{num_sessions}] Model Hold'em: {pct:4.1f}% ({titles['model_holdem']}) | TAG: {titles.get('TAGBot', 0)} | LAG: {titles.get('LAGBot', 0)}")

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
    suffix = "_league" if league_mode else "_bots"
    out_file = f"results/holdem/comparison_holdem_{obs_version}{suffix}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(f"results/holdem/comparison_holdem_{obs_version}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Texas Hold'em Tournament Benchmark")
    parser.add_argument("--sessions", type=int, default=50)
    parser.add_argument("--model", type=str, default="models/holdem/model_holdem_d_best.zip")
    parser.add_argument("--league", action="store_true", help="Spar against jittered league pool")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_benchmark_holdem(
        model_path=args.model,
        num_sessions=args.sessions,
        league_mode=args.league,
        seed=args.seed,
    )
