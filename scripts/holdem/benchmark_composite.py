"""
scripts/holdem/benchmark_composite.py — Standardized Benchmark for Ablation 6.

Evaluates and compares:
1. Monolithic Model E (Previous 158-dim champion)
2. Monolithic Model F (168-dim tournament specialist, 2.5M timesteps)
3. Model G alone (168-dim 1v1 heads-up specialist)
4. Composite Model (Model F in Tournament slot + Model G in Heads-Up slot)

Runs across N standardized Sit-and-Go tournaments against the identical benchmark bot field:
- TAGBot (Tight-Aggressive)
- LAGBot (Loose-Aggressive)
- CallingStationBot (Passive Station)
- RandomHoldemBot (Noise)

Measures:
- 1st-Place Title Win Rate
- Heads-Up Appearances (Top-2 Finishes)
- Heads-Up Title Conversion Rate (Titles / HU Appearances)
- Podium Finish Rate (Top-3)
- Zero Early Elimination Rate
- Average Net Chips & Finish Rank
"""

from __future__ import annotations
import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List, Optional

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sb3_contrib import MaskablePPO
from core.neural.composite_policy import HoldemCompositeModel
from games.holdem.gym_wrapper import HoldemPokerGymEnv


def evaluate_agent_tournaments(
    agent_name: str,
    predict_fn: Any,
    obs_version: str = "f",
    n_tournaments: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates an agent (either an SB3 model or HoldemCompositeModel)
    across n_tournaments in the standard 5-player Sit-and-Go environment.
    """
    env = HoldemPokerGymEnv(
        num_seats=5,
        starting_chips=200,
        small_blind=1,
        big_blind=2,
        hero_seat=0,
        max_hands_per_session=150,
        blind_escalation_interval=15,
        obs_version=obs_version,
        pos_gto_masking=True,
        exploitative_masking=True,
        uncontested_pot_bonus=0.015 if obs_version == "f" else 0.0,
        rng_seed=seed,
    )

    wins = 0
    top2_appearances = 0
    podiums = 0
    eliminations = 0
    net_chips: List[int] = []
    finish_ranks: List[int] = []
    hands_played_list: List[int] = []

    print(f"\n  Evaluating [{agent_name}] across {n_tournaments} tournaments (obs_version='{obs_version}')...")
    start_t = time.time()

    for s_idx in range(n_tournaments):
        session_seed = seed + s_idx * 77
        obs, _ = env.reset(seed=session_seed)
        done = False

        while not done:
            mask = env.action_masks()
            act, _ = predict_fn(obs, action_masks=mask, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(act))
            done = terminated or truncated

        hero_chips = env.env.seats[0].chips
        net_chips.append(hero_chips - 200)
        hands_played_list.append(env.env.hands_played)

        # Rank determination
        alive_seats = sorted([s for s in env.env.seats], key=lambda x: x.chips, reverse=True)
        hero_rank = [s.seat_idx for s in alive_seats].index(0) + 1
        finish_ranks.append(hero_rank)

        if hero_rank == 1:
            wins += 1
        if hero_rank <= 2:
            top2_appearances += 1
        if hero_rank <= 3:
            podiums += 1
        if hero_chips == 0:
            eliminations += 1

    elapsed = time.time() - start_t
    win_rate = float(wins) / float(n_tournaments)
    top2_rate = float(top2_appearances) / float(n_tournaments)
    hu_conversion = (float(wins) / float(top2_appearances)) if top2_appearances > 0 else 0.0
    podium_rate = float(podiums) / float(n_tournaments)
    elim_rate = float(eliminations) / float(n_tournaments)
    avg_chips = float(np.mean(net_chips))
    avg_rank = float(np.mean(finish_ranks))
    score = (win_rate * 100.0) + (avg_chips / 10.0)

    # 95% Wilson Score Confidence Interval for win rate
    z = 1.96
    p = win_rate
    n = n_tournaments
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    ci_low = max(0.0, float(center - margin))
    ci_high = min(1.0, float(center + margin))

    print(f"    Titles Won:      {wins}/{n_tournaments} ({win_rate*100:.1f}%) [95% CI: {ci_low*100:.1f}% - {ci_high*100:.1f}%]")
    print(f"    Heads-Up (Top-2): {top2_appearances}/{n_tournaments} ({top2_rate*100:.1f}%)")
    print(f"    HU Conversion:   {hu_conversion*100:.1f}% (Titles / Top-2)")
    print(f"    Podium Rate:     {podium_rate*100:.1f}% ({podiums}/{n_tournaments})")
    print(f"    Elimination:     {elim_rate*100:.1f}% ({eliminations}/{n_tournaments})")
    print(f"    Avg Finish Rank: {avg_rank:.2f}")
    print(f"    Avg Net Chips:   {avg_chips:+.1f} chips")
    print(f"    Benchmark Score: {score:.2f}")
    print(f"    Elapsed Time:    {elapsed:.1f}s ({elapsed/n_tournaments:.2f}s per tournament)")

    return {
        "agent": agent_name,
        "n_tournaments": n_tournaments,
        "wins": wins,
        "win_rate": win_rate,
        "ci_95": [ci_low, ci_high],
        "top2_appearances": top2_appearances,
        "top2_rate": top2_rate,
        "hu_conversion_rate": hu_conversion,
        "podiums": podiums,
        "podium_rate": podium_rate,
        "eliminations": eliminations,
        "elimination_rate": elim_rate,
        "avg_chips": avg_chips,
        "avg_rank": avg_rank,
        "avg_hands": float(np.mean(hands_played_list)),
        "score": score,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Ablation 6: Monolithic vs. Composite Models")
    parser.add_argument("--tournaments", type=int, default=50, help="Number of tournaments per evaluation")
    parser.add_argument("--model-f-path", type=str, default="models/holdem/model_holdem_f_champion_2.5m.zip")
    parser.add_argument("--model-g-path", type=str, default="models/holdem/model_holdem_g_1v1_specialist.zip")
    parser.add_argument("--model-e-path", type=str, default="models/holdem/model_holdem_e_best.zip")
    parser.add_argument("--include-model-e", action="store_true", help="Include legacy Model E in benchmark")
    parser.add_argument("--output-json", type=str, default="results/holdem/benchmark_composite_ablation6.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_config": {
            "n_tournaments": args.tournaments,
            "seed": args.seed,
            "bot_field": ["TAGBot", "LAGBot", "CallingStationBot", "RandomHoldemBot"],
        },
        "agents": {},
    }

    # 1. Monolithic Model F (Tournament Specialist)
    if os.path.exists(args.model_f_path):
        model_f = MaskablePPO.load(args.model_f_path, device="cpu")
        res_f = evaluate_agent_tournaments(
            agent_name="Model F (Tournament Specialist - Monolithic)",
            predict_fn=model_f.predict,
            obs_version="f",
            n_tournaments=args.tournaments,
            seed=args.seed,
        )
        results["agents"]["model_f_monolithic"] = res_f
    else:
        print(f"[Warning] Model F checkpoint not found at: {args.model_f_path}")

    # 2. Model G (1v1 Specialist alone)
    if os.path.exists(args.model_g_path):
        model_g = MaskablePPO.load(args.model_g_path, device="cpu")
        res_g = evaluate_agent_tournaments(
            agent_name="Model G (1v1 Specialist - Alone)",
            predict_fn=model_g.predict,
            obs_version="f",
            n_tournaments=args.tournaments,
            seed=args.seed,
        )
        results["agents"]["model_g_alone"] = res_g
    else:
        print(f"[Notice] Model G checkpoint not yet found at: {args.model_g_path} (train Model G first)")

    # 3. Composite Model (Model F + Model G)
    if os.path.exists(args.model_f_path) and os.path.exists(args.model_g_path):
        composite_model = HoldemCompositeModel.load(
            tournament_model_path=args.model_f_path,
            heads_up_model_path=args.model_g_path,
            device="cpu",
            name="Composite [Model F + Model G]",
        )
        res_comp = evaluate_agent_tournaments(
            agent_name="Composite Model [Tournament: Model F + 1v1: Model G]",
            predict_fn=composite_model.predict,
            obs_version="f",
            n_tournaments=args.tournaments,
            seed=args.seed,
        )
        results["agents"]["composite_f_plus_g"] = res_comp

    # 4. (Optional) Legacy Model E
    if args.include_model_e and os.path.exists(args.model_e_path):
        model_e = MaskablePPO.load(args.model_e_path, device="cpu")
        res_e = evaluate_agent_tournaments(
            agent_name="Model E (Legacy Champion - Monolithic)",
            predict_fn=model_e.predict,
            obs_version="e",
            n_tournaments=args.tournaments,
            seed=args.seed,
        )
        results["agents"]["model_e_monolithic"] = res_e

    # Save output
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Benchmark Complete] Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()

