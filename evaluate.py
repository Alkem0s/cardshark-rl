"""
evaluate.py — Tournament runner, BB/100 calculation, and behavioral analysis.

Pits trained models against each archetype and a random baseline.
Produces the "Behavioral Matrix" (The Tell) for Model B.
"""

from __future__ import annotations
import os
import numpy as np
from collections import defaultdict
from typing import Optional

from visualizer import COLORS, OPP_NAMES, OPP_SHORT

from sb3_contrib import MaskablePPO

from gym_wrapper import DrawPokerGymEnv, mask_fn
from opponents import OPPONENT_CLASSES, NUM_ARCHETYPES
from draw_poker_env import A_FOLD, A_CALL, A_RAISE, A_DRAW_START


# ---------------------------------------------------------------------------
# Tournament runner
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tournament runner
# ---------------------------------------------------------------------------

def run_tournament(
    model: MaskablePPO,
    use_implicit: bool,
    opponent_id: int,
    num_hands: int = 10_000,
    seed: int = 0,
    collect_behavioral: bool = False,
) -> dict:
    """Run a model against a specific opponent for N hands.

    Returns dict with:
        'total_profit': int
        'bb_per_100': float
        'hand_profits': list[float]
        'post_draw_actions': list (if collect_behavioral)
    """
    env = DrawPokerGymEnv(
        use_implicit_modeling=use_implicit,
        opponent_id=opponent_id,
        rolling_window=50,
        rng_seed=seed,
    )

    hand_profits = []
    total_profit = 0

    # Warm up rolling stats for implicit model (play 50 random hands first)
    if use_implicit:
        for _ in range(50):
            obs, info = env.reset()
            done = False
            while not done:
                mask = env.action_mask()
                # Random valid action for warm-up
                valid = np.where(mask == 1)[0]
                action = int(np.random.choice(valid))
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
        env.clear_post_draw_actions()

    # Actual evaluation
    for hand_idx in range(num_hands):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            mask = env.action_mask()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        hand_profits.append(episode_reward)
        total_profit += episode_reward

    bb_per_100 = (total_profit / max(1, num_hands)) * 100

    result = {
        "total_profit": total_profit,
        "bb_per_100": bb_per_100,
        "hand_profits": hand_profits,
    }

    if collect_behavioral:
        result["post_draw_actions"] = env.get_post_draw_actions()

    env.close()
    return result


def run_random_baseline(
    opponent_id: int,
    num_hands: int = 10_000,
    seed: int = 0,
) -> dict:
    """Run a random agent against a specific opponent for N hands."""
    env = DrawPokerGymEnv(
        use_implicit_modeling=False,
        opponent_id=opponent_id,
        rolling_window=50,
        rng_seed=seed,
    )
    rng = np.random.default_rng(seed)

    hand_profits = []
    total_profit = 0

    for _ in range(num_hands):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            mask = env.action_mask()
            valid = np.where(mask == 1)[0]
            action = int(rng.choice(valid))
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        hand_profits.append(episode_reward)
        total_profit += episode_reward

    bb_per_100 = (total_profit / max(1, num_hands)) * 100

    env.close()
    return {
        "total_profit": total_profit,
        "bb_per_100": bb_per_100,
        "hand_profits": hand_profits,
    }


# ---------------------------------------------------------------------------
# Full cross-evaluation
# ---------------------------------------------------------------------------

def full_evaluation(
    model_a: MaskablePPO,
    model_b: MaskablePPO,
    num_hands: int = 10_000,
    seed: int = 0,
    results_dir: str = "results",
    eval_seeds: list[int] | None = None,
) -> dict:
    """Run full cross-evaluation tournament, optionally over multiple seeds.

    When eval_seeds is provided (e.g. [0, 1, 2]), the tournament is run once
    per seed and the results are aggregated to give mean ± std BB/100,
    providing confidence intervals around each matchup score.

    Args:
        eval_seeds: List of RNG seeds to evaluate. Defaults to [seed].

    Returns:
        Results dict. Each entry has 'bb_per_100' (mean across seeds)
        and optionally 'bb_per_100_std', 'bb_per_100_all' (per-seed list).
    """
    os.makedirs(results_dir, exist_ok=True)

    if eval_seeds is None:
        eval_seeds = [seed]

    n_seeds = len(eval_seeds)
    print(f"\n{'='*60}")
    print(f"  CROSS-EVALUATION TOURNAMENT ({num_hands:,} hands × {n_seeds} seed(s))")
    print(f"  Seeds: {eval_seeds}")
    print(f"{'='*60}")

    # Accumulate per-seed results
    # Structure: raw[model_key][opp_name] = list of single-seed result dicts
    raw: dict[str, dict[str, list]] = {
        "model_a": {n: [] for n in OPP_NAMES},
        "model_b": {n: [] for n in OPP_NAMES},
        "random":  {n: [] for n in OPP_NAMES},
    }

    for s_idx, s in enumerate(eval_seeds):
        print(f"\n  --- Seed {s} ({s_idx+1}/{n_seeds}) ---")
        for opp_id, opp_name in enumerate(OPP_NAMES):
            print(f"  vs {opp_name}:")

            res_a = run_tournament(
                model_a, use_implicit=False, opponent_id=opp_id,
                num_hands=num_hands, seed=s,
            )
            raw["model_a"][opp_name].append(res_a)
            print(f"    Model A (Explicit):  {res_a['bb_per_100']:+8.2f} BB/100")

            res_b = run_tournament(
                model_b, use_implicit=True, opponent_id=opp_id,
                num_hands=num_hands, seed=s,
                collect_behavioral=(s_idx == 0),  # collect behavioral on first seed only
            )
            raw["model_b"][opp_name].append(res_b)
            print(f"    Model B (Implicit):  {res_b['bb_per_100']:+8.2f} BB/100")

            res_r = run_random_baseline(
                opponent_id=opp_id, num_hands=num_hands, seed=s,
            )
            raw["random"][opp_name].append(res_r)
            print(f"    Random Baseline:     {res_r['bb_per_100']:+8.2f} BB/100")

    # Aggregate: compute mean ± std across seeds
    results: dict[str, dict] = {"model_a": {}, "model_b": {}, "random": {}}

    for model_key in results:
        for opp_name in OPP_NAMES:
            seed_results = raw[model_key][opp_name]
            bbs = [r["bb_per_100"] for r in seed_results]
            # Use first-seed hand_profits for plots (representative)
            results[model_key][opp_name] = {
                "bb_per_100":      float(np.mean(bbs)),
                "bb_per_100_std":  float(np.std(bbs)) if n_seeds > 1 else 0.0,
                "bb_per_100_all":  bbs,
                "hand_profits":    seed_results[0]["hand_profits"],
            }
            # Preserve behavioral data from first seed for Model B
            if model_key == "model_b" and "post_draw_actions" in seed_results[0]:
                results[model_key][opp_name]["post_draw_actions"] = (
                    seed_results[0]["post_draw_actions"]
                )

    print(f"\n{'='*60}\n")
    return results


# ---------------------------------------------------------------------------
# Behavioral matrix analysis
# ---------------------------------------------------------------------------

def compute_behavioral_matrix(results: dict) -> dict:
    """Compute Model B's post-draw Fold/Call/Raise frequencies
    segmented by opponent draw count.

    Returns a nested dict:
        matrix[opp_name][draw_count] = {'fold': %, 'call': %, 'raise': %}
    """
    matrix = {}

    for opp_name in OPP_NAMES:
        opp_data = results["model_b"].get(opp_name, {})
        post_draw = opp_data.get("post_draw_actions", [])

        if not post_draw:
            matrix[opp_name] = {}
            continue

        draw_groups = defaultdict(lambda: {"fold": 0, "call": 0, "raise": 0, "total": 0})

        for opp_draw, agent_action, opp_id in post_draw:
            group = draw_groups[opp_draw]
            group["total"] += 1

            if agent_action == A_FOLD:
                group["fold"] += 1
            elif agent_action == A_RAISE:
                group["raise"] += 1
            else:
                group["call"] += 1

        # Convert to frequencies
        matrix[opp_name] = {}
        for draw_count in sorted(draw_groups.keys()):
            g = draw_groups[draw_count]
            n = max(1, g["total"])
            matrix[opp_name][draw_count] = {
                "fold":  g["fold"] / n,
                "call":  g["call"] / n,
                "raise": g["raise"] / n,
                "count": g["total"],
            }

    return matrix


def print_behavioral_matrix(matrix: dict):
    """Pretty-print the behavioral matrix."""
    print(f"\n{'='*70}")
    print(f"  THE TELL -- Model B Behavioral Matrix (Post-Draw Actions)")
    print(f"{'='*70}")

    for opp_name, data in matrix.items():
        print(f"\n  vs {opp_name}:")
        print(f"  {'Draw':>6}  {'Fold%':>8}  {'Call%':>8}  {'Raise%':>8}  {'N':>6}")
        print(f"  {'-'*42}")

        for draw_count in sorted(data.keys()):
            d = data[draw_count]
            print(
                f"  {draw_count:>6}  "
                f"{d['fold']*100:>7.1f}%  "
                f"{d['call']*100:>7.1f}%  "
                f"{d['raise']*100:>7.1f}%  "
                f"{d['count']:>6}"
            )

    print(f"\n{'='*70}\n")


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def print_summary_table(results: dict):
    """Print a clean summary table of BB/100 results, with ±std if available."""
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS -- BB/100 (Big Blinds per 100 Hands)")
    print(f"{'='*70}")
    print(f"\n  {'Opponent':<18} {'Model A':>14} {'Model B':>14} {'Random':>14}")
    print(f"  {'-'*62}")

    def _fmt(res_entry: dict) -> str:
        bb = res_entry["bb_per_100"]
        std = res_entry.get("bb_per_100_std", 0.0)
        n_seeds = len(res_entry.get("bb_per_100_all", [1]))
        if n_seeds > 1:
            return f"{bb:>+7.1f}±{std:.1f}"
        return f"{bb:>+9.2f}    "

    for opp_name in OPP_NAMES:
        fa = _fmt(results["model_a"][opp_name])
        fb = _fmt(results["model_b"][opp_name])
        fr = _fmt(results["random"][opp_name])
        print(f"  {opp_name:<18} {fa:>14} {fb:>14} {fr:>14}")

    # Averages
    avg_a = np.mean([results["model_a"][n]["bb_per_100"] for n in OPP_NAMES])
    avg_b = np.mean([results["model_b"][n]["bb_per_100"] for n in OPP_NAMES])
    avg_r = np.mean([results["random"][n]["bb_per_100"] for n in OPP_NAMES])
    print(f"  {'-'*62}")
    print(
        f"  {'AVERAGE':<18} {avg_a:>+9.2f}      {avg_b:>+9.2f}      {avg_r:>+9.2f}"
    )
    print(f"\n{'='*70}\n")
