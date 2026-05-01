"""
main.py — Entry point for CardShark-RL.

Decoding "The Tell": Exploitative Reinforcement Learning
and Implicit Opponent Modeling in 5-Card Draw Poker.

Usage:
    python main.py                       # Full run (1M steps, 3-seed eval)
    python main.py --medium              # Medium run (500K steps)
    python main.py --quick               # Quick dev run (50K steps)
    python main.py --eval-only           # Skip training, evaluate saved models
    python main.py --hpo                 # HPO for Model B (default), then train
    python main.py --hpo --hpo-model a   # HPO for Model A only
    python main.py --hpo --hpo-model both# Separate HPO for A and B
    python main.py --timesteps N         # Custom timestep count
    python main.py --n-envs 4            # Train with 4 parallel envs
    python main.py --eval-seeds 0 1 2    # Multi-seed evaluation
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def make_run_dir(base: str = "results") -> str:
    """Create a timestamped run directory and return its path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_run_info(run_dir: str, args, total_timesteps: int, results: dict | None = None):
    """Save run metadata to a text file inside the run directory."""
    info_path = os.path.join(run_dir, "run_info.txt")
    with open(info_path, "w") as f:
        f.write(f"CardShark-RL Run Report\n")
        f.write(f"{'='*60}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Timesteps per model: {total_timesteps:,}\n")
        f.write(f"Eval hands: {args.eval_hands:,}\n")
        f.write(f"Eval seeds: {args.eval_seeds}\n")
        f.write(f"n_envs: {args.n_envs}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Eval-only: {args.eval_only}\n")
        f.write(f"HPO: {args.hpo} (model={args.hpo_model if args.hpo else 'n/a'})\n\n")

        if results:
            f.write(f"Results (BB/100):\n")
            f.write(f"{'-'*62}\n")
            f.write(f"{'Opponent':<18} {'Model A':>14} {'Model B':>14} {'Random':>14}\n")
            f.write(f"{'-'*62}\n")
            from evaluate import OPP_NAMES
            import numpy as np
            for opp_name in OPP_NAMES:
                def _fmt(entry):
                    bb = entry["bb_per_100"]
                    std = entry.get("bb_per_100_std", 0.0)
                    n = len(entry.get("bb_per_100_all", [1]))
                    return f"{bb:>+7.1f}±{std:.1f}" if n > 1 else f"{bb:>+9.2f}    "
                fa = _fmt(results["model_a"][opp_name])
                fb = _fmt(results["model_b"][opp_name])
                fr = _fmt(results["random"][opp_name])
                f.write(f"{opp_name:<18} {fa:>14} {fb:>14} {fr:>14}\n")
            avg_a = np.mean([results["model_a"][n]["bb_per_100"] for n in OPP_NAMES])
            avg_b = np.mean([results["model_b"][n]["bb_per_100"] for n in OPP_NAMES])
            avg_r = np.mean([results["random"][n]["bb_per_100"] for n in OPP_NAMES])
            f.write(f"{'-'*62}\n")
            f.write(f"{'AVERAGE':<18} {avg_a:>+9.2f}      {avg_b:>+9.2f}      {avg_r:>+9.2f}\n")

    print(f"  Saved: {info_path}")


def update_latest(run_dir: str, base: str = "results"):
    """Copy key files to results/latest/ for easy access."""
    latest_dir = os.path.join(base, "latest")
    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)
    print(f"  Copied to: {latest_dir}/")


def load_best_params() -> dict:
    """Load saved HPO params from best_params.json.

    Returns a dict that may have keys 'model_a' and/or 'model_b',
    each pointing to a params sub-dict, OR a flat params dict (legacy format).
    """
    path = "best_params.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)

    # New format: {"model_a": {"params": {...}}, "model_b": {"params": {...}}}
    if "model_a" in data or "model_b" in data:
        result = {}
        if "model_a" in data:
            result["model_a"] = data["model_a"].get("params", data["model_a"])
        if "model_b" in data:
            result["model_b"] = data["model_b"].get("params", data["model_b"])
        return result

    # Legacy flat format: {"params": {...}} or {"best_value": ..., "params": {...}}
    params = data.get("params", data)
    # Remove non-train_model keys
    params.pop("best_value", None)
    return {"model_b": params}  # Legacy assumed to be Model B


def main():
    parser = argparse.ArgumentParser(
        description="CardShark-RL: Exploitative RL in 5-Card Draw Poker"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick training run (50K steps instead of 1M)"
    )
    parser.add_argument(
        "--medium", action="store_true",
        help="Medium training run (500K steps instead of 1M)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, load saved models and evaluate"
    )
    parser.add_argument(
        "--hpo", action="store_true",
        help="Run Optuna hyperparameter optimisation before training"
    )
    parser.add_argument(
        "--hpo-trials", type=int, default=40,
        help="Number of HPO trials (default: 40)"
    )
    parser.add_argument(
        "--hpo-model", type=str, default="b", choices=["a", "b", "both"],
        help="Which model to run HPO for: 'a', 'b', or 'both' (default: 'b')"
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Custom training timesteps (overrides --quick/--medium)"
    )
    parser.add_argument(
        "--eval-hands", type=int, default=10_000,
        help="Number of hands per opponent per seed in evaluation (default: 10000)"
    )
    parser.add_argument(
        "--eval-seeds", type=int, nargs="+", default=None,
        help="RNG seeds for multi-seed evaluation (default: single seed = --seed). "
             "Example: --eval-seeds 0 1 2"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Training device: 'auto', 'cpu', or 'cuda'"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--n-envs", type=int, default=4,
        help="Number of parallel training environments (default: 4)"
    )
    parser.add_argument(
        "--no-eval-callback", action="store_true",
        help="Disable per-opponent evaluation callbacks during training"
    )
    args = parser.parse_args()

    # Default eval seeds: 3-seed evaluation unless overridden
    if args.eval_seeds is None:
        args.eval_seeds = [args.seed, args.seed + 100, args.seed + 200]

    # Determine timesteps
    if args.timesteps:
        total_timesteps = args.timesteps
    elif args.quick:
        total_timesteps = 50_000
    elif args.medium:
        total_timesteps = 500_000
    else:
        total_timesteps = 1_000_000

    save_dir = "models"
    log_dir = "logs"
    run_dir = make_run_dir("results")

    print_banner()

    # ------------------------------------------------------------------
    # Phase 0: HPO (optional)
    # ------------------------------------------------------------------
    # hpo_params: either a flat dict (single model) or {"model_a": ..., "model_b": ...}
    hpo_params: dict = {}
    if args.hpo:
        from hpo import run_hpo
        print(f"\n  Running HPO ({args.hpo_model.upper()}) — {args.hpo_trials} trials @ 300K steps each...")
        hpo_params = run_hpo(
            n_trials=args.hpo_trials,
            timesteps_per_trial=300_000,
            eval_hands=2_000,
            seed=args.seed,
            device=args.device,
            results_dir=run_dir,
            hpo_model=args.hpo_model,
        )
        print(f"\n  HPO complete. Best params: {hpo_params}\n")
    else:
        loaded = load_best_params()
        if loaded:
            hpo_params = loaded
            print(f"\n  Loaded saved hyperparameters from 'best_params.json'")
            for k, v in hpo_params.items():
                print(f"    {k}: {v}")
            print()

    # Resolve per-model param dicts
    # hpo_params may be:
    #   {"model_a": {...}, "model_b": {...}}  — separate HPO
    #   {"model_b": {...}}                    — legacy / single-model HPO
    #   {}                                    — no HPO, use defaults
    if "model_a" in hpo_params or "model_b" in hpo_params:
        model_a_params = hpo_params.get("model_a", {})
        model_b_params = hpo_params.get("model_b", {})
        shared_params = {}
    else:
        # Flat dict — treat as Model B params, Model A uses defaults
        model_a_params = {}
        model_b_params = hpo_params
        shared_params = {}

    # ------------------------------------------------------------------
    # Phase 1: Training
    # ------------------------------------------------------------------
    if not args.eval_only:
        from train import train_both_models

        print(f"\n  Training with {total_timesteps:,} timesteps per model...")
        print(f"  n_envs: {args.n_envs} | Device: {args.device} | Seed: {args.seed}\n")

        train_results = train_both_models(
            total_timesteps=total_timesteps,
            save_dir=save_dir,
            log_dir=log_dir,
            seed=args.seed,
            device=args.device,
            n_envs=args.n_envs,
            model_a_params=model_a_params,
            model_b_params=model_b_params,
            shared_params=shared_params,
            eval_during_training=not args.no_eval_callback,
        )

        model_a = train_results["model_a"]
        model_b = train_results["model_b"]
        callback_a = train_results["callback_a"]
        callback_b = train_results["callback_b"]
        eval_callback_a = train_results.get("eval_callback_a")
        eval_callback_b = train_results.get("eval_callback_b")

    else:
        print("\n  Loading saved models...\n")
        from sb3_contrib import MaskablePPO

        model_a_path = os.path.join(save_dir, "model_a_explicit.zip")
        model_b_path = os.path.join(save_dir, "model_b_implicit.zip")

        if not os.path.exists(model_a_path) or not os.path.exists(model_b_path):
            print(f"  ERROR: Saved models not found in '{save_dir}/'")
            print(f"  Run training first: python main.py")
            sys.exit(1)

        model_a = MaskablePPO.load(model_a_path)
        model_b = MaskablePPO.load(model_b_path)
        callback_a = None
        callback_b = None
        eval_callback_a = None
        eval_callback_b = None
        print(f"  Loaded Model A from {model_a_path}")
        print(f"  Loaded Model B from {model_b_path}")

    # ------------------------------------------------------------------
    # Phase 2: Evaluation
    # ------------------------------------------------------------------
    from evaluate import (
        full_evaluation, compute_behavioral_matrix,
        print_behavioral_matrix, print_summary_table,
        plot_bb_comparison, plot_learning_curves,
        plot_behavioral_heatmap, plot_cumulative_profit,
    )

    n_seeds = len(args.eval_seeds)
    print(f"\n  Running evaluation ({args.eval_hands:,} hands × {n_seeds} seeds)...\n")

    eval_results = full_evaluation(
        model_a=model_a,
        model_b=model_b,
        num_hands=args.eval_hands,
        seed=args.seed,
        results_dir=run_dir,
        eval_seeds=args.eval_seeds,
    )

    # ------------------------------------------------------------------
    # Phase 3: Analysis & Plots
    # ------------------------------------------------------------------
    print_summary_table(eval_results)

    matrix = compute_behavioral_matrix(eval_results)
    print_behavioral_matrix(matrix)

    print(f"\n  Generating visualisations...\n")

    plot_bb_comparison(
        eval_results,
        save_path=os.path.join(run_dir, "bb_per_100_comparison.png"),
    )

    if callback_a and callback_b:
        plot_learning_curves(
            callback_a, callback_b,
            save_path=os.path.join(run_dir, "learning_curves.png"),
        )

    plot_behavioral_heatmap(
        matrix,
        save_path=os.path.join(run_dir, "behavioral_matrix.png"),
    )

    plot_cumulative_profit(
        eval_results,
        save_path=os.path.join(run_dir, "cumulative_profit.png"),
    )

    # Save per-opponent training eval history if available
    if eval_callback_a or eval_callback_b:
        _save_training_eval_history(
            eval_callback_a, eval_callback_b, run_dir
        )

    save_run_info(run_dir, args, total_timesteps, eval_results)
    update_latest(run_dir, "results")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  ALL DONE — Results saved to '{run_dir}/'")
    print(f"{'='*60}")
    print(f"\n  Generated files:")
    for fname in sorted(os.listdir(run_dir)):
        fpath = os.path.join(run_dir, fname)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"    - {fname}  ({size_kb:.0f} KB)")
    print()


def _save_training_eval_history(eval_cb_a, eval_cb_b, run_dir: str):
    """Persist the per-opponent eval callback history to JSON for later plotting."""
    history = {}
    if eval_cb_a:
        history["model_a"] = {
            opp: [(ts, bb) for ts, bb in pts]
            for opp, pts in eval_cb_a.eval_history.items()
        }
    if eval_cb_b:
        history["model_b"] = {
            opp: [(ts, bb) for ts, bb in pts]
            for opp, pts in eval_cb_b.eval_history.items()
        }
    path = os.path.join(run_dir, "training_eval_history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Saved: {path}")


def print_banner():
    banner = """
    +===============================================================+
    |                                                               |
    |   CARDSHARK-RL                                                |
    |                                                               |
    |   Decoding "The Tell"                                         |
    |   Exploitative RL & Implicit Opponent Modeling                |
    |   in 5-Card Draw Poker                                        |
    |                                                               |
    +===============================================================+
    """
    print(banner)


if __name__ == "__main__":
    main()
