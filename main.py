"""
main.py — Entry point for CardShark-RL.

Decoding "The Tell": Exploitative Reinforcement Learning
and Implicit Opponent Modeling in 5-Card Draw Poker.

Usage:
    python main.py                  # Full run (1M steps)
    python main.py --medium         # Medium run (500K steps)
    python main.py --quick          # Quick dev run (50K steps)
    python main.py --eval-only      # Skip training, evaluate saved models
    python main.py --hpo            # Run hyperparameter optimisation first
    python main.py --timesteps N    # Custom timestep count
"""

from __future__ import annotations
import argparse
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
        f.write(f"Device: {args.device}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Eval-only: {args.eval_only}\n")
        f.write(f"HPO: {args.hpo}\n\n")

        if results:
            f.write(f"Results (BB/100):\n")
            f.write(f"{'-'*50}\n")
            f.write(f"{'Opponent':<18} {'Model A':>10} {'Model B':>10} {'Random':>10}\n")
            f.write(f"{'-'*50}\n")
            from evaluate import OPP_NAMES
            import numpy as np
            for opp_name in OPP_NAMES:
                bb_a = results["model_a"][opp_name]["bb_per_100"]
                bb_b = results["model_b"][opp_name]["bb_per_100"]
                bb_r = results["random"][opp_name]["bb_per_100"]
                f.write(f"{opp_name:<18} {bb_a:>+9.2f}  {bb_b:>+9.2f}  {bb_r:>+9.2f}\n")
            avg_a = np.mean([results["model_a"][n]["bb_per_100"] for n in OPP_NAMES])
            avg_b = np.mean([results["model_b"][n]["bb_per_100"] for n in OPP_NAMES])
            avg_r = np.mean([results["random"][n]["bb_per_100"] for n in OPP_NAMES])
            f.write(f"{'-'*50}\n")
            f.write(f"{'AVERAGE':<18} {avg_a:>+9.2f}  {avg_b:>+9.2f}  {avg_r:>+9.2f}\n")

    print(f"  Saved: {info_path}")


def update_latest(run_dir: str, base: str = "results"):
    """Copy key files to results/latest/ for easy access."""
    latest_dir = os.path.join(base, "latest")
    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)
    print(f"  Copied to: {latest_dir}/")


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
        "--hpo-trials", type=int, default=20,
        help="Number of HPO trials (default: 20)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Custom training timesteps (overrides --quick)"
    )
    parser.add_argument(
        "--eval-hands", type=int, default=10_000,
        help="Number of hands per opponent in evaluation (default: 10000)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Training device: 'auto', 'cpu', or 'cuda'"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

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

    # Create timestamped results directory
    run_dir = make_run_dir("results")

    print_banner()

    # ------------------------------------------------------------------
    # Phase 0: HPO (optional)
    # ------------------------------------------------------------------
    hpo_params = None
    if args.hpo:
        from hpo import run_hpo
        print(f"\n  Running HPO with {args.hpo_trials} trials...")
        print(f"  (Each trial = 200K steps, ~6 min)\n")
        hpo_params = run_hpo(
            n_trials=args.hpo_trials,
            timesteps_per_trial=200_000,
            eval_hands=2_000,
            seed=args.seed,
            device=args.device,
            results_dir=run_dir,
        )
        print(f"\n  Best HPO params: {hpo_params}\n")
    else:
        best_params_path = "best_params.json"
        if os.path.exists(best_params_path):
            import json
            with open(best_params_path, "r") as f:
                saved_data = json.load(f)
                hpo_params = saved_data.get("params", saved_data)
            print(f"\n  Loaded optimised hyperparameters from '{best_params_path}'")
            print(f"  {hpo_params}\n")

    # ------------------------------------------------------------------
    # Phase 1: Training
    # ------------------------------------------------------------------
    if not args.eval_only:
        from train import train_both_models

        print(f"\n  Training with {total_timesteps:,} timesteps per model...")
        print(f"  Device: {args.device}")
        print(f"  Seed: {args.seed}\n")

        train_kwargs = {}
        if hpo_params:
            train_kwargs = hpo_params

        train_results = train_both_models(
            total_timesteps=total_timesteps,
            save_dir=save_dir,
            log_dir=log_dir,
            seed=args.seed,
            device=args.device,
            **train_kwargs,
        )

        model_a = train_results["model_a"]
        model_b = train_results["model_b"]
        callback_a = train_results["callback_a"]
        callback_b = train_results["callback_b"]

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

    print(f"\n  Running evaluation ({args.eval_hands:,} hands per matchup)...\n")

    eval_results = full_evaluation(
        model_a=model_a,
        model_b=model_b,
        num_hands=args.eval_hands,
        seed=args.seed,
        results_dir=run_dir,
    )

    # ------------------------------------------------------------------
    # Phase 3: Analysis & Plots
    # ------------------------------------------------------------------
    print_summary_table(eval_results)

    # Behavioral matrix
    matrix = compute_behavioral_matrix(eval_results)
    print_behavioral_matrix(matrix)

    # Generate plots — all saved to timestamped run directory
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

    # Save run metadata
    save_run_info(run_dir, args, total_timesteps, eval_results)

    # Copy to latest
    update_latest(run_dir, "results")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  ALL DONE — Results saved to '{run_dir}/'")
    print(f"{'='*60}")
    print(f"\n  Generated files:")
    for f in sorted(os.listdir(run_dir)):
        fpath = os.path.join(run_dir, f)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"    - {f}  ({size_kb:.0f} KB)")
    print()


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
