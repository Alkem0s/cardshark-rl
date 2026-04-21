"""
main.py — Entry point for CardShark-RL.

Decoding "The Tell": Exploitative Reinforcement Learning
and Implicit Opponent Modeling in 5-Card Draw Poker.

Usage:
    python main.py                  # Full run (500K steps)
    python main.py --quick          # Quick dev run (50K steps)
    python main.py --eval-only      # Skip training, evaluate saved models
    python main.py --timesteps N    # Custom timestep count
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    parser = argparse.ArgumentParser(
        description="CardShark-RL: Exploitative RL in 5-Card Draw Poker"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick training run (50K steps instead of 500K)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, load saved models and evaluate"
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
    else:
        total_timesteps = 500_000

    save_dir = "models"
    log_dir = "logs"
    results_dir = "results"

    print_banner()

    # ------------------------------------------------------------------
    # Phase 1: Training
    # ------------------------------------------------------------------
    if not args.eval_only:
        from train import train_both_models

        print(f"\n  Training with {total_timesteps:,} timesteps per model...")
        print(f"  Device: {args.device}")
        print(f"  Seed: {args.seed}\n")

        train_results = train_both_models(
            total_timesteps=total_timesteps,
            save_dir=save_dir,
            log_dir=log_dir,
            seed=args.seed,
            device=args.device,
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
        results_dir=results_dir,
    )

    # ------------------------------------------------------------------
    # Phase 3: Analysis & Plots
    # ------------------------------------------------------------------
    print_summary_table(eval_results)

    # Behavioral matrix
    matrix = compute_behavioral_matrix(eval_results)
    print_behavioral_matrix(matrix)

    # Generate plots
    print(f"\n  Generating visualisations...\n")

    plot_bb_comparison(
        eval_results,
        save_path=os.path.join(results_dir, "bb_per_100_comparison.png"),
    )

    if callback_a and callback_b:
        plot_learning_curves(
            callback_a, callback_b,
            save_path=os.path.join(results_dir, "learning_curves.png"),
        )

    plot_behavioral_heatmap(
        matrix,
        save_path=os.path.join(results_dir, "behavioral_matrix.png"),
    )

    plot_cumulative_profit(
        eval_results,
        save_path=os.path.join(results_dir, "cumulative_profit.png"),
    )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  ALL DONE -- Results saved to '{results_dir}/'")
    print(f"{'='*60}")
    print(f"\n  Generated files:")
    for f in os.listdir(results_dir):
        fpath = os.path.join(results_dir, f)
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
