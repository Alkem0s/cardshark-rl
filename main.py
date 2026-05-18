"""
main.py — Entry point for CardShark-RL.

Decoding "The Tell": Exploitative Reinforcement Learning
and Implicit Opponent Modeling in 5-Card Draw Poker.

Usage:
    python main.py                        # Full run (1M steps, 3-seed eval)
    python main.py --medium               # Medium run (500K steps)
    python main.py --quick                # Quick dev run (50K steps)
    python main.py --eval-only            # Skip training, evaluate saved models
    python main.py --hpo                  # HPO for Model B (default), then train
    python main.py --hpo --hpo-model a    # HPO for Model A only
    python main.py --hpo --hpo-model both # Separate HPO for A and B
    python main.py --timesteps N          # Custom timestep count
    python main.py --n-envs 4             # Train with 4 parallel envs
    python main.py --eval-seeds 0 1 2     # Multi-seed evaluation
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


def save_run_info(
    run_dir: str,
    args,
    total_timesteps: int,
    results: dict | None = None,
    sig_results: dict | None = None,
    non_stationary_results: tuple | None = None,
):
    """Save run metadata to a text file inside the run directory."""
    info_path = os.path.join(run_dir, "run_info.txt")
    with open(info_path, "w") as f:
        f.write(f"CardShark-RL Run Report\n")
        f.write(f"{'='*60}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Timesteps per model: {total_timesteps:,}\n")
        f.write(f"Training seeds: {args.train_seeds}\n")
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

        if sig_results:
            n_total = len(args.train_seeds) * len(args.eval_seeds) * args.eval_hands * len(OPP_NAMES)
            f.write(f"\nStatistical Validation (Pooled over {n_total:,} hands):\n")
            f.write(f"{'='*60}\n")
            f.write(f"Model B vs Model A (Paired t-test):  t = {sig_results['ba_t_stat']:+.3f}, p = {sig_results['ba_t_p']:.3e}\n")
            f.write(f"Model B vs Random (Paired t-test):   t = {sig_results['br_t_stat']:+.3f}, p = {sig_results['br_t_p']:.3e}\n")
            if sig_results.get("scipy_available"):
                f.write(f"Model B vs Model A (Wilcoxon):      W = {sig_results['ba_w_stat']:.1f}, p = {sig_results['ba_w_p']:.3e}\n")
                f.write(f"Model B vs Random (Wilcoxon):       W = {sig_results['br_w_stat']:.1f}, p = {sig_results['br_w_p']:.3e}\n")
            f.write(f"\nModel B 95% Confidence Intervals:\n")
            f.write(f"  Average profit/hand: {sig_results['b_mean_profit']:+.4f} chips  [95% CI: {sig_results['b_profit_ci_lower']:.4f}, {sig_results['b_profit_ci_upper']:.4f}]\n")
            f.write(f"  Average BB/100 hands: {sig_results['b_bb_mean']:+.2f} BB      [95% CI: {sig_results['b_bb_ci_lower']:.2f}, {sig_results['b_bb_ci_upper']:.2f}]\n")

        if non_stationary_results:
            res_implicit, res_oracle, res_blind = non_stationary_results
            f.write(f"\nNon-Stationary Adaptation Tournament (1,500 hands):\n")
            f.write(f"{'='*60}\n")
            f.write(f"Model B (Implicit):         {np.mean(res_implicit['hand_rewards'])*100:+.2f} BB/100\n")
            f.write(f"Model A (Explicit Oracle):  {np.mean(res_oracle['hand_rewards'])*100:+.2f} BB/100\n")
            f.write(f"Model A (Explicit Blind):   {np.mean(res_blind['hand_rewards'])*100:+.2f} BB/100\n")

    print(f"  Saved: {info_path}")



def update_latest(run_dir: str, base: str = "results"):
    """Copy key files to results/latest/ for easy access."""
    latest_dir = os.path.join(base, "latest")
    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)
    print(f"  Copied to: {latest_dir}/")


def load_best_params() -> dict:
    """Load saved HPO params from best_params_a.json and best_params_b.json.

    Returns a dict that may have keys 'model_a' and/or 'model_b',
    each pointing to a params sub-dict.
    """
    result = {}

    # 1. Load Model A
    path_a = "best_params_a.json"
    if os.path.exists(path_a):
        with open(path_a) as f:
            data = json.load(f)
            result["model_a"] = data.get("params", data)

    # 2. Load Model B
    path_b = "best_params_b.json"
    if os.path.exists(path_b):
        with open(path_b) as f:
            data = json.load(f)
            result["model_b"] = data.get("params", data)

    # 3. Legacy fallback (best_params.json)
    if not result:
        path_legacy = "best_params.json"
        if os.path.exists(path_legacy):
            with open(path_legacy) as f:
                data = json.load(f)

            if "model_a" in data or "model_b" in data:
                if "model_a" in data:
                    result["model_a"] = data["model_a"].get("params", data["model_a"])
                if "model_b" in data:
                    result["model_b"] = data["model_b"].get("params", data["model_b"])
            else:
                params = data.get("params", data)
                if isinstance(params, dict):
                    params.pop("best_value", None)
                result["model_b"] = params

    return result


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
        "--hpo-jobs", type=int, default=1,
        help="Number of parallel Optuna trials (default: 1)"
    )
    parser.add_argument(
        "--train-model", type=str, default="both", choices=["a", "b", "both"],
        help="Which model to train: 'a', 'b', or 'both' (default: 'both')"
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
        "--train-seeds", type=int, nargs="+", default=None,
        help="RNG seeds for multi-seed training (default: 3 seeds [42, 142, 242] for medium/standard, 1 seed for quick)"
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
        "--n-envs", type=int, default=1,
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

    # Default train seeds: 3 seeds for scientific rigor, 1 seed for quick runs to save time
    if args.train_seeds is None:
        if args.quick:
            args.train_seeds = [args.seed]
        else:
            args.train_seeds = [args.seed, args.seed + 100, args.seed + 200]

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
            n_envs=args.n_envs,
            n_jobs=args.hpo_jobs,
            results_dir=run_dir,
            hpo_model=args.hpo_model,
        )
        print(f"\n  HPO complete. Best params: {hpo_params}\n")
    else:
        loaded = load_best_params()
        if loaded:
            hpo_params = loaded
            print(f"\n  Loaded saved hyperparameters from best_params_[a|b].json")
            for k, v in hpo_params.items():
                print(f"    {k}: {v}")
            print()

    # Resolve per-model param dicts
    # hpo_params is usually {"model_a": {...}, "model_b": {...}}
    if "model_a" in hpo_params or "model_b" in hpo_params:
        model_a_params = hpo_params.get("model_a", {})
        model_b_params = hpo_params.get("model_b", {})
    else:
        # Fallback for flat dict (legacy/manual configs)
        if args.hpo and args.hpo_model == "a":
            model_a_params = hpo_params
            model_b_params = {}
        else:
            # Default fallback (usually Model B)
            model_a_params = {}
            model_b_params = hpo_params

    shared_params = {}

    # ------------------------------------------------------------------
    # Phase 1: Training
    # ------------------------------------------------------------------
    all_callbacks_a = []
    all_callbacks_b = []
    all_eval_callbacks_a = []
    all_eval_callbacks_b = []

    if not args.eval_only:
        from train import train_model
        from sb3_contrib import MaskablePPO

        print(f"\n  Training requested for: {args.train_model.upper()}")
        print(f"  n_envs: {args.n_envs} | Device: {args.device} | Seeds: {args.train_seeds}\n")

        for s in args.train_seeds:
            print(f"\n  ==================================================")
            print(f"  >>> TRAINING SEED {s} <<<")
            print(f"  ==================================================")

            # --- Train Model A ---
            if args.train_model in ["both", "a"]:
                print(f"  Training Model A (Explicit) Seed {s}...")
                model_a, callback_a, eval_callback_a = train_model(
                    model_name=f"model_a_explicit_seed{s}",
                    use_implicit=False,
                    total_timesteps=total_timesteps,
                    save_dir=save_dir,
                    log_dir=log_dir,
                    seed=s,
                    device=args.device,
                    n_envs=args.n_envs,
                    eval_during_training=not args.no_eval_callback,
                    **{**shared_params, **model_a_params},
                )
                all_callbacks_a.append(callback_a)
                all_eval_callbacks_a.append(eval_callback_a)

            # --- Train Model B ---
            if args.train_model in ["both", "b"]:
                print(f"  Training Model B (Implicit) Seed {s}...")
                model_b, callback_b, eval_callback_b = train_model(
                    model_name=f"model_b_implicit_seed{s}",
                    use_implicit=True,
                    total_timesteps=total_timesteps,
                    save_dir=save_dir,
                    log_dir=log_dir,
                    seed=s + 1,
                    device=args.device,
                    n_envs=args.n_envs,
                    eval_during_training=not args.no_eval_callback,
                    **{**shared_params, **model_b_params},
                )
                all_callbacks_b.append(callback_b)
                all_eval_callbacks_b.append(eval_callback_b)
    else:
        print("\n  Skip training requested (--eval-only). We will load models for each seed.\n")

    # ------------------------------------------------------------------
    # Phase 2: Evaluation
    # ------------------------------------------------------------------
    from evaluate import (
        full_evaluation, compute_behavioral_matrix,
        print_behavioral_matrix, print_summary_table,
        run_non_stationary_tournament, compute_statistical_significance,
    )
    from visualizer import (
        set_style, plot_architecture,
        plot_bb_comparison, plot_learning_curves,
        plot_behavioral_heatmap, plot_cumulative_profit,
        plot_non_stationary_adaptation,
    )
    import numpy as np
    from sb3_contrib import MaskablePPO

    set_style()
    n_train_seeds = len(args.train_seeds)
    n_eval_seeds = len(args.eval_seeds)
    print(f"\n  Running evaluation for {n_train_seeds} training seeds × {args.eval_hands:,} hands × {n_eval_seeds} eval seeds...\n")

    all_eval_results = []
    
    first_model_a = None
    first_model_b = None

    for idx, s in enumerate(args.train_seeds):
        print(f"\n  --- Evaluating Seed {s} ---")
        
        model_a_path = os.path.join(save_dir, f"model_a_explicit_seed{s}.zip")
        model_b_path = os.path.join(save_dir, f"model_b_implicit_seed{s}.zip")
        
        # Fallback to standard names if seed-specific zip doesn't exist
        if not os.path.exists(model_a_path):
            model_a_path = os.path.join(save_dir, "model_a_explicit.zip")
        if not os.path.exists(model_b_path):
            model_b_path = os.path.join(save_dir, "model_b_implicit.zip")
            
        if not os.path.exists(model_a_path) or not os.path.exists(model_b_path):
            print(f"  ERROR: Models not found for seed {s} or default names. Run training first.")
            sys.exit(1)
            
        model_a_seed = MaskablePPO.load(model_a_path)
        model_b_seed = MaskablePPO.load(model_b_path)
        
        if idx == 0:
            first_model_a = model_a_seed
            first_model_b = model_b_seed

        res = full_evaluation(
            model_a=model_a_seed,
            model_b=model_b_seed,
            num_hands=args.eval_hands,
            seed=args.seed,
            results_dir=run_dir,
            eval_seeds=args.eval_seeds,
        )
        all_eval_results.append(res)

    # Aggregate results across training seeds
    print(f"\n  Aggregating evaluation results across {n_train_seeds} training seeds...")
    
    def aggregate_results(all_results: list[dict]) -> dict:
        aggregated = {
            "model_a": {},
            "model_b": {},
            "random": {},
        }
        from evaluate import OPP_NAMES
        for model_key in ["model_a", "model_b", "random"]:
            for opp in OPP_NAMES:
                pooled_bb = []
                pooled_profits = []
                for res in all_results:
                    pooled_bb.extend(res[model_key][opp]["bb_per_100_all"])
                    pooled_profits.extend(res[model_key][opp]["hand_profits"])
                aggregated[model_key][opp] = {
                    "bb_per_100": float(np.mean(pooled_bb)),
                    "bb_per_100_std": float(np.std(pooled_bb)) if len(pooled_bb) > 1 else 0.0,
                    "bb_per_100_all": pooled_bb,
                    "hand_profits": pooled_profits,
                }
                if model_key == "model_b":
                    aggregated[model_key][opp]["post_draw_actions"] = all_results[0]["model_b"][opp].get("post_draw_actions", [])
        return aggregated

    eval_results = aggregate_results(all_eval_results)

    # ------------------------------------------------------------------
    # Non-Stationary Opponent Shift Adaptation Experiment
    # ------------------------------------------------------------------
    print(f"\n  Running Non-Stationary Opponent Shift Adaptation Experiment...")
    # 1,500 hands: 500 CallingStation (0), 500 Maniac (1), 500 Rock (2)
    opp_seq = [0] * 500 + [1] * 500 + [2] * 500
    
    print("    Model B (Implicit) evaluating...")
    res_implicit = run_non_stationary_tournament(
        first_model_b, use_implicit=True, opponent_sequence=opp_seq, seed=args.seed
    )
    
    print("    Model A (Explicit Oracle) evaluating...")
    res_oracle = run_non_stationary_tournament(
        first_model_a, use_implicit=False, opponent_sequence=opp_seq, seed=args.seed
    )
    
    print("    Model A (Explicit Blind) evaluating...")
    res_blind = run_non_stationary_tournament(
        first_model_a, use_implicit=False, opponent_sequence=opp_seq, seed=args.seed, blind_opponent_id=0
    )
    
    non_stationary_results = (res_implicit, res_oracle, res_blind)
    print("    Adaptation experiment complete!")

    # ------------------------------------------------------------------
    # Statistical Significance Testing
    # ------------------------------------------------------------------
    print(f"\n  Computing statistical significance...")
    from evaluate import OPP_NAMES
    profits_b = []
    profits_a = []
    profits_r = []
    for opp in OPP_NAMES:
        profits_b.extend(eval_results["model_b"][opp]["hand_profits"])
        profits_a.extend(eval_results["model_a"][opp]["hand_profits"])
        profits_r.extend(eval_results["random"][opp]["hand_profits"])
        
    sig_results = compute_statistical_significance(profits_b, profits_a, profits_r)
    print("    Statistical tests completed successfully!")

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

    if not args.eval_only:
        # Plot training curves with shaded confidence regions
        plot_learning_curves(
            all_callbacks_a, all_callbacks_b,
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

    # Dynamic network architecture plot based on Model B's actual MLP layout
    net_arch_b = model_b_params.get("net_arch", [256, 256, 256, 256])
    plot_architecture(
        net_arch=net_arch_b,
        save_path=os.path.join(run_dir, "architecture.png"),
    )

    # Non-stationary adaptation plots
    plot_non_stationary_adaptation(
        res_implicit,
        save_path=os.path.join(run_dir, "non_stationary_adaptation.png")
    )

    # Save per-opponent training eval history if available
    if len(all_eval_callbacks_a) > 0 or len(all_eval_callbacks_b) > 0:
        eval_callback_a = all_eval_callbacks_a[0] if all_eval_callbacks_a else None
        eval_callback_b = all_eval_callbacks_b[0] if all_eval_callbacks_b else None
        _save_training_eval_history(
            eval_callback_a, eval_callback_b, run_dir
        )

    save_run_info(run_dir, args, total_timesteps, eval_results, sig_results, non_stationary_results)
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
