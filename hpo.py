"""
hpo.py — Optuna-based hyperparameter optimisation for CardShark-RL.

Searches over PPO hyperparameters using a shortened training budget
(200K steps) and evaluates average BB/100 across all archetypes.

Trains Model B (implicit) with block scheduling, since that's the model
that needs the most optimisation.

Usage:
    python main.py --hpo                 # 20 trials (default)
    python main.py --hpo --hpo-trials 40 # 40 trials
"""

from __future__ import annotations
import os
import json
import logging
import sys
import warnings
from datetime import datetime
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import optuna
    from optuna.exceptions import TrialPruned
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from gym_wrapper import DrawPokerGymEnv, mask_fn
from evaluate import run_tournament, OPP_NAMES


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_hpo_logger(results_dir: str) -> logging.Logger:
    """Create a logger that writes to both console and a log file."""
    logger = logging.getLogger("cardshark_hpo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "hpo_log.txt")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("  %(message)s"))
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Pruning callback — abort bad trials early
# ---------------------------------------------------------------------------

class TrialEvalCallback(BaseCallback):
    """Periodically evaluate the model during training and report to Optuna."""

    def __init__(
        self,
        trial: "optuna.Trial",
        use_implicit: bool = True,
        eval_interval: int = 50_000,
        eval_hands: int = 1_000,
        seed: int = 0,
        logger: logging.Logger | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.trial = trial
        self.use_implicit = use_implicit
        self.eval_interval = eval_interval
        self.eval_hands = eval_hands
        self.seed = seed
        self.hpo_logger = logger
        self._last_eval_step = 0
        self.best_score = -999.0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step >= self.eval_interval:
            self._last_eval_step = self.num_timesteps

            # Quick evaluation against all archetypes
            avg_bb = self._evaluate()

            if self.hpo_logger:
                self.hpo_logger.info(
                    f"  Trial {self.trial.number} @ {self.num_timesteps:,} steps: "
                    f"avg BB/100 = {avg_bb:+.2f}"
                )

            self.trial.report(avg_bb, step=self.num_timesteps)
            if self.trial.should_prune():
                raise TrialPruned()

            self.best_score = max(self.best_score, avg_bb)

        return True

    def _evaluate(self) -> float:
        """Evaluate current model against all archetypes, return min BB/100.

        Uses the MINIMUM across opponents so the objective directly
        penalises weakness against any single archetype (e.g. the Rock).
        """
        model = self.model  # Set by SB3 during learn()
        bb_scores = []

        for opp_id in range(len(OPP_NAMES)):
            result = run_tournament(
                model,
                use_implicit=self.use_implicit,
                opponent_id=opp_id,
                num_hands=self.eval_hands,
                seed=self.seed + opp_id,
            )
            bb_scores.append(result["bb_per_100"])

        return min(bb_scores)


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def create_objective(
    timesteps: int = 200_000,
    eval_hands: int = 2_000,
    seed: int = 42,
    device: str = "auto",
    logger: logging.Logger | None = None,
):
    """Create an Optuna objective closure.

    Trains the IMPLICIT model (Model B) with block scheduling,
    since that's the model we most want to optimise.
    """

    def objective(trial: "optuna.Trial") -> float:
        """Train a model with sampled hyperparameters and return avg BB/100."""

        # Sample hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        n_epochs = trial.suggest_int("n_epochs", 5, 15)
        gamma = trial.suggest_float("gamma", 0.95, 1.0)
        ent_coef = trial.suggest_float("ent_coef", 0.001, 0.05, log=True)
        rolling_window = trial.suggest_categorical("rolling_window", [50, 100, 150])
        block_size = trial.suggest_categorical("block_size", [100, 200, 400])

        # Network architecture
        n_layers = trial.suggest_int("n_layers", 2, 3)
        layer_size = trial.suggest_categorical("layer_size", [128, 256, 512])
        net_arch = [layer_size] * n_layers

        # Ensure batch_size <= n_steps
        if batch_size > n_steps:
            batch_size = n_steps

        # Log trial parameters
        if logger:
            logger.info(f"{'='*60}")
            logger.info(f"Trial {trial.number} starting")
            logger.info(f"  learning_rate: {learning_rate:.6f}")
            logger.info(f"  n_steps: {n_steps}")
            logger.info(f"  batch_size: {batch_size}")
            logger.info(f"  n_epochs: {n_epochs}")
            logger.info(f"  gamma: {gamma:.4f}")
            logger.info(f"  ent_coef: {ent_coef:.6f}")
            logger.info(f"  net_arch: {net_arch}")
            logger.info(f"  rolling_window: {rolling_window}")
            logger.info(f"  block_size: {block_size}")

        # Create environment — IMPLICIT model with block scheduling
        env = DrawPokerGymEnv(
            use_implicit_modeling=True,
            opponent_id=None,
            rolling_window=rolling_window,
            rng_seed=seed,
            opponent_schedule="block",
            block_size=block_size,
        )
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)

        # Create model
        policy_kwargs = dict(net_arch=net_arch)
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            ent_coef=ent_coef,
            verbose=0,
            seed=seed,
            device=device,
            policy_kwargs=policy_kwargs,
        )

        # Pruning callback
        eval_callback = TrialEvalCallback(
            trial=trial,
            use_implicit=True,
            eval_interval=50_000,
            eval_hands=eval_hands // 2,  # Smaller eval during training
            seed=seed,
            logger=logger,
        )

        try:
            model.learn(
                total_timesteps=timesteps,
                callback=eval_callback,
                progress_bar=False,
            )
        except TrialPruned:
            if logger:
                logger.info(f"Trial {trial.number} PRUNED")
            env.close()
            raise

        # Final evaluation — per-opponent breakdown
        per_opp = {}
        for opp_id, opp_name in enumerate(OPP_NAMES):
            result = run_tournament(
                model,
                use_implicit=True,
                opponent_id=opp_id,
                num_hands=eval_hands,
                seed=seed + opp_id,
            )
            per_opp[opp_name] = result["bb_per_100"]

        min_bb = min(per_opp.values())
        avg_bb = sum(per_opp.values()) / len(per_opp)
        worst_opp = min(per_opp, key=per_opp.get)

        if logger:
            logger.info(f"Trial {trial.number} COMPLETE")
            for opp_name, bb in per_opp.items():
                marker = " ◄ WORST" if opp_name == worst_opp else ""
                logger.info(f"  vs {opp_name}: {bb:+.2f} BB/100{marker}")
            logger.info(f"  avg = {avg_bb:+.2f}, min = {min_bb:+.2f} (objective)")

        env.close()
        return min_bb

    return objective


# ---------------------------------------------------------------------------
# Main HPO runner
# ---------------------------------------------------------------------------

def run_hpo(
    n_trials: int = 20,
    timesteps_per_trial: int = 200_000,
    eval_hands: int = 2_000,
    seed: int = 42,
    device: str = "auto",
    results_dir: str = "results",
) -> dict:
    """Run Optuna HPO and return the best hyperparameters as a dict.

    Returns a dict of kwargs suitable for passing to train_model():
        {learning_rate, n_steps, batch_size, n_epochs, gamma, ent_coef, net_arch}
    """
    if not HAS_OPTUNA:
        print("  ERROR: Optuna not installed. Run: pip install optuna")
        print("  Falling back to default hyperparameters.\n")
        return {}

    # Set up logging
    logger = setup_hpo_logger(results_dir)
    logger.info("CardShark-RL Hyperparameter Optimisation")
    logger.info(f"Trials: {n_trials}")
    logger.info(f"Steps per trial: {timesteps_per_trial:,}")
    logger.info(f"Eval hands: {eval_hands:,}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Device: {device}")
    logger.info(f"Model: Implicit (Model B) with block scheduling")
    logger.info("")

    study = optuna.create_study(
        study_name="cardshark_hpo",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100_000),
    )

    objective = create_objective(
        timesteps=timesteps_per_trial,
        eval_hands=eval_hands,
        seed=seed,
        device=device,
        logger=logger,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Log all trials summary
    logger.info("")
    logger.info(f"{'='*60}")
    logger.info(f"HPO COMPLETE — {len(study.trials)} trials")
    logger.info(f"{'='*60}")
    logger.info("")
    logger.info(f"{'Trial':>6} {'Status':>10} {'Min BB/100':>12}  Parameters")
    logger.info(f"{'-'*80}")
    for t in sorted(study.trials, key=lambda x: x.value if x.value is not None else -9999, reverse=True):
        status = t.state.name
        val = f"{t.value:+.2f}" if t.value is not None else "N/A"
        params_str = ", ".join(f"{k}={v}" for k, v in t.params.items())
        logger.info(f"{t.number:>6} {status:>10} {val:>12}  {params_str}")

    # Extract best params
    best = study.best_trial
    logger.info("")
    logger.info(f"BEST Trial #{best.number}: min BB/100 = {best.value:+.2f} (worst-case opponent)")
    logger.info(f"Best params:")
    for k, v in best.params.items():
        logger.info(f"  {k}: {v}")

    # Convert to train_model kwargs
    best_params = dict(best.params)

    # Reconstruct net_arch from n_layers and layer_size
    n_layers = best_params.pop("n_layers")
    layer_size = best_params.pop("layer_size")
    best_params["net_arch"] = [layer_size] * n_layers

    # Ensure batch_size <= n_steps
    if best_params["batch_size"] > best_params["n_steps"]:
        best_params["batch_size"] = best_params["n_steps"]

    # Save HPO results as JSON
    hpo_path = os.path.join(results_dir, "hpo_results.json")
    os.makedirs(results_dir, exist_ok=True)
    serializable_params = {k: v for k, v in best_params.items()}
    with open(hpo_path, "w") as f:
        json.dump({
            "best_value": best.value,
            "best_params": serializable_params,
            "n_trials": n_trials,
            "timesteps_per_trial": timesteps_per_trial,
            "all_trials": [
                {
                    "number": t.number,
                    "state": t.state.name,
                    "value": t.value,
                    "params": dict(t.params),
                }
                for t in study.trials
            ],
        }, f, indent=2)
    logger.info(f"\nSaved HPO results to: {hpo_path}")
    logger.info(f"Full log saved to: {os.path.join(results_dir, 'hpo_log.txt')}")

    return best_params
