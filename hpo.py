"""
hpo.py — Optuna-based hyperparameter optimisation for CardShark-RL.

Searches over PPO hyperparameters using a shortened training budget
(300K steps) and evaluates a maximin objective that maximizes the minimum BB/100 across opponents.

Can run HPO for Model A (explicit), Model B (implicit), or both.

Usage:
    python main.py --hpo                    # 40 trials, Model B (default)
    python main.py --hpo --hpo-trials 60    # custom trial count
    python main.py --hpo --hpo-model a      # HPO for Model A only
    python main.py --hpo --hpo-model both   # separate HPO for A and B
"""

from __future__ import annotations
import os

# ---------------------------------------------------------------------------
# IMPORTANT: prevent CPU oversubscription
# Must happen BEFORE torch import
# ---------------------------------------------------------------------------

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import logging
import sys
import warnings
import time
from datetime import datetime
from typing import Optional

import numpy as np
import torch

torch.set_num_threads(1)

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
from train import linear_schedule


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_hpo_logger(results_dir: str, suffix: str = "") -> logging.Logger:
    """Create a logger that writes to both console and a log file."""
    name = f"cardshark_hpo{suffix}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    os.makedirs(results_dir, exist_ok=True)
    fname = f"hpo_log{suffix}.txt"
    log_path = os.path.join(results_dir, fname)
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
# HPO objective function
# ---------------------------------------------------------------------------

def _weighted_bb(per_opp: dict[str, float]) -> float:
    """Compute the HPO objective: weighted average that penalises Rock weakness.

    Weights:
        Rock:          0.5  (hardest to exploit, gets extra weight)
        CallingStation: 0.25
        Maniac:        0.25

    This directly pushes the search towards configs that don't abandon
    Rock exploitation to farm easy Maniac wins.
    """
    weights = {"Rock": 0.5, "CallingStation": 0.25, "Maniac": 0.25}
    return sum(weights[k] * per_opp[k] for k in OPP_NAMES)


# ---------------------------------------------------------------------------
# Pruning callback
# ---------------------------------------------------------------------------

class TrialEvalCallback(BaseCallback):
    """Periodically evaluate the model during training and report to Optuna."""

    def __init__(
        self,
        trial: "optuna.Trial",
        use_implicit: bool = True,
        eval_interval: int = 25_000,
        eval_hands: int = 500,
        seed: int = 0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.trial = trial
        self.use_implicit = use_implicit
        self.eval_interval = eval_interval
        self.eval_hands = eval_hands
        self.seed = seed
        self._last_eval_step = 0
        self.best_score = -999.0
        self.last_wall_time = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step >= self.eval_interval:
            now = time.time()

            elapsed = now - self.last_wall_time
            self.last_wall_time = now

            print(
                f"[Trial {self.trial.number}] "
                f"reached {self.num_timesteps:,} steps "
                f"(+{elapsed:.1f}s since last report)",
                flush=True,
            )

            self._last_eval_step = self.num_timesteps

            eval_start = time.time()

            per_opp = {}

            for opp_id, opp_name in enumerate(OPP_NAMES):
                print(
                    f"[Trial {self.trial.number}] "
                    f"evaluating vs {opp_name}...",
                    flush=True,
                )

                result = run_tournament(
                    self.model,
                    use_implicit=self.use_implicit,
                    opponent_id=opp_id,
                    num_hands=self.eval_hands,
                    seed=self.seed + opp_id,
                )

                per_opp[opp_name] = result["bb_per_100"]

            eval_time = time.time() - eval_start

            score = min(per_opp.values())

            print(
                f"[Trial {self.trial.number}] "
                f"EVAL DONE in {eval_time:.1f}s | "
                f"min_bb={score:+.2f} | "
                + " | ".join(f"{k}={v:+.1f}" for k, v in per_opp.items()),
                flush=True,
            )

            self.trial.report(score, step=self.num_timesteps)

            if self.trial.should_prune():
                print(f"[Trial {self.trial.number}] PRUNED", flush=True)
                raise TrialPruned()

            self.best_score = max(self.best_score, score)

        return True


class DebugCallback(BaseCallback):
    def _on_step(self):
        if self.num_timesteps % 1000 == 0:
            print(
                f"[Trial] {self.num_timesteps} timesteps",
                flush=True,
            )
        return True

# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------

def create_objective(
    use_implicit: bool = True,
    timesteps: int = 300_000,
    eval_hands: int = 2_000,
    seed: int = 42,
    device: str = "auto",
    n_envs: int = 1,
    logger: Optional[logging.Logger] = None,
):
    """Create an Optuna objective closure.

    Args:
        use_implicit: If True, optimise Model B (implicit/block scheduling).
                      If False, optimise Model A (explicit/random scheduling).
        n_envs: Number of parallel environments per trial.
    """

    def objective(trial: "optuna.Trial") -> float:
        """Train a model with sampled hyperparameters and return weighted BB/100."""

        # ------------------------------------------------------------------
        # Sample hyperparameters
        # ------------------------------------------------------------------

        learning_rate = trial.suggest_float(
            "learning_rate",
            1e-4,
            5e-4,
            log=True,
        )

        ent_coef = trial.suggest_float(
            "ent_coef",
            0.002,
            0.015,
            log=True,
        )

        clip_range = trial.suggest_float(
            "clip_range",
            0.15,
            0.25,
        )

        vf_coef = trial.suggest_float(
            "vf_coef",
            0.3,
            0.7,
        )

        max_grad_norm = trial.suggest_float(
            "max_grad_norm",
            0.3,
            0.7,
        )

        fold_penalty = trial.suggest_float(
            "fold_penalty",
            0.2,
            0.4,
        )

        steal_bonus = trial.suggest_float(
            "steal_bonus",
            0.1,
            0.5,
        )

        if use_implicit:
            # Model B Search Space (Implicit)
            n_steps = trial.suggest_categorical("n_steps", [2048])
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
            n_epochs = trial.suggest_int("n_epochs", 5, 12)
            gamma = trial.suggest_float("gamma", 0.98, 1.0)
            gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.98)
            
            lr_schedule = trial.suggest_categorical("lr_schedule", ["linear"])
            rolling_window = trial.suggest_categorical("rolling_window", [20, 50, 100])
            block_size = trial.suggest_categorical("block_size", [400])
            
            n_layers = trial.suggest_int("n_layers", 4, 4)
            layer_size = trial.suggest_categorical("layer_size", [256])
            
            hybrid_switch_timesteps_frac = trial.suggest_float("hybrid_switch_timesteps_frac", 0.7, 1.0)
        else:
            # Model A Search Space (Explicit)
            n_steps = trial.suggest_categorical("n_steps", [2048, 4096])
            batch_size = trial.suggest_categorical("batch_size", [32, 64])
            n_epochs = trial.suggest_int("n_epochs", 15, 25)
            gamma = trial.suggest_float("gamma", 0.95, 0.98)
            gae_lambda = trial.suggest_float("gae_lambda", 0.85, 0.95)
            
            lr_schedule = trial.suggest_categorical("lr_schedule", ["linear", "constant"])
            rolling_window = trial.suggest_categorical("rolling_window", [5, 10, 20])
            block_size = trial.suggest_categorical("block_size", [50, 100, 200])
            
            n_layers = trial.suggest_int("n_layers", 2, 3)
            layer_size = trial.suggest_categorical("layer_size", [128, 256])
            
            hybrid_switch_timesteps_frac = trial.suggest_float("hybrid_switch_timesteps_frac", 0.1, 0.9)

        hybrid_switch_ts = int(hybrid_switch_timesteps_frac * timesteps)

        net_arch = [layer_size] * n_layers
        opponent_schedule = "hybrid"

        # Ensure batch_size is valid for vectorised env
        total_rollout = n_steps * n_envs
        if batch_size > total_rollout:
            batch_size = total_rollout

        # Log trial parameters locally
        trial_header = f"""
        ============================================================
        Trial {trial.number} starting ({'implicit' if use_implicit else 'explicit'}) | n_envs={n_envs}
        learning_rate: {learning_rate:.6f} ({lr_schedule})
        n_steps: {n_steps} | batch_size: {batch_size} | n_epochs: {n_epochs}
        gamma: {gamma:.4f} | ent_coef: {ent_coef:.6f}
        clip_range: {clip_range:.3f} | gae_lambda: {gae_lambda:.3f}
        vf_coef: {vf_coef:.3f} | max_grad_norm: {max_grad_norm:.3f}
        fold_penalty: {fold_penalty:.3f} | steal_bonus: {steal_bonus:.3f}
        rolling_window: {rolling_window} | block_size: {block_size}
        switch_frac: {hybrid_switch_timesteps_frac:.3f} | switch_ts: {hybrid_switch_ts}
        net_arch: {net_arch}
        ============================================================
        """

        print(trial_header, flush=True)

        # ------------------------------------------------------------------
        # Build vectorized environment
        # ------------------------------------------------------------------
        from stable_baselines3.common.vec_env import DummyVecEnv

        def _make_env(env_seed: int):
            def _init():
                raw = DrawPokerGymEnv(
                    use_implicit_modeling=use_implicit,
                    opponent_id=None,
                    rolling_window=rolling_window,
                    rng_seed=env_seed,
                    opponent_schedule=opponent_schedule,
                    block_size=block_size,
                    hybrid_switch_timesteps=hybrid_switch_ts,
                    fold_penalty=fold_penalty,
                    steal_bonus=steal_bonus,
                )
                return Monitor(ActionMasker(raw, mask_fn))
            return _init

        env = DummyVecEnv([_make_env(seed + i) for i in range(n_envs)])

        # LR param
        if lr_schedule == "linear":
            lr_param = linear_schedule(learning_rate)
        else:
            lr_param = learning_rate

        # ------------------------------------------------------------------
        # Create model
        # ------------------------------------------------------------------
        policy_kwargs = dict(net_arch=net_arch)
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=lr_param,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            ent_coef=ent_coef,
            clip_range=clip_range,
            gae_lambda=gae_lambda,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=0,
            seed=seed,
            device=device,
            policy_kwargs=policy_kwargs,
        )

        # Pruning callback
        eval_callback = TrialEvalCallback(
            trial=trial,
            use_implicit=use_implicit,
            eval_interval=25_000,
            eval_hands=eval_hands // 4,
            seed=seed,
        )

        debug_callback = DebugCallback()

        train_start = time.time()

        try:
            print(
                f"[Trial {trial.number}] starting learn()...",
                flush=True,
            )

            model.learn(
                total_timesteps=timesteps,
                callback=eval_callback,
                progress_bar=False,
            )

            train_time = time.time() - train_start

            print(
                f"[Trial {trial.number}] training finished "
                f"in {train_time/60:.1f} minutes",
                flush=True,
            )

        except TrialPruned:
            print(f"[Trial {trial.number}] PRUNED", flush=True)
            env.close()
            raise

        # ------------------------------------------------------------------
        # Final evaluation with weighted objective
        # ------------------------------------------------------------------
        per_opp = {}
        for opp_id, opp_name in enumerate(OPP_NAMES):
            result = run_tournament(
                model,
                use_implicit=use_implicit,
                opponent_id=opp_id,
                num_hands=eval_hands,
                seed=seed + opp_id,
            )
            per_opp[opp_name] = result["bb_per_100"]

        weighted = _weighted_bb(per_opp)
        min_bb = min(per_opp.values())
        worst_opp = min(per_opp, key=per_opp.get)

        print(f"Trial {trial.number} COMPLETE")
        for opp_name, bb in per_opp.items():
            marker = " < WORST" if opp_name == worst_opp else ""
            print(f"  vs {opp_name}: {bb:+.2f} BB/100{marker}")
        print(f"  weighted={weighted:+.2f}, min={min_bb:+.2f}")

        env.close()
        return min_bb

    return objective


# ---------------------------------------------------------------------------
# Main HPO runner (single model)
# ---------------------------------------------------------------------------

def run_hpo_single(
    use_implicit: bool = True,
    n_trials: int = 40,
    timesteps_per_trial: int = 300_000,
    eval_hands: int = 2_000,
    seed: int = 42,
    device: str = "auto",
    n_envs: int = 1,
    n_jobs: int = 1,
    results_dir: str = "results",
    suffix: str = "",
) -> dict:
    """Run Optuna HPO for one model (implicit or explicit).

    Args:
        n_envs: Parallel environments per trial.
        n_jobs: Parallel trials (using multiple CPU cores).
    """
    if not HAS_OPTUNA:
        print("  ERROR: Optuna not installed. Run: pip install optuna")
        print("  Falling back to default hyperparameters.\n")
        return {}

    model_label = "Model B (Implicit)" if use_implicit else "Model A (Explicit)"
    logger = setup_hpo_logger(results_dir, suffix=suffix)
    logger.info(f"CardShark-RL HPO — {model_label}")
    logger.info(f"Trials: {n_trials}")
    logger.info(f"Steps per trial: {timesteps_per_trial:,}")
    logger.info(f"Eval hands: {eval_hands:,}")
    logger.info(f"Objective: Maximin (maximize minimum BB/100 across opponents)")
    logger.info(f"Seed: {seed} | Device: {device} | n_envs: {n_envs} | n_jobs: {n_jobs}")
    logger.info("")

    study = optuna.create_study(
        study_name=f"cardshark_hpo{'_implicit' if use_implicit else '_explicit'}",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=150_000),
    )

    objective = create_objective(
        use_implicit=use_implicit,
        timesteps=timesteps_per_trial,
        eval_hands=eval_hands,
        seed=seed,
        device=device,
        n_envs=n_envs,
        logger=logger,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    # Log summary
    logger.info("")
    logger.info(f"{'='*60}")
    logger.info(f"HPO COMPLETE — {len(study.trials)} trials")
    logger.info(f"{'='*60}")
    logger.info("")
    logger.info(f"{'Trial':>6} {'Status':>10} {'Min BB':>13}  Parameters")
    logger.info(f"{'-'*80}")
    for t in sorted(study.trials, key=lambda x: x.value if x.value is not None else -9999, reverse=True):
        status = t.state.name
        val = f"{t.value:+.2f}" if t.value is not None else "N/A"
        params_str = ", ".join(f"{k}={v}" for k, v in t.params.items())
        logger.info(f"{t.number:>6} {status:>10} {val:>13}  {params_str}")

    best = study.best_trial
    logger.info("")
    logger.info(f"BEST Trial #{best.number}: min BB/100 = {best.value:+.2f}")
    logger.info("Best params:")
    for k, v in best.params.items():
        logger.info(f"  {k}: {v}")

    # Convert to train_model kwargs
    best_params = dict(best.params)

    # Reconstruct net_arch
    n_layers = best_params.pop("n_layers")
    layer_size = best_params.pop("layer_size")
    best_params["net_arch"] = [layer_size] * n_layers

    # Handle hybrid_switch_episodes_frac (implicit only)
    if "hybrid_switch_episodes_frac" in best_params:
        frac = best_params.pop("hybrid_switch_episodes_frac")
        best_params["hybrid_switch_timesteps"] = int(frac * timesteps_per_trial)

    # Ensure batch_size <= n_steps
    if best_params["batch_size"] > best_params["n_steps"]:
        best_params["batch_size"] = best_params["n_steps"]

    # Save HPO results as JSON
    fname = f"hpo_results{suffix}.json"
    hpo_path = os.path.join(results_dir, fname)
    os.makedirs(results_dir, exist_ok=True)
    with open(hpo_path, "w") as f:
        json.dump({
            "model": "implicit" if use_implicit else "explicit",
            "objective": "weighted_bb (Rock×0.5, Station×0.25, Maniac×0.25)",
            "best_value": best.value,
            "best_params": {k: v for k, v in best_params.items()},
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

    return best_params


# ---------------------------------------------------------------------------
# Main HPO runner (dispatches to single or both models)
# ---------------------------------------------------------------------------

def run_hpo(
    n_trials: int = 40,
    timesteps_per_trial: int = 300_000,
    eval_hands: int = 2_000,
    seed: int = 42,
    device: str = "auto",
    n_envs: int = 1,
    n_jobs: int = 1,
    results_dir: str = "results",
    hpo_model: str = "b",  # "a", "b", or "both"
) -> dict:
    """Run HPO and return best params dict(s).

    Args:
        hpo_model: Which model to optimise.
            "b"    → return single params dict (applied to Model B, defaults used for A)
            "a"    → return single params dict (applied to Model A, defaults used for B)
            "both" → return {"model_a": {...}, "model_b": {...}}

    Returns:
        For "a" or "b": dict of kwargs for train_model()
        For "both": dict with keys "model_a" and "model_b"
    """
    if not HAS_OPTUNA:
        print("  ERROR: Optuna not installed. Run: pip install optuna")
        return {}

    if hpo_model == "both":
        print("\n  Running HPO for Model A (Explicit)...")
        params_a = run_hpo_single(
            use_implicit=False, n_trials=n_trials,
            timesteps_per_trial=timesteps_per_trial, eval_hands=eval_hands,
            seed=seed, device=device, n_envs=n_envs, n_jobs=n_jobs,
            results_dir=results_dir, suffix="_model_a",
        )
        print("\n  Running HPO for Model B (Implicit)...")
        params_b = run_hpo_single(
            use_implicit=True, n_trials=n_trials,
            timesteps_per_trial=timesteps_per_trial, eval_hands=eval_hands,
            seed=seed, device=device, n_envs=n_envs, n_jobs=n_jobs,
            results_dir=results_dir, suffix="_model_b",
        )
        # Save combined best_params.json
        _save_best_params(params_a, params_b, results_dir)
        return {"model_a": params_a, "model_b": params_b}

    elif hpo_model == "a":
        params = run_hpo_single(
            use_implicit=False, n_trials=n_trials,
            timesteps_per_trial=timesteps_per_trial, eval_hands=eval_hands,
            seed=seed, device=device, n_envs=n_envs, n_jobs=n_jobs,
            results_dir=results_dir, suffix="_model_a",
        )
        _save_best_params(params, None, results_dir)
        return {"model_a": params}

    else:  # "b" (default)
        params = run_hpo_single(
            use_implicit=True, n_trials=n_trials,
            timesteps_per_trial=timesteps_per_trial, eval_hands=eval_hands,
            seed=seed, device=device, n_envs=n_envs, n_jobs=n_jobs,
            results_dir=results_dir, suffix="_model_b",
        )
        _save_best_params(None, params, results_dir)
        return {"model_b": params}


def _save_best_params(params_a: dict | None, params_b: dict | None, results_dir: str):
    """Persist best params to best_params_a.json and best_params_b.json."""
    if params_a is not None:
        path_a = "best_params_a.json"
        with open(path_a, "w") as f:
            # Save as flat dict for clarity
            json.dump(params_a, f, indent=2)
        print(f"  Saved Model A params to: {path_a}")

    if params_b is not None:
        path_b = "best_params_b.json"
        with open(path_b, "w") as f:
            json.dump(params_b, f, indent=2)
        print(f"  Saved Model B params to: {path_b}")
