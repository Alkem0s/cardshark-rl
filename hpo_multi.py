"""
hpo_multi.py — Optuna-based Hyperparameter Optimisation for Multi-Player CardShark-RL.

Searches over PPO and game-level hyperparameters for the 5-Seat Freezeout Table,
evaluating a composite multi-player tournament objective:
    Fitness = (BB / 100) + 0.5 * (Survival Rate %) + 1.0 * (1st Place Rate %)

Features:
- Automated Pruning (MedianPruner) after intermediate evaluations to abort weak trials early.
- Atomic saving of `best_params_multi.json` at every new best trial.
- Full parallel CPU vectorization across n_envs (optimized for high-core machines).
- Considers all newly added parameters (blind escalation, fold penalty, stack depth, ent_coef).

Usage:
    python hpo_multi.py --n-trials 25 --n-envs 8 --timesteps 100000
    python hpo_multi.py --smoke-test
"""

from __future__ import annotations
import os

# Prevent CPU oversubscription before importing torch / numpy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import time
import argparse
import logging
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import torch
torch.set_num_threads(1)

import optuna
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from multi_gym_wrapper import make_multi_env, multi_mask_fn
from evaluate_multi import evaluate_tournament_sessions
from train_multi import linear_schedule


# ---------------------------------------------------------------------------
# Logging & Checkpoint Persistence
# ---------------------------------------------------------------------------

def setup_hpo_logger(log_file: str = "results/hpo_multi_log.txt") -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("hpo_multi")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File output
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    # Console output
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("  %(message)s"))
    logger.addHandler(ch)

    return logger


def save_best_params(params: dict, filepath: str = "best_params_multi.json"):
    """Atomically saves the best hyperparameters as a JSON file."""
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    os.replace(tmp_path, filepath)


# ---------------------------------------------------------------------------
# Intermediate Evaluation & Pruning Callback
# ---------------------------------------------------------------------------

class MultiTrialEvalCallback(BaseCallback):
    """Evaluates the model every eval_interval steps and reports to Optuna."""

    def __init__(
        self,
        trial: optuna.Trial,
        eval_interval: int = 25_000,
        eval_sessions: int = 12,
        blind_escalation: int = 15,
        seed: int = 42,
    ):
        super().__init__()
        self.trial = trial
        self.eval_interval = eval_interval
        self.eval_sessions = eval_sessions
        self.blind_escalation = blind_escalation
        self.seed = seed
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step >= self.eval_interval:
            self._last_eval_step = self.num_timesteps
            eval_res = evaluate_tournament_sessions(
                model_path=self.model,
                num_sessions=self.eval_sessions,
                starting_chips=200,
                blind_escalation_interval=self.blind_escalation,
                randomize_stacks=True,
                max_hands=120,
                seed=self.seed + self.num_timesteps,
                verbose=False,
            )

            # Intermediate fitness score
            score = (
                eval_res["bb_per_100"]
                + 0.5 * eval_res["survival_rate_pct"]
                + 1.0 * eval_res["first_place_rate_pct"]
            )

            self.trial.report(score, self.num_timesteps)

            if self.trial.should_prune():
                print(f"  [Optuna] Trial {self.trial.number} PRUNED at step {self.num_timesteps:,} (Score: {score:+.2f})")
                raise TrialPruned()

        return True


# ---------------------------------------------------------------------------
# Objective Function
# ---------------------------------------------------------------------------

def create_objective(
    timesteps_per_trial: int = 150_000,
    n_envs: int = 16,
    eval_sessions: int = 40,
    seed: int = 42,
    best_params_path: str = "best_params_multi.json",
    logger: Optional[logging.Logger] = None,
):
    best_score = -float("inf")

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_score

        # 1. Refined PPO Hyperparameters (pruned unviable ranges)
        learning_rate = trial.suggest_float("learning_rate", 2.0e-4, 3.8e-4, log=True)
        n_steps = trial.suggest_categorical("n_steps", [1024, 1536])
        batch_size = trial.suggest_categorical("batch_size", [64, 128])
        n_epochs = trial.suggest_int("n_epochs", 7, 10)
        gamma = trial.suggest_float("gamma", 0.988, 0.996)
        gae_lambda = trial.suggest_float("gae_lambda", 0.91, 0.96)
        clip_range = trial.suggest_float("clip_range", 0.17, 0.23)
        ent_coef = trial.suggest_float("ent_coef", 0.006, 0.013, log=True)
        vf_coef = trial.suggest_float("vf_coef", 0.40, 0.58)
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.4, 0.7)

        # 2. Refined Game & Tournament Dynamics
        fold_penalty = trial.suggest_float("fold_penalty", 0.18, 0.30)
        blind_escalation = trial.suggest_categorical("blind_escalation", [14, 15, 16])

        # 3. Dedicated High-Capacity Network Architecture (256 units required for 63-dim input)
        n_layers = trial.suggest_categorical("n_layers", [3, 4])
        layer_size = 256
        net_arch = [layer_size] * n_layers

        total_rollout = n_steps * n_envs
        if batch_size > total_rollout:
            batch_size = total_rollout

        header = (
            f"\n--- [Trial {trial.number}] STARTING ---\n"
            f"  LR: {learning_rate:.6f} | n_steps: {n_steps} | batch_size: {batch_size} | epochs: {n_epochs}\n"
            f"  gamma: {gamma:.4f} | gae: {gae_lambda:.3f} | clip: {clip_range:.2f} | ent: {ent_coef:.5f}\n"
            f"  vf: {vf_coef:.2f} | fold_penalty: {fold_penalty:.2f} | blind_escalation: every {blind_escalation} hands\n"
            f"  net_arch: {net_arch}"
        )
        print(header, flush=True)

        # Build vectorized multi-player environments
        def _make_env(s: int):
            raw = make_multi_env(
                num_seats=5,
                starting_chips=200,
                blind_escalation_interval=blind_escalation,
                randomize_stacks=True,
                fold_penalty=fold_penalty,
                max_hands_per_session=150,
                seed=seed + trial.number * 1000 + s * 10,
            )()
            return Monitor(ActionMasker(raw, multi_mask_fn))

        env = DummyVecEnv([lambda s=i: _make_env(s) for i in range(n_envs)])

        policy_kwargs = dict(
            net_arch=dict(pi=net_arch, vf=net_arch),
        )

        model = MaskablePPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=linear_schedule(learning_rate),
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed + trial.number,
        )

        # Pruning callback
        eval_callback = MultiTrialEvalCallback(
            trial=trial,
            eval_interval=max(10_000, timesteps_per_trial // 4),
            eval_sessions=12,
            blind_escalation=blind_escalation,
            seed=seed + trial.number,
        )

        train_start = time.time()
        try:
            model.learn(
                total_timesteps=timesteps_per_trial,
                callback=eval_callback,
                progress_bar=False,
            )
        except TrialPruned:
            env.close()
            raise
        finally:
            env.close()

        train_dur = time.time() - train_start

        # ------------------------------------------------------------------
        # Final Evaluation across Tournament Sessions
        # ------------------------------------------------------------------
        res = evaluate_tournament_sessions(
            model_path=model,
            num_sessions=eval_sessions,
            starting_chips=200,
            blind_escalation_interval=blind_escalation,
            randomize_stacks=True,
            max_hands=180,
            seed=seed + 9999,
            verbose=False,
        )

        bb_100 = res["bb_per_100"]
        survival = res["survival_rate_pct"]
        win_rate = res["first_place_rate_pct"]
        chip_share = res["avg_final_chip_share_pct"]

        # Composite Fitness Metric
        fitness_score = bb_100 + 0.5 * survival + 1.0 * win_rate

        summary = (
            f"--- [Trial {trial.number} RESULTS] ({train_dur/60:.1f} mins) ---\n"
            f"  Winrate:       {bb_100:+.2f} BB/100\n"
            f"  Survival:      {survival:.1f}%\n"
            f"  1st Place:     {win_rate:.1f}%\n"
            f"  Avg Chip Share: {chip_share:.1f}%\n"
            f"  Composite Score: {fitness_score:+.2f}"
        )
        print(summary, flush=True)

        if logger:
            logger.info(
                f"Trial {trial.number} | Score: {fitness_score:+.2f} | BB/100: {bb_100:+.2f} | "
                f"Survival: {survival:.1f}% | 1st: {win_rate:.1f}% | LR: {learning_rate:.6f} | Ent: {ent_coef:.5f}"
            )

        # Update best params if record broken
        if fitness_score > best_score:
            best_score = fitness_score
            best_dict = {
                "fitness_score": round(fitness_score, 2),
                "bb_per_100": round(bb_100, 2),
                "survival_rate_pct": round(survival, 1),
                "first_place_rate_pct": round(win_rate, 1),
                "avg_chip_share_pct": round(chip_share, 1),
                "learning_rate": learning_rate,
                "lr_schedule": "linear",
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_range": clip_range,
                "ent_coef": ent_coef,
                "vf_coef": vf_coef,
                "max_grad_norm": max_grad_norm,
                "fold_penalty": fold_penalty,
                "blind_escalation": blind_escalation,
                "net_arch": net_arch,
            }
            save_best_params(best_dict, filepath=best_params_path)
            print(f"  >>> [NEW BEST RECORD!] Saved parameters to {best_params_path} <<<")

        return fitness_score

    return objective


# ---------------------------------------------------------------------------
# Main Optimization Runner
# ---------------------------------------------------------------------------

def run_multi_hpo(
    n_trials: int = 25,
    timesteps_per_trial: int = 100_000,
    n_envs: int = 8,
    eval_sessions: int = 25,
    seed: int = 42,
    output_params: str = "best_params_multi.json",
    log_file: str = "results/hpo_multi_log.txt",
    smoke_test: bool = False,
):
    if smoke_test:
        print("=== RUNNING MULTI-PLAYER HPO SMOKE TEST (1 trial, 500 steps) ===")
        n_trials = 1
        timesteps_per_trial = 500
        n_envs = 2
        eval_sessions = 5

    logger = setup_hpo_logger(log_file)
    logger.info("=== STARTING CARDSHARK-RL MULTI-PLAYER HPO ===")
    logger.info(f"Trials: {n_trials} | Steps/Trial: {timesteps_per_trial:,} | Parallel Envs: {n_envs}")

    # Median pruner ignores the first 3 trials (warmup)
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=timesteps_per_trial // 3)
    study = optuna.create_study(
        study_name="cardshark_multiplayer_hpo",
        direction="maximize",
        pruner=pruner,
    )

    objective_fn = create_objective(
        timesteps_per_trial=timesteps_per_trial,
        n_envs=n_envs,
        eval_sessions=eval_sessions,
        seed=seed,
        best_params_path=output_params,
        logger=logger,
    )

    study.optimize(
        objective_fn,
        n_trials=n_trials,
        show_progress_bar=False,
    )

    print("\n" + "=" * 60)
    print("  HPO SEARCH COMPLETE!")
    print("=" * 60)
    print(f"  Best Composite Score: {study.best_value:+.2f}")
    print(f"  Best Parameters saved to: {output_params}")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print("=" * 60 + "\n")

    return study.best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Player CardShark-RL Hyperparameter Optimisation")
    parser.add_argument("--n-trials", type=int, default=25, help="Number of Optuna trials (default: 25)")
    parser.add_argument("--timesteps", type=int, default=150_000, help="Timesteps per trial (default: 150,000)")
    parser.add_argument("--n-envs", type=int, default=16, help="Number of parallel CPU environments per trial (default: 16)")
    parser.add_argument("--eval-sessions", type=int, default=40, help="Sessions per final trial evaluation (default: 40)")
    parser.add_argument("--output-params", type=str, default="best_params_multi.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    run_multi_hpo(
        n_trials=args.n_trials,
        timesteps_per_trial=args.timesteps,
        n_envs=args.n_envs,
        eval_sessions=args.eval_sessions,
        seed=args.seed,
        output_params=args.output_params,
        smoke_test=args.smoke_test,
    )
