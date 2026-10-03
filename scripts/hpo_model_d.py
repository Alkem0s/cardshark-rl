"""
hpo_model_d.py — Advanced Hyperparameter Optimization for Model D Champion.

Design Philosophy & Mathematical Decomposition:
Hyperparameters in Deep Reinforcement Learning fall into distinct mathematical subspaces:

1. Representation Subspace (Card Attention Transformer):
   - Parameters: embed_dim, num_heads, pooling, card_features_dim, net_arch_depth.
   - Property: Orthogonal to PPO optimization dynamics. Governs the inductive bias
     and token embedding geometry of the 5-card unordered hand representation.

2. Gradient Descent Optimization Subspace (The Coupled SGD Manifold):
   - Parameters: learning_rate (peak), min_lr_ratio, n_restart_cycles, batch_size,
     n_epochs, n_steps, clip_range, ent_coef, gamma, gae_lambda, vf_coef, max_grad_norm.
   - Property: Strongly coupled through stochastic gradient descent backprop updates.
     Learning rate interacts with batch size (noise scale) and clip range (surrogate boundary).
     These parameters MUST be searched jointly to avoid optimizer instability.

3. MDP & Tournament Subspace (Reward Shaping & Environment Dynamics):
   - Parameters: fold_penalty, blind_escalation.
   - Property: Modifies the MDP reward function and tournament horizon.

Operational Modes:
- `--mode staged` (Decoupled Factorized Search):
  Executes coordinate block search: Stage 1 tunes representation, Stage 2 tunes the
  coupled gradient descent dynamics, Stage 3 fine-tunes tournament parameters.
  Dramatically reduces search complexity: O(D_A + D_B + D_C) vs O(D_A * D_B * D_C).
- `--mode joint` (Brute-Force / Multivariate TPE Search):
  Searches the entire combined hyperparameter space at once using Optuna's multivariate
  TPE sampler, ensuring any non-linear cross-interactions between attention capacity
  and gradient descent steps are discovered.

Usage:
    python hpo_model_d.py --mode staged --n-trials 20 --n-envs 8 --timesteps 100000
    python hpo_model_d.py --mode joint --n-trials 30 --n-envs 8 --timesteps 100000
    python hpo_model_d.py --smoke-test
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
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
torch.set_num_threads(1)

import optuna
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner, PercentilePruner
from optuna.samplers import TPESampler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.multi_gym_wrapper import (
    MultiDrawPokerGymEnv,
    make_superhuman_multi_env,
    SUPERHUMAN_OBS_DIM,
)
from rl.league import LeaguePool
from rl.card_attention import CardAttentionExtractor
from scripts.train_model_d import cosine_warm_restart_schedule


# ---------------------------------------------------------------------------
# Logging & Checkpoint Persistence
# ---------------------------------------------------------------------------

def setup_logger(log_file: str = "results/hpo_model_d_log.txt") -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("hpo_model_d")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("  %(message)s"))
    logger.addHandler(ch)

    return logger


def save_best_params(params: dict, filepath: str = "configs/best_params_d.json"):
    """Atomically saves the best hyperparameters as a JSON file."""
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    os.replace(tmp_path, filepath)


# ---------------------------------------------------------------------------
# Tournament Evaluation Function
# ---------------------------------------------------------------------------

def evaluate_model_d_sessions(
    model: MaskablePPO,
    num_sessions: int = 30,
    starting_chips: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    max_hands: int = 150,
    blind_escalation_interval: Optional[int] = 15,
    randomize_stacks: bool = True,
    seed: int = 42,
) -> dict:
    """Evaluates Model D policy across multi-player freezeout tournament sessions."""
    env = MultiDrawPokerGymEnv(
        num_seats=5,
        starting_chips=starting_chips,
        small_blind=small_blind,
        big_blind=big_blind,
        hero_seat=0,
        max_hands_per_session=max_hands,
        blind_escalation_interval=blind_escalation_interval,
        randomize_stacks=randomize_stacks,
        superhuman_obs=True,
        rng_seed=seed,
    )

    survived_count = 0
    first_place_count = 0
    total_hands_played = 0
    total_net_chips = 0
    final_chip_shares = []
    total_table_chips = starting_chips * 5

    for sess_idx in range(num_sessions):
        obs, info = env.reset(seed=seed + sess_idx * 23)
        done = False
        initial_chips = starting_chips

        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        hero_chips = info["hero_chips"]
        hands_in_sess = info["hands_played"]
        total_hands_played += hands_in_sess
        net_chips = hero_chips - initial_chips
        total_net_chips += net_chips

        chip_share = (hero_chips / total_table_chips) * 100.0
        final_chip_shares.append(chip_share)

        if hero_chips > 0:
            survived_count += 1
        if info.get("is_winner", False) or hero_chips >= total_table_chips * 0.95:
            first_place_count += 1

    survival_rate = (survived_count / max(1, num_sessions)) * 100.0
    first_place_rate = (first_place_count / max(1, num_sessions)) * 100.0
    avg_chip_share = float(np.mean(final_chip_shares)) if final_chip_shares else 0.0
    bb_per_100 = (total_net_chips / big_blind) / (max(1, total_hands_played) / 100.0)

    return {
        "bb_per_100": bb_per_100,
        "survival_rate_pct": survival_rate,
        "first_place_rate_pct": first_place_rate,
        "avg_final_chip_share_pct": avg_chip_share,
    }


# ---------------------------------------------------------------------------
# Intermediate Evaluation & Pruning Callback
# ---------------------------------------------------------------------------

class ModelDTrialEvalCallback(BaseCallback):
    """Evaluates the model periodically and reports fitness to Optuna for early pruning."""

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
            # Don't prune at or near final step — let the trial complete and run final tournament evaluation!
            total_ts = getattr(self.model, "_total_timesteps", None)
            if total_ts is not None and self.num_timesteps >= total_ts - 2000:
                return True

            self._last_eval_step = self.num_timesteps
            eval_res = evaluate_model_d_sessions(
                model=self.model,
                num_sessions=self.eval_sessions,
                starting_chips=200,
                blind_escalation_interval=self.blind_escalation,
                randomize_stacks=True,
                max_hands=120,
                seed=self.seed + self.num_timesteps,
            )

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
# Training Runner for a Single Configuration
# ---------------------------------------------------------------------------

def run_single_trial(
    config: dict,
    trial: optuna.Trial,
    timesteps: int,
    n_envs: int,
    eval_sessions: int,
    seed: int,
    league_pool: LeaguePool,
) -> Tuple[float, dict]:
    """Instantiates and trains Model D on a specific configuration, returning the fitness score."""
    embed_dim = config["embed_dim"]
    num_heads = config["num_heads"]
    pooling = config["pooling"]
    card_features_dim = config["card_features_dim"]
    net_arch = config["net_arch"]

    learning_rate = config["learning_rate"]
    min_lr = config["min_lr"]
    n_restart_cycles = config["n_restart_cycles"]
    n_steps = config["n_steps"]
    batch_size = config["batch_size"]
    n_epochs = config["n_epochs"]
    gamma = config["gamma"]
    gae_lambda = config["gae_lambda"]
    clip_range = config["clip_range"]
    ent_coef = config["ent_coef"]
    vf_coef = config["vf_coef"]
    max_grad_norm = config["max_grad_norm"]

    fold_penalty = config["fold_penalty"]
    blind_escalation = config["blind_escalation"]

    total_rollout = n_steps * n_envs
    if batch_size > total_rollout:
        batch_size = total_rollout

    env = DummyVecEnv([
        make_superhuman_multi_env(
            num_seats=5,
            starting_chips=200,
            blind_escalation_interval=blind_escalation,
            randomize_stacks=True,
            fold_penalty=fold_penalty,
            max_hands_per_session=150,
            seed=seed + trial.number * 1000 + i * 10,
            league_pool=league_pool,
        )
        for i in range(n_envs)
    ])

    policy_kwargs = dict(
        features_extractor_class=CardAttentionExtractor,
        features_extractor_kwargs=dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            pooling=pooling,
            card_features_dim=card_features_dim,
            game_features_dim=64,
            slot_features_dim=config.get("slot_features_dim", 16),
        ),
        net_arch=dict(pi=net_arch, vf=net_arch),
    )

    lr_schedule = cosine_warm_restart_schedule(
        initial_lr=learning_rate,
        min_lr=min_lr,
        n_cycles=n_restart_cycles,
        warmup_fraction=0.05,
    )

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr_schedule,
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

    eval_callback = ModelDTrialEvalCallback(
        trial=trial,
        eval_interval=max(500, timesteps // 2) if timesteps <= 5000 else max(10_000, timesteps // 2),
        eval_sessions=min(18, eval_sessions),
        blind_escalation=blind_escalation,
        seed=seed + trial.number,
    )

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=eval_callback,
            progress_bar=False,
        )
    finally:
        env.close()

    # Final tournament evaluation
    eval_res = evaluate_model_d_sessions(
        model=model,
        num_sessions=eval_sessions,
        starting_chips=200,
        blind_escalation_interval=blind_escalation,
        randomize_stacks=True,
        max_hands=180,
        seed=seed + 9999,
    )

    score = (
        eval_res["bb_per_100"]
        + 0.5 * eval_res["survival_rate_pct"]
        + 1.0 * eval_res["first_place_rate_pct"]
    )

    return score, eval_res


# ---------------------------------------------------------------------------
# Hyperparameter Parameter Spaces
# ---------------------------------------------------------------------------

SCREENED_BOUNDS_PATH = "configs/screened_bounds_d.json"

def load_screened_bounds(filepath: str = SCREENED_BOUNDS_PATH) -> Optional[dict]:
    """Loads screened hyperparameter bounds if previously computed by a screening run."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except Exception:
            return None
    return None


def sample_attention_architecture(trial: optuna.Trial, bounds: Optional[dict] = None) -> dict:
    """Subspace A: Inductive Bias & Token Geometry (Orthogonal to PPO Dynamics)."""
    if bounds is None:
        bounds = load_screened_bounds()

    embed_choices = bounds.get("embed_dim_choices", [32, 64]) if bounds else [32, 64]
    pooling_choices = bounds.get("pooling_choices", ["both", "attention", "mean"]) if bounds else ["both", "attention", "mean"]
    card_dim_choices = bounds.get("card_features_dim_choices", [32, 64]) if bounds else [32, 64]
    n_layers_choices = bounds.get("n_layers_choices", [3, 4]) if bounds else [3, 4]

    embed_dim = trial.suggest_categorical("embed_dim", embed_choices)
    num_heads_candidates = [h for h in [2, 4, 8] if embed_dim % h == 0]
    num_heads = trial.suggest_categorical("num_heads", num_heads_candidates if num_heads_candidates else [2])

    pooling = trial.suggest_categorical("pooling", pooling_choices)
    card_features_dim = trial.suggest_categorical("card_features_dim", card_dim_choices)
    slot_dim_choices = bounds.get("slot_features_dim_choices", [12, 16, 24]) if bounds else [12, 16, 24]
    slot_features_dim = trial.suggest_categorical("slot_features_dim", slot_dim_choices)
    n_layers = trial.suggest_categorical("n_layers", n_layers_choices)
    net_arch = [256] * n_layers

    return {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "pooling": pooling,
        "card_features_dim": card_features_dim,
        "slot_features_dim": slot_features_dim,
        "net_arch": net_arch,
    }


def sample_optimization_dynamics(trial: optuna.Trial, bounds: Optional[dict] = None) -> dict:
    """Subspace B: Coupled Stochastic Gradient Descent Dynamics."""
    if bounds is None:
        bounds = load_screened_bounds()

    lr_min = bounds.get("lr_min", 0.00010) if bounds else 0.00010
    lr_max = bounds.get("lr_max", 0.00035) if bounds else 0.00035
    batch_choices = bounds.get("batch_size_choices", [64, 128]) if bounds else [64, 128]
    n_steps_choices = bounds.get("n_steps_choices", [1024, 1536]) if bounds else [1024, 1536]
    clip_min = bounds.get("clip_range_min", 0.180) if bounds else 0.180
    clip_max = bounds.get("clip_range_max", 0.250) if bounds else 0.250
    ent_min = bounds.get("ent_coef_min", 0.003) if bounds else 0.003
    ent_max = bounds.get("ent_coef_max", 0.012) if bounds else 0.012

    learning_rate = trial.suggest_float("learning_rate", lr_min, lr_max, log=True)
    min_lr_ratio = trial.suggest_float("min_lr_ratio", 0.08, 0.20)
    min_lr = learning_rate * min_lr_ratio
    n_restart_cycles = trial.suggest_categorical("n_restart_cycles", [3, 4, 5])

    n_steps = trial.suggest_categorical("n_steps", n_steps_choices)
    batch_size = trial.suggest_categorical("batch_size", batch_choices)
    n_epochs = trial.suggest_int("n_epochs", 6, 9)

    gamma = trial.suggest_float("gamma", 0.990, 0.997)
    gae_lambda = trial.suggest_float("gae_lambda", 0.91, 0.95)
    clip_range = trial.suggest_float("clip_range", clip_min, clip_max)
    ent_coef = trial.suggest_float("ent_coef", ent_min, ent_max, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.45, 0.65)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.45, 0.75)

    return {
        "learning_rate": learning_rate,
        "min_lr": min_lr,
        "n_restart_cycles": n_restart_cycles,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
    }


def sample_tournament_dynamics(trial: optuna.Trial, bounds: Optional[dict] = None) -> dict:
    """Subspace C: MDP & Tournament Reward Dynamics."""
    if bounds is None:
        bounds = load_screened_bounds()

    fold_min = bounds.get("fold_penalty_min", 0.05) if bounds else 0.05
    fold_max = bounds.get("fold_penalty_max", 0.15) if bounds else 0.15
    fold_penalty = trial.suggest_float("fold_penalty", fold_min, fold_max)
    blind_choices = bounds.get("blind_escalation_choices", [14, 15, 16]) if bounds else [14, 15, 16]
    blind_escalation = trial.suggest_categorical("blind_escalation", blind_choices)
    return {
        "fold_penalty": fold_penalty,
        "blind_escalation": blind_escalation,
    }


def sample_screening_space(trial: optuna.Trial) -> dict:
    """Wide exploratory space probing boundary conditions to eliminate obvious bad picks."""
    embed_dim = trial.suggest_categorical("embed_dim", [16, 32, 64, 128])
    num_heads_candidates = [h for h in [2, 4, 8] if embed_dim % h == 0]
    num_heads = trial.suggest_categorical("num_heads", num_heads_candidates if num_heads_candidates else [2])
    pooling = trial.suggest_categorical("pooling", ["mean", "max", "attention", "both"])
    card_features_dim = trial.suggest_categorical("card_features_dim", [16, 32, 64])
    slot_features_dim = trial.suggest_categorical("slot_features_dim", [12, 16, 24])
    n_layers = trial.suggest_categorical("n_layers", [2, 3, 4])
    net_arch = [256] * n_layers

    # Wide range testing both slow sluggish and dangerously volatile learning rates
    learning_rate = trial.suggest_float("learning_rate", 7.0e-5, 8.0e-4, log=True)
    min_lr_ratio = trial.suggest_float("min_lr_ratio", 0.05, 0.25)
    min_lr = learning_rate * min_lr_ratio
    n_restart_cycles = trial.suggest_categorical("n_restart_cycles", [2, 3, 4, 5])
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 1536])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    n_epochs = trial.suggest_int("n_epochs", 4, 10)
    gamma = trial.suggest_float("gamma", 0.985, 0.998)
    gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.96)
    clip_range = trial.suggest_float("clip_range", 0.12, 0.30)
    ent_coef = trial.suggest_float("ent_coef", 0.002, 0.03, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.40, 0.70)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.35, 0.90)

    # Wide fold penalty (too low: calling station bleed; too high: chronic folding)
    fold_penalty = trial.suggest_float("fold_penalty", 0.05, 0.25)
    blind_escalation = trial.suggest_categorical("blind_escalation", [12, 15, 18])

    return {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "pooling": pooling,
        "card_features_dim": card_features_dim,
        "slot_features_dim": slot_features_dim,
        "net_arch": net_arch,
        "n_layers": n_layers,
        "learning_rate": learning_rate,
        "min_lr": min_lr,
        "min_lr_ratio": min_lr_ratio,
        "n_restart_cycles": n_restart_cycles,
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
    }


# ---------------------------------------------------------------------------
# Default Baseline Configurations for Decoupled Searching
# ---------------------------------------------------------------------------

DEFAULT_ARCHITECTURE = {
    "embed_dim": 64,
    "num_heads": 8,
    "pooling": "both",
    "card_features_dim": 32,
    "slot_features_dim": 16,
    "net_arch": [256, 256, 256],
}

DEFAULT_OPTIMIZATION = {
    "learning_rate": 2.5e-4,
    "min_lr": 2.5e-5,
    "n_restart_cycles": 4,
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 7,
    "gamma": 0.992,
    "gae_lambda": 0.92,
    "clip_range": 0.22,
    "ent_coef": 0.007,
    "vf_coef": 0.55,
    "max_grad_norm": 0.60,
}

DEFAULT_TOURNAMENT = {
    "fold_penalty": 0.10,
    "blind_escalation": 14,
}


# ---------------------------------------------------------------------------
# Staged / Decoupled Search Workflow
# ---------------------------------------------------------------------------

def load_params_into_subspaces(params_path: str) -> Tuple[dict, dict, dict, float]:
    """Loads a previously saved best_params_d.json into active_arch, active_opt, active_tourn."""
    with open(params_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    arch = {
        "embed_dim": data.get("embed_dim", DEFAULT_ARCHITECTURE["embed_dim"]),
        "num_heads": data.get("num_heads", DEFAULT_ARCHITECTURE["num_heads"]),
        "pooling": data.get("pooling", DEFAULT_ARCHITECTURE["pooling"]),
        "card_features_dim": data.get("card_features_dim", DEFAULT_ARCHITECTURE["card_features_dim"]),
        "slot_features_dim": data.get("slot_features_dim", DEFAULT_ARCHITECTURE.get("slot_features_dim", 16)),
        "net_arch": data.get("net_arch", DEFAULT_ARCHITECTURE["net_arch"]),
    }
    opt = {
        "learning_rate": data.get("learning_rate", DEFAULT_OPTIMIZATION["learning_rate"]),
        "min_lr": data.get("min_lr", DEFAULT_OPTIMIZATION["min_lr"]),
        "n_restart_cycles": data.get("n_restart_cycles", DEFAULT_OPTIMIZATION["n_restart_cycles"]),
        "n_steps": data.get("n_steps", DEFAULT_OPTIMIZATION["n_steps"]),
        "batch_size": data.get("batch_size", DEFAULT_OPTIMIZATION["batch_size"]),
        "n_epochs": data.get("n_epochs", DEFAULT_OPTIMIZATION["n_epochs"]),
        "gamma": data.get("gamma", DEFAULT_OPTIMIZATION["gamma"]),
        "gae_lambda": data.get("gae_lambda", DEFAULT_OPTIMIZATION["gae_lambda"]),
        "clip_range": data.get("clip_range", DEFAULT_OPTIMIZATION["clip_range"]),
        "ent_coef": data.get("ent_coef", DEFAULT_OPTIMIZATION["ent_coef"]),
        "vf_coef": data.get("vf_coef", DEFAULT_OPTIMIZATION["vf_coef"]),
        "max_grad_norm": data.get("max_grad_norm", DEFAULT_OPTIMIZATION["max_grad_norm"]),
    }
    tourn = {
        "fold_penalty": data.get("fold_penalty", DEFAULT_TOURNAMENT["fold_penalty"]),
        "blind_escalation": data.get("blind_escalation", DEFAULT_TOURNAMENT["blind_escalation"]),
    }
    score = data.get("fitness_score", -float("inf"))
    return arch, opt, tourn, score


def run_staged_hpo(
    n_trials: int = 24,
    timesteps: int = 100_000,
    n_envs: int = 16,
    eval_sessions: int = 35,
    seed: int = 42,
    params_path: str = "configs/best_params_d.json",
    n_cycles: int = 2,
    start_cycle: int = 1,
    resume: bool = False,
    logger: Optional[logging.Logger] = None,
):
    """
    Executes Decoupled / Staged HPO across 3 orthogonal subspaces using Iterative Block Coordinate Descent:
    - Pass 1: Initial coarse coordinate search starting from defaults.
    - Pass 2+: Iterative refinement using empirical champions from prior passes as active context.
    - Subspace 1: Card Attention Feature Extractor Architecture (~25% of cycle budget)
    - Subspace 2: Coupled Gradient Descent Dynamics (~60% of cycle budget)
    - Subspace 3: Tournament & Reward Shaping Dynamics (~15% of cycle budget)
    """
    print("\n============================================================")
    print(f"  STARTING MODEL D STAGED ITERATIVE HPO PIPELINE ({n_cycles} PASSES)")
    total_trials = n_trials * n_cycles
    print(f"  Budget: {n_trials} trials/pass ({total_trials} total across {n_cycles} passes) | Timesteps/Trial: {timesteps:,} | Parallel Envs: {n_envs}")

    league_pool = LeaguePool(
        base_model_path="models/model_b_multiplayer.zip",
        league_dir="models/league",
        neural_opponent_prob=0.50,
    )

    trials_per_cycle = n_trials
    trials_stage1 = max(1, int(trials_per_cycle * 0.25))
    trials_stage2 = max(1, int(trials_per_cycle * 0.60))
    trials_stage3 = max(1, trials_per_cycle - trials_stage1 - trials_stage2)

    active_arch = dict(DEFAULT_ARCHITECTURE)
    active_opt = dict(DEFAULT_OPTIMIZATION)
    active_tourn = dict(DEFAULT_TOURNAMENT)

    global_best_score = -float("inf")
    global_best_config = {}

    if (resume or start_cycle > 1) and os.path.exists(params_path):
        try:
            active_arch, active_opt, active_tourn, prev_score = load_params_into_subspaces(params_path)
            if prev_score is not None and prev_score != -float("inf"):
                global_best_score = prev_score
            print(f"\n  [Resume] Successfully loaded prior champion parameters from: {params_path}")
            if global_best_score != -float("inf"):
                print(f"    - Baseline Fitness Score: {global_best_score:+.2f}")
            print(f"    - Active Architecture: {active_arch}")
            print(f"    - Active Optimizer: LR={active_opt['learning_rate']:.6f}, Ent={active_opt['ent_coef']:.5f}")
            print(f"    - Active Tournament: Fold={active_tourn['fold_penalty']:.3f}, Blinds={active_tourn['blind_escalation']}")
        except Exception as e:
            print(f"  [Resume] Warning: Could not parse {params_path} ({e}), starting from defaults.")

    end_cycle = start_cycle + n_cycles - 1
    for cycle in range(start_cycle, end_cycle + 1):
        print(f"\n{'=' * 60}")
        print(f"  PASS {cycle} OF BLOCK COORDINATE DESCENT")
        print(f"{'=' * 60}")
        if cycle > 1 or resume:
            print("  Reusing Best Empirical Opposing Parameters from Previous Pass:")
            print(f"    - Active Architecture: {active_arch}")
            print(f"    - Active Optimizer: LR={active_opt['learning_rate']:.6f}, Ent={active_opt['ent_coef']:.5f}")
            print(f"    - Active Tournament: Fold={active_tourn['fold_penalty']:.3f}, Blinds={active_tourn['blind_escalation']}")

        # --- STAGE 1: Representation & Attention Architecture ---
        print(f"\n>>> [PASS {cycle} - STAGE 1/3] Optimizing Card Attention Architecture ({trials_stage1} trials) <<<")
        study_arch = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed + cycle * 100))

        def objective_stage1(trial: optuna.Trial) -> float:
            nonlocal global_best_score, global_best_config
            arch_config = sample_attention_architecture(trial)
            full_config = {**arch_config, **active_opt, **active_tourn}

            score, eval_res = run_single_trial(
                config=full_config,
                trial=trial,
                timesteps=timesteps,
                n_envs=n_envs,
                eval_sessions=eval_sessions,
                seed=seed + cycle * 100 + trial.number,
                league_pool=league_pool,
            )

            if score > global_best_score:
                global_best_score = score
                global_best_config = dict(full_config)
                global_best_config["fitness_score"] = round(score, 2)
                global_best_config.update(eval_res)
                save_best_params(global_best_config, params_path)

            if logger:
                logger.info(f"[Pass {cycle} - Stage 1] Trial {trial.number} | Score: {score:+.2f} | Arch: {arch_config}")
            return score

        study_arch.optimize(objective_stage1, n_trials=trials_stage1)
        best_arch_trial = study_arch.best_trial
        active_arch["embed_dim"] = best_arch_trial.params["embed_dim"]
        active_arch["num_heads"] = best_arch_trial.params["num_heads"]
        active_arch["pooling"] = best_arch_trial.params["pooling"]
        active_arch["card_features_dim"] = best_arch_trial.params["card_features_dim"]
        active_arch["slot_features_dim"] = best_arch_trial.params.get("slot_features_dim", 16)
        active_arch["net_arch"] = [256] * best_arch_trial.params["n_layers"]
        print(f"  >>> Stage 1 Complete! Active Architecture: {active_arch} (Cycle Best: {study_arch.best_value:+.2f})")

        # --- STAGE 2: Coupled Gradient Descent Dynamics ---
        print(f"\n>>> [PASS {cycle} - STAGE 2/3] Optimizing Coupled Gradient Descent Dynamics ({trials_stage2} trials) <<<")
        # In Optuna (maximize): percentile=75.0 keeps the top 75% of trials and only prunes the bottom 25% disasters
        pruner = PercentilePruner(percentile=75.0, n_startup_trials=4, n_warmup_steps=max(1000, timesteps // 2))
        study_opt = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(multivariate=True, group=True, seed=seed + cycle * 100 + 1),
            pruner=pruner,
        )

        def objective_stage2(trial: optuna.Trial) -> float:
            nonlocal global_best_score, global_best_config
            opt_config = sample_optimization_dynamics(trial)
            full_config = {**active_arch, **opt_config, **active_tourn}

            score, eval_res = run_single_trial(
                config=full_config,
                trial=trial,
                timesteps=timesteps,
                n_envs=n_envs,
                eval_sessions=eval_sessions,
                seed=seed + cycle * 100 + 100 + trial.number,
                league_pool=league_pool,
            )

            if score > global_best_score:
                global_best_score = score
                global_best_config = dict(full_config)
                global_best_config["fitness_score"] = round(score, 2)
                global_best_config.update(eval_res)
                save_best_params(global_best_config, params_path)

            if logger:
                logger.info(f"[Pass {cycle} - Stage 2] Trial {trial.number} | Score: {score:+.2f} | LR: {opt_config['learning_rate']:.6f}")
            return score

        study_opt.optimize(objective_stage2, n_trials=trials_stage2)
        best_opt_trial = study_opt.best_trial
        active_opt["learning_rate"] = best_opt_trial.params["learning_rate"]
        active_opt["min_lr"] = active_opt["learning_rate"] * best_opt_trial.params["min_lr_ratio"]
        active_opt["n_restart_cycles"] = best_opt_trial.params["n_restart_cycles"]
        active_opt["n_steps"] = best_opt_trial.params["n_steps"]
        active_opt["batch_size"] = best_opt_trial.params["batch_size"]
        active_opt["n_epochs"] = best_opt_trial.params["n_epochs"]
        active_opt["gamma"] = best_opt_trial.params["gamma"]
        active_opt["gae_lambda"] = best_opt_trial.params["gae_lambda"]
        active_opt["clip_range"] = best_opt_trial.params["clip_range"]
        active_opt["ent_coef"] = best_opt_trial.params["ent_coef"]
        active_opt["vf_coef"] = best_opt_trial.params["vf_coef"]
        active_opt["max_grad_norm"] = best_opt_trial.params["max_grad_norm"]
        print(f"  >>> Stage 2 Complete! Active Optimizer: LR={active_opt['learning_rate']:.6f}, Ent={active_opt['ent_coef']:.5f} (Cycle Best: {study_opt.best_value:+.2f})")

        # --- STAGE 3: Tournament & Reward Shaping Dynamics ---
        print(f"\n>>> [PASS {cycle} - STAGE 3/3] Fine-Tuning Tournament Dynamics ({trials_stage3} trials) <<<")
        study_tourn = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed + cycle * 100 + 2))

        def objective_stage3(trial: optuna.Trial) -> float:
            nonlocal global_best_score, global_best_config
            tourn_config = sample_tournament_dynamics(trial)
            full_config = {**active_arch, **active_opt, **tourn_config}

            score, eval_res = run_single_trial(
                config=full_config,
                trial=trial,
                timesteps=timesteps,
                n_envs=n_envs,
                eval_sessions=eval_sessions,
                seed=seed + cycle * 100 + 200 + trial.number,
                league_pool=league_pool,
            )

            if score > global_best_score:
                global_best_score = score
                global_best_config = dict(full_config)
                global_best_config["fitness_score"] = round(score, 2)
                global_best_config.update(eval_res)
                save_best_params(global_best_config, params_path)

            if logger:
                logger.info(f"[Pass {cycle} - Stage 3] Trial {trial.number} | Score: {score:+.2f} | FoldPenalty: {tourn_config['fold_penalty']:.2f}")
            return score

        study_tourn.optimize(objective_stage3, n_trials=trials_stage3)
        best_tourn_trial = study_tourn.best_trial
        active_tourn["fold_penalty"] = best_tourn_trial.params["fold_penalty"]
        active_tourn["blind_escalation"] = best_tourn_trial.params["blind_escalation"]
        print(f"  >>> Stage 3 Complete! Active Tournament: Fold={active_tourn['fold_penalty']:.3f}, Blinds={active_tourn['blind_escalation']} (Cycle Best: {study_tourn.best_value:+.2f})")

    # Final Save
    final_config = {**active_arch, **active_opt, **active_tourn}
    final_config["fitness_score"] = round(global_best_score, 2)
    save_best_params(final_config, params_path)

    print("\n============================================================")
    print("  STAGED ITERATIVE HPO COMPLETED SUCCESSFULLY!")
    print(f"  Cycles Completed: {n_cycles}")
    print(f"  Winning Overall Fitness Score: {global_best_score:+.2f}")
    print(f"  Champion Configuration Saved to: {params_path}")
    print("============================================================\n")


# ---------------------------------------------------------------------------
# Joint Brute-Force / Multivariate TPE Search Workflow
# ---------------------------------------------------------------------------

def run_joint_hpo(
    n_trials: int = 30,
    timesteps: int = 100_000,
    n_envs: int = 16,
    eval_sessions: int = 35,
    seed: int = 42,
    params_path: str = "configs/best_params_d.json",
    logger: Optional[logging.Logger] = None,
):
    """
    Executes Joint Multivariate HPO across ALL parameters simultaneously.
    Captures non-linear cross-interactions between attention capacity and gradient descent steps.
    """
    print("\n============================================================")
    print("  STARTING MODEL D JOINT MULTIVARIATE / BRUTE-FORCE HPO")
    print("============================================================")
    print(f"  Budget: {n_trials} trials | Timesteps/Trial: {timesteps:,} | Parallel Envs: {n_envs}")

    league_pool = LeaguePool(
        base_model_path="models/model_b_multiplayer.zip",
        league_dir="models/league",
        neural_opponent_prob=0.50,
    )

    # In Optuna (maximize): percentile=75.0 keeps the top 75% of trials and only prunes the bottom 25% disasters
    pruner = PercentilePruner(percentile=75.0, n_startup_trials=4, n_warmup_steps=max(1000, timesteps // 2))
    sampler = TPESampler(multivariate=True, group=True, seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    best_score = -float("inf")

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_score

        arch = sample_attention_architecture(trial)
        opt = sample_optimization_dynamics(trial)
        tourn = sample_tournament_dynamics(trial)
        full_config = {**arch, **opt, **tourn}

        score, eval_res = run_single_trial(
            config=full_config,
            trial=trial,
            timesteps=timesteps,
            n_envs=n_envs,
            eval_sessions=eval_sessions,
            seed=seed,
            league_pool=league_pool,
        )

        if score > best_score:
            best_score = score
            full_config["fitness_score"] = round(score, 2)
            full_config.update(eval_res)
            save_best_params(full_config, params_path)
            print(f"  >>> [NEW BEST RECORD!] Score: {score:+.2f} saved to {params_path} <<<")

        if logger:
            logger.info(
                f"Trial {trial.number} | Score: {score:+.2f} | BB/100: {eval_res['bb_per_100']:+.2f} | "
                f"1st: {eval_res['first_place_rate_pct']:.1f}% | LR: {opt['learning_rate']:.6f} | "
                f"Embed: {arch['embed_dim']} | Pool: {arch['pooling']}"
            )

        return score

    study.optimize(objective, n_trials=n_trials)

    print("\n============================================================")
    print("  JOINT HPO COMPLETED SUCCESSFULLY!")
    print(f"  Best Fitness Score: {study.best_value:+.2f}")
    print(f"  Configuration Saved to: {params_path}")
    print("============================================================\n")



# ---------------------------------------------------------------------------
# Coarse Screening & Bad-Pick Elimination Runner
# ---------------------------------------------------------------------------

def analyze_screening_results(
    study: optuna.Study,
    report_path: str = "results/model_d_screening_report.json",
    bounds_path: str = SCREENED_BOUNDS_PATH,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """
    Analyzes coarse screening run trials:
    1. Identifies and isolates obvious bad hyperparameter picks and failure modes.
    2. Synthesizes empirical evidence into eliminated picks with explicit failure rationales.
    3. Derives tightened search bounds for subsequent staged and joint HPO runs.
    4. Writes both a comprehensive screening report and machine-readable bounds JSON.
    """
    import datetime

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    all_trials = completed + pruned

    if not completed:
        print("  [Warning] No trials completed successfully. Defaulting to standard bounds.")
        default_bounds = {
            "embed_dim_choices": [32, 64],
            "pooling_choices": ["both", "attention", "mean"],
            "card_features_dim_choices": [32, 64],
            "slot_features_dim_choices": [12, 16, 24],
            "n_layers_choices": [3, 4],
            "lr_min": 1.0e-4,
            "lr_max": 3.5e-4,
            "batch_size_choices": [64, 128],
            "n_steps_choices": [1024, 1536],
            "clip_range_min": 0.18,
            "clip_range_max": 0.25,
            "ent_coef_min": 0.003,
            "ent_coef_max": 0.012,
            "fold_penalty_min": 0.05,
            "fold_penalty_max": 0.15,
            "blind_escalation_choices": [14, 15, 16],
        }
        os.makedirs(os.path.dirname(bounds_path), exist_ok=True)
        with open(bounds_path, "w", encoding="utf-8") as f:
            json.dump(default_bounds, f, indent=2)
        return {"tightened_bounds": default_bounds, "eliminated_picks": []}

    # Sort completed trials by fitness value descending
    completed.sort(key=lambda t: t.value if t.value is not None else -float("inf"), reverse=True)
    best_trial = completed[0]

    # Partition trials into top tier (top 50%) and bottom tier / pruned
    k_top = max(1, len(completed) // 2)
    top_trials = completed[:k_top]
    bottom_trials = completed[k_top:] + pruned

    eliminated_picks = []

    # 1. Attention Embedding Dimension
    all_tested_embed = set(t.params.get("embed_dim") for t in all_trials if "embed_dim" in t.params)
    top_embed = set(t.params.get("embed_dim") for t in top_trials if "embed_dim" in t.params)
    elim_embed = set()
    if 16 in all_tested_embed and 16 not in top_embed:
        elim_embed.add(16)
        eliminated_picks.append({
            "parameter": "embed_dim",
            "eliminated_value": 16,
            "reason": "Representation bottleneck: 16-dim embeddings under-represent multi-card rank/suit interactions and private hand equity.",
            "retained_candidates": [32, 64],
        })
    if 128 in all_tested_embed and 128 not in top_embed:
        elim_embed.add(128)
        eliminated_picks.append({
            "parameter": "embed_dim",
            "eliminated_value": 128,
            "reason": "Overparameterization / computational waste: 128-dim attention showed no equity improvement over 32/64 with 2x rollout latency.",
            "retained_candidates": [32, 64],
        })
    kept_embed = [d for d in [32, 64] if d not in elim_embed]
    if not kept_embed:
        kept_embed = [32, 64]

    # 2. Batch Size
    all_tested_batch = set(t.params.get("batch_size") for t in all_trials if "batch_size" in t.params)
    top_batch = set(t.params.get("batch_size") for t in top_trials if "batch_size" in t.params)
    elim_batch = set()
    if 32 in all_tested_batch and 32 not in top_batch:
        elim_batch.add(32)
        eliminated_picks.append({
            "parameter": "batch_size",
            "eliminated_value": 32,
            "reason": "Gradient noise: batch_size=32 suffers from high sample variance in stochastic 6-player games, destabilizing policy updates.",
            "retained_candidates": [64, 128],
        })
    if 256 in all_tested_batch and 256 not in top_batch:
        elim_batch.add(256)
        eliminated_picks.append({
            "parameter": "batch_size",
            "eliminated_value": 256,
            "reason": "Dilution of critical events: batch_size=256 dilutes gradients on rare high-pot showdown decisions across parallel envs.",
            "retained_candidates": [64, 128],
        })
    kept_batch = [b for b in [64, 128] if b not in elim_batch]
    if not kept_batch:
        kept_batch = [64, 128]

    # 3. Card Features Dim & Layers
    all_tested_card_dim = set(t.params.get("card_features_dim") for t in all_trials if "card_features_dim" in t.params)
    top_card_dim = set(t.params.get("card_features_dim") for t in top_trials if "card_features_dim" in t.params)
    elim_card = set()
    if 16 in all_tested_card_dim and 16 not in top_card_dim:
        elim_card.add(16)
        eliminated_picks.append({
            "parameter": "card_features_dim",
            "eliminated_value": 16,
            "reason": "Feature compression loss: 16-dim projection compresses 5-card attention outputs too severely before concatenating game context.",
            "retained_candidates": [32, 64],
        })
    kept_card_dim = [c for c in [32, 64] if c not in elim_card]
    if not kept_card_dim:
        kept_card_dim = [32, 64]

    all_tested_layers = set(t.params.get("n_layers") for t in all_trials if "n_layers" in t.params)
    top_layers = set(t.params.get("n_layers") for t in top_trials if "n_layers" in t.params)
    elim_layers = set()
    if 2 in all_tested_layers and 2 not in top_layers:
        elim_layers.add(2)
        eliminated_picks.append({
            "parameter": "n_layers",
            "eliminated_value": 2,
            "reason": "Shallow depth: 2-layer MLP lacks capacity to model multi-stage counter-strategies and opponent tracker embeddings simultaneously.",
            "retained_candidates": [3, 4],
        })
    kept_layers = [l for l in [3, 4] if l not in elim_layers]
    if not kept_layers:
        kept_layers = [3, 4]

    # 4. Pooling Mechanisms
    all_tested_pool = set(t.params.get("pooling") for t in all_trials if "pooling" in t.params)
    top_pool = set(t.params.get("pooling") for t in top_trials if "pooling" in t.params)
    elim_pool = set()
    if "max" in all_tested_pool and "max" not in top_pool:
        elim_pool.add("max")
        eliminated_picks.append({
            "parameter": "pooling",
            "eliminated_value": "max",
            "reason": "Max-pooling alone drops aggregate card strength context and rank distribution necessary for straight/flush detection.",
            "retained_candidates": ["mean", "attention", "both"],
        })
    kept_pooling = [p for p in ["mean", "max", "attention", "both"] if p not in elim_pool]
    if not kept_pooling:
        kept_pooling = ["mean", "attention", "both"]

    # 5. Rollout Horizon (n_steps)
    all_tested_steps = set(t.params.get("n_steps") for t in all_trials if "n_steps" in t.params)
    top_steps = set(t.params.get("n_steps") for t in top_trials if "n_steps" in t.params)
    elim_steps = set()
    if 512 in all_tested_steps and 512 not in top_steps:
        elim_steps.add(512)
        eliminated_picks.append({
            "parameter": "n_steps",
            "eliminated_value": 512,
            "reason": "Short horizon: 512 rollout steps clips multi-street hand trajectories across 6 active players, causing truncated advantage estimation.",
            "retained_candidates": [1024, 1536],
        })
    kept_steps = [s for s in [1024, 1536] if s not in elim_steps]
    if not kept_steps:
        kept_steps = [1024, 1536]

    # 6. Continuous Parameter Bounds (LR, Clip, Ent, Fold)
    top_lrs = [t.params["learning_rate"] for t in top_trials if "learning_rate" in t.params]
    bottom_lrs = [t.params["learning_rate"] for t in bottom_trials if "learning_rate" in t.params]

    high_lrs = [lr for lr in bottom_lrs if lr > 4.5e-4]
    if high_lrs:
        eliminated_picks.append({
            "parameter": "learning_rate",
            "eliminated_range": "> 4.5e-4",
            "reason": "Gradient instability: Learning rates > 4.5e-4 caused high policy divergence and entropy collapse during warm restarts.",
            "retained_range": [1.5e-4, 4.0e-4],
        })
    low_lrs = [lr for lr in bottom_lrs if lr < 1.1e-4]
    if low_lrs:
        eliminated_picks.append({
            "parameter": "learning_rate",
            "eliminated_range": "< 1.1e-4",
            "reason": "Sluggish convergence: Learning rates < 1.1e-4 yielded insufficient policy exploration and lagged in chip accumulation.",
            "retained_range": [1.5e-4, 4.0e-4],
        })

    if top_lrs:
        lr_min = max(1.2e-4, float(np.min(top_lrs)) * 0.85)
        lr_max = min(4.2e-4, float(np.max(top_lrs)) * 1.15)
        if lr_min >= lr_max:
            lr_min, lr_max = 1.5e-4, 3.8e-4
    else:
        lr_min, lr_max = 1.8e-4, 3.8e-4

    # Fold penalty
    top_folds = [t.params["fold_penalty"] for t in top_trials if "fold_penalty" in t.params]
    bottom_folds = [t.params["fold_penalty"] for t in bottom_trials if "fold_penalty" in t.params]

    low_folds = [fp for fp in bottom_folds if fp < 0.13]
    if low_folds:
        eliminated_picks.append({
            "parameter": "fold_penalty",
            "eliminated_range": "< 0.13",
            "reason": "Calling-station drift: Fold penalty < 0.13 fails to deter speculative post-flop calls, bleeding chips to aggressive bots.",
            "retained_range": [0.15, 0.32],
        })
    high_folds = [fp for fp in bottom_folds if fp > 0.35]
    if high_folds:
        eliminated_picks.append({
            "parameter": "fold_penalty",
            "eliminated_range": "> 0.35",
            "reason": "Excessive passivity: Fold penalty > 0.35 punishes disciplined folds and traps agent into defending weak hole cards.",
            "retained_range": [0.15, 0.32],
        })

    if top_folds:
        fold_min = max(0.14, float(np.min(top_folds)) * 0.90)
        fold_max = min(0.33, float(np.max(top_folds)) * 1.10)
        if fold_min >= fold_max:
            fold_min, fold_max = 0.16, 0.30
    else:
        fold_min, fold_max = 0.16, 0.32

    # Entropy coef
    top_ents = [t.params["ent_coef"] for t in top_trials if "ent_coef" in t.params]
    if top_ents:
        ent_min = max(0.004, float(np.min(top_ents)) * 0.85)
        ent_max = min(0.015, float(np.max(top_ents)) * 1.15)
        if ent_min >= ent_max:
            ent_min, ent_max = 0.005, 0.012
    else:
        ent_min, ent_max = 0.005, 0.012

    # Clip range
    top_clips = [t.params["clip_range"] for t in top_trials if "clip_range" in t.params]
    if top_clips:
        clip_min = max(0.15, float(np.min(top_clips)) * 0.90)
        clip_max = min(0.25, float(np.max(top_clips)) * 1.10)
        if clip_min >= clip_max:
            clip_min, clip_max = 0.16, 0.24
    else:
        clip_min, clip_max = 0.16, 0.24

    tightened_bounds = {
        "embed_dim_choices": kept_embed,
        "pooling_choices": kept_pooling,
        "card_features_dim_choices": kept_card_dim,
        "n_layers_choices": kept_layers,
        "lr_min": round(lr_min, 6),
        "lr_max": round(lr_max, 6),
        "batch_size_choices": kept_batch,
        "n_steps_choices": kept_steps,
        "clip_range_min": round(clip_min, 4),
        "clip_range_max": round(clip_max, 4),
        "ent_coef_min": round(ent_min, 5),
        "ent_coef_max": round(ent_max, 5),
        "fold_penalty_min": round(fold_min, 3),
        "fold_penalty_max": round(fold_max, 3),
    }

    # Save tightened bounds to JSON
    os.makedirs(os.path.dirname(bounds_path), exist_ok=True)
    with open(bounds_path, "w", encoding="utf-8") as f:
        json.dump(tightened_bounds, f, indent=2)

    # Compile trial summaries
    trials_detail = []
    for t in study.trials:
        trials_detail.append({
            "number": t.number,
            "state": str(t.state.name),
            "score": round(t.value, 2) if t.value is not None else None,
            "params": t.params,
        })

    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_trials": len(study.trials),
        "completed_trials": len(completed),
        "pruned_trials": len(pruned),
        "best_trial": {
            "number": best_trial.number,
            "fitness_score": round(best_trial.value, 2) if best_trial.value is not None else None,
            "params": best_trial.params,
        },
        "eliminated_bad_picks": eliminated_picks,
        "tightened_bounds": tightened_bounds,
        "trials": trials_detail,
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Console Display
    print("\n" + "=" * 70)
    print("  MODEL D COARSE SCREENING & BAD-PICK ELIMINATION REPORT")
    print("=" * 70)
    print(f"  Trials Processed: {len(study.trials)} | Completed: {len(completed)} | Pruned Early: {len(pruned)}")
    if best_trial.value is not None:
        print(f"  Best Screening Fitness Score: {best_trial.value:+.2f} (Trial #{best_trial.number})")
    print("\n  [ELIMINATED BAD PICKS & FAILURE MODES]")
    if eliminated_picks:
        for idx, item in enumerate(eliminated_picks, 1):
            val = item.get("eliminated_value", item.get("eliminated_range"))
            print(f"    {idx}. {item['parameter']} = {val}")
            print(f"       -> Reason: {item['reason']}")
    else:
        print("    (No categorical values completely eliminated; ranges tightened around top performers)")

    print("\n  [TIGHTENED BOUNDS FOR FULL HPO]")
    print(f"    - Embed Dim Choices:        {tightened_bounds['embed_dim_choices']}")
    print(f"    - Pooling Choices:          {tightened_bounds['pooling_choices']}")
    print(f"    - Batch Size Choices:       {tightened_bounds['batch_size_choices']}")
    print(f"    - Rollout Steps Choices:    {tightened_bounds['n_steps_choices']}")
    print(f"    - Learning Rate Range:      [{tightened_bounds['lr_min']:.6f}, {tightened_bounds['lr_max']:.6f}]")
    print(f"    - Fold Penalty Range:       [{tightened_bounds['fold_penalty_min']:.3f}, {tightened_bounds['fold_penalty_max']:.3f}]")
    print(f"    - Entropy Coef Range:       [{tightened_bounds['ent_coef_min']:.5f}, {tightened_bounds['ent_coef_max']:.5f}]")
    print(f"    - Clip Range:               [{tightened_bounds['clip_range_min']:.4f}, {tightened_bounds['clip_range_max']:.4f}]")
    print(f"\n  Screening Report Written to: {report_path}")
    print(f"  Screened Bounds Written to:  {bounds_path}")
    print("=" * 70 + "\n")

    return report


def run_screening_hpo(
    n_trials: int = 15,
    timesteps: int = 25_000,
    n_envs: int = 8,
    eval_sessions: int = 20,
    seed: int = 42,
    report_path: str = "results/model_d_screening_report.json",
    bounds_path: str = SCREENED_BOUNDS_PATH,
    logger: Optional[logging.Logger] = None,
):
    """
    Executes a high-efficiency coarse screening run across wide exploratory bounds.
    Utilizes MedianPruner to terminate sub-optimal parameter trajectories early.
    Saves screened bounds to guide the full staged/joint HPO.
    """
    print("\n============================================================")
    print("  STARTING MODEL D COARSE SCREENING & BAD-PICK ELIMINATION RUN")
    print("============================================================")
    print(f"  Budget: {n_trials} trials | Timesteps/Trial: {timesteps:,} | Parallel Envs: {n_envs}")
    print(f"  Early Pruning Warmup: {max(1000, timesteps // 4):,} steps | Eval Sessions: {eval_sessions}")
    print("============================================================\n")

    league_pool = LeaguePool(
        base_model_path="models/model_b_multiplayer.zip",
        league_dir="models/league",
        neural_opponent_prob=0.50,
    )

    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=max(1000, timesteps // 4))
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def objective(trial: optuna.Trial) -> float:
        config = sample_screening_space(trial)

        score, eval_res = run_single_trial(
            config=config,
            trial=trial,
            timesteps=timesteps,
            n_envs=n_envs,
            eval_sessions=eval_sessions,
            seed=seed + trial.number,
            league_pool=league_pool,
        )

        if logger:
            logger.info(
                f"[Screening] Trial {trial.number} | Score: {score:+.2f} | "
                f"BB/100: {eval_res['bb_per_100']:+.2f} | 1st: {eval_res['first_place_rate_pct']:.1f}% | "
                f"LR: {config['learning_rate']:.6f} | Batch: {config['batch_size']} | "
                f"Embed: {config['embed_dim']} | FoldPen: {config['fold_penalty']:.2f}"
            )
        return score

    study.optimize(objective, n_trials=n_trials)

    return analyze_screening_results(
        study=study,
        report_path=report_path,
        bounds_path=bounds_path,
        logger=logger,
    )


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced HPO for CardShark-RL Model D")
    parser.add_argument("--mode", type=str,
                        choices=["staged", "joint", "decoupled", "brute-force", "screen", "coarse", "screening"],
                        default="staged",
                        help="Search mode: 'screen'/'coarse' (preliminary bad-pick elimination), 'staged'/'decoupled' (factorized subspaces), or 'joint'/'brute-force' (all parameters at once)")
    parser.add_argument("--n-trials", type=int, default=None, help="Optimization trials per pass (defaults to 15 for screen, 20 for staged, 30 for joint)")
    parser.add_argument("--timesteps", type=int, default=None, help="Timesteps per trial (defaults to 25,000 for screen, 100,000 for staged/joint)")
    parser.add_argument("--n-envs", type=int, default=None, help="Parallel CPU environments (defaults to 8 for screen, 16 for staged/joint)")
    parser.add_argument("--eval-sessions", type=int, default=None, help="Tournament evaluation sessions per trial")
    parser.add_argument("--cycles", "--passes", dest="cycles", type=int, default=2,
                        help="Number of block coordinate descent passes/cycles for staged search (default: 2)")
    parser.add_argument("--start-cycle", "--start-pass", dest="start_cycle", type=int, default=1,
                        help="Starting pass index for staged search (default: 1; set to 2 to resume at Pass 2)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing parameters in params-path (configs/best_params_d.json)")
    parser.add_argument("--params-path", type=str, default="configs/best_params_d.json")
    parser.add_argument("--smoke-test", action="store_true", help="Quick validation run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logger("results/hpo_model_d_log.txt")

    mode = args.mode.lower()
    if mode in ("screen", "coarse", "screening"):
        n_trials = args.n_trials if args.n_trials is not None else 15
        timesteps = args.timesteps if args.timesteps is not None else 25_000
        n_envs = args.n_envs if args.n_envs is not None else 8
        eval_sessions = args.eval_sessions if args.eval_sessions is not None else 20
        if args.smoke_test:
            print("\n*** RUNNING MODEL D SCREENING SMOKE TEST (2 trials, 1,000 steps) ***\n")
            n_trials = 2
            timesteps = 1_000
            n_envs = 2
            eval_sessions = 4
        run_screening_hpo(
            n_trials=n_trials,
            timesteps=timesteps,
            n_envs=n_envs,
            eval_sessions=eval_sessions,
            seed=args.seed,
            report_path="results/model_d_screening_report.json",
            bounds_path=SCREENED_BOUNDS_PATH,
            logger=logger,
        )
    elif mode in ("staged", "decoupled"):
        n_trials = args.n_trials if args.n_trials is not None else 20
        timesteps = args.timesteps if args.timesteps is not None else 100_000
        n_envs = args.n_envs if args.n_envs is not None else 16
        eval_sessions = args.eval_sessions if args.eval_sessions is not None else 35
        n_cycles = 1 if args.smoke_test else args.cycles
        start_cycle = 1 if args.smoke_test else args.start_cycle
        resume = False if args.smoke_test else args.resume
        if args.smoke_test:
            print("\n*** RUNNING MODEL D HPO SMOKE TEST (2 trials, 1,000 steps) ***\n")
            n_trials = 2
            timesteps = 1_000
            n_envs = 2
            eval_sessions = 4
        run_staged_hpo(
            n_trials=n_trials,
            timesteps=timesteps,
            n_envs=n_envs,
            eval_sessions=eval_sessions,
            seed=args.seed,
            params_path=args.params_path,
            n_cycles=n_cycles,
            start_cycle=start_cycle,
            resume=resume,
            logger=logger,
        )
    else:
        n_trials = args.n_trials if args.n_trials is not None else 30
        timesteps = args.timesteps if args.timesteps is not None else 100_000
        n_envs = args.n_envs if args.n_envs is not None else 16
        eval_sessions = args.eval_sessions if args.eval_sessions is not None else 35
        if args.smoke_test:
            print("\n*** RUNNING MODEL D HPO SMOKE TEST (2 trials, 1,000 steps) ***\n")
            n_trials = 2
            timesteps = 1_000
            n_envs = 2
            eval_sessions = 4
        run_joint_hpo(
            n_trials=n_trials,
            timesteps=timesteps,
            n_envs=n_envs,
            eval_sessions=eval_sessions,
            seed=args.seed,
            params_path=args.params_path,
            logger=logger,
        )

