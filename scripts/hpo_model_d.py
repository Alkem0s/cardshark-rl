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
from optuna.pruners import MedianPruner
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
        eval_interval=max(10_000, timesteps // 4),
        eval_sessions=10,
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

def sample_attention_architecture(trial: optuna.Trial) -> dict:
    """Subspace A: Inductive Bias & Token Geometry (Orthogonal to PPO Dynamics)."""
    embed_dim = trial.suggest_categorical("embed_dim", [32, 64])
    num_heads = trial.suggest_categorical("num_heads", [2, 4])
    if embed_dim % num_heads != 0:
        num_heads = 2

    pooling = trial.suggest_categorical("pooling", ["mean", "max", "attention", "both"])
    card_features_dim = trial.suggest_categorical("card_features_dim", [32, 64])
    n_layers = trial.suggest_categorical("n_layers", [3, 4])
    net_arch = [256] * n_layers

    return {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "pooling": pooling,
        "card_features_dim": card_features_dim,
        "net_arch": net_arch,
    }


def sample_optimization_dynamics(trial: optuna.Trial) -> dict:
    """Subspace B: Coupled Stochastic Gradient Descent Dynamics."""
    learning_rate = trial.suggest_float("learning_rate", 1.8e-4, 3.8e-4, log=True)
    min_lr_ratio = trial.suggest_float("min_lr_ratio", 0.05, 0.20)
    min_lr = learning_rate * min_lr_ratio
    n_restart_cycles = trial.suggest_categorical("n_restart_cycles", [3, 4, 5])

    n_steps = trial.suggest_categorical("n_steps", [1024, 1536])
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    n_epochs = trial.suggest_int("n_epochs", 6, 9)

    gamma = trial.suggest_float("gamma", 0.990, 0.997)
    gae_lambda = trial.suggest_float("gae_lambda", 0.91, 0.95)
    clip_range = trial.suggest_float("clip_range", 0.16, 0.24)
    ent_coef = trial.suggest_float("ent_coef", 0.005, 0.012, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.45, 0.60)
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


def sample_tournament_dynamics(trial: optuna.Trial) -> dict:
    """Subspace C: MDP & Tournament Reward Dynamics."""
    fold_penalty = trial.suggest_float("fold_penalty", 0.16, 0.32)
    blind_escalation = trial.suggest_categorical("blind_escalation", [14, 15, 16])
    return {
        "fold_penalty": fold_penalty,
        "blind_escalation": blind_escalation,
    }


# ---------------------------------------------------------------------------
# Default Baseline Configurations for Decoupled Searching
# ---------------------------------------------------------------------------

DEFAULT_ARCHITECTURE = {
    "embed_dim": 32,
    "num_heads": 2,
    "pooling": "mean",
    "card_features_dim": 32,
    "net_arch": [256, 256, 256],
}

DEFAULT_OPTIMIZATION = {
    "learning_rate": 2.5e-4,
    "min_lr": 2.5e-5,
    "n_restart_cycles": 5,
    "n_steps": 1536,
    "batch_size": 64,
    "n_epochs": 7,
    "gamma": 0.9958,
    "gae_lambda": 0.9227,
    "clip_range": 0.1851,
    "ent_coef": 0.00764,
    "vf_coef": 0.5316,
    "max_grad_norm": 0.5723,
}

DEFAULT_TOURNAMENT = {
    "fold_penalty": 0.215,
    "blind_escalation": 15,
}


# ---------------------------------------------------------------------------
# Staged / Decoupled Search Workflow
# ---------------------------------------------------------------------------

def run_staged_hpo(
    n_trials: int = 24,
    timesteps: int = 100_000,
    n_envs: int = 16,
    eval_sessions: int = 35,
    seed: int = 42,
    params_path: str = "configs/best_params_d.json",
    logger: Optional[logging.Logger] = None,
):
    """
    Executes Decoupled / Staged HPO across 3 orthogonal subspaces:
    - Stage 1: Card Attention Feature Extractor Architecture (Trials: ~25%)
    - Stage 2: Coupled Gradient Descent Dynamics (Trials: ~60%)
    - Stage 3: Tournament & Reward Shaping (Trials: ~15%)
    """
    print("\n============================================================")
    print("  STARTING MODEL D STAGED / DECOUPLED HPO PIPELINE")
    print("============================================================")
    print(f"  Total Budget: {n_trials} trials | Timesteps/Trial: {timesteps:,} | Parallel Envs: {n_envs}")

    league_pool = LeaguePool(
        base_model_path="models/model_b_multiplayer.zip",
        league_dir="models/league",
        neural_opponent_prob=0.50,
    )

    trials_stage1 = max(4, int(n_trials * 0.25))
    trials_stage2 = max(6, int(n_trials * 0.60))
    trials_stage3 = max(2, n_trials - trials_stage1 - trials_stage2)

    active_arch = dict(DEFAULT_ARCHITECTURE)
    active_opt = dict(DEFAULT_OPTIMIZATION)
    active_tourn = dict(DEFAULT_TOURNAMENT)

    global_best_score = -float("inf")
    global_best_config = {}

    # --- STAGE 1: Representation & Attention Architecture ---
    print(f"\n>>> [STAGE 1/3] Optimizing Card Attention Architecture ({trials_stage1} trials) <<<")
    study_arch = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))

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
            seed=seed,
            league_pool=league_pool,
        )

        if score > global_best_score:
            global_best_score = score
            global_best_config = dict(full_config)
            global_best_config["fitness_score"] = round(score, 2)
            global_best_config.update(eval_res)
            save_best_params(global_best_config, params_path)

        if logger:
            logger.info(f"[Stage 1] Trial {trial.number} | Score: {score:+.2f} | Arch: {arch_config}")
        return score

    study_arch.optimize(objective_stage1, n_trials=trials_stage1)
    best_arch_trial = study_arch.best_trial
    active_arch["embed_dim"] = best_arch_trial.params["embed_dim"]
    active_arch["num_heads"] = best_arch_trial.params["num_heads"]
    active_arch["pooling"] = best_arch_trial.params["pooling"]
    active_arch["card_features_dim"] = best_arch_trial.params["card_features_dim"]
    active_arch["net_arch"] = [256] * best_arch_trial.params["n_layers"]
    print(f"  >>> Stage 1 Complete! Best Architecture: {active_arch} (Score: {study_arch.best_value:+.2f})")

    # --- STAGE 2: Coupled Gradient Descent Dynamics ---
    print(f"\n>>> [STAGE 2/3] Optimizing Coupled Gradient Descent Dynamics ({trials_stage2} trials) <<<")
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=timesteps // 4)
    study_opt = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(multivariate=True, group=True, seed=seed + 1),
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
            seed=seed + 100,
            league_pool=league_pool,
        )

        if score > global_best_score:
            global_best_score = score
            global_best_config = dict(full_config)
            global_best_config["fitness_score"] = round(score, 2)
            global_best_config.update(eval_res)
            save_best_params(global_best_config, params_path)

        if logger:
            logger.info(f"[Stage 2] Trial {trial.number} | Score: {score:+.2f} | LR: {opt_config['learning_rate']:.6f}")
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
    print(f"  >>> Stage 2 Complete! Best Optimizer: LR={active_opt['learning_rate']:.6f}, Ent={active_opt['ent_coef']:.5f}")

    # --- STAGE 3: Tournament & Reward Shaping Dynamics ---
    print(f"\n>>> [STAGE 3/3] Fine-Tuning Tournament Dynamics ({trials_stage3} trials) <<<")
    study_tourn = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed + 2))

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
            seed=seed + 200,
            league_pool=league_pool,
        )

        if score > global_best_score:
            global_best_score = score
            global_best_config = dict(full_config)
            global_best_config["fitness_score"] = round(score, 2)
            global_best_config.update(eval_res)
            save_best_params(global_best_config, params_path)

        if logger:
            logger.info(f"[Stage 3] Trial {trial.number} | Score: {score:+.2f} | FoldPenalty: {tourn_config['fold_penalty']:.2f}")
        return score

    study_tourn.optimize(objective_stage3, n_trials=trials_stage3)
    best_tourn_trial = study_tourn.best_trial
    active_tourn["fold_penalty"] = best_tourn_trial.params["fold_penalty"]
    active_tourn["blind_escalation"] = best_tourn_trial.params["blind_escalation"]

    # Final Save
    final_config = {**active_arch, **active_opt, **active_tourn}
    final_config["fitness_score"] = round(global_best_score, 2)
    save_best_params(final_config, params_path)

    print("\n============================================================")
    print("  STAGED HPO COMPLETED SUCCESSFULLY!")
    print(f"  Winning Overall Fitness Score: {global_best_score:+.2f}")
    print(f"  Configuration Saved to: {params_path}")
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

    pruner = MedianPruner(n_startup_trials=4, n_warmup_steps=timesteps // 4)
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
# CLI Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced HPO for CardShark-RL Model D")
    parser.add_argument("--mode", type=str, choices=["staged", "joint", "decoupled", "brute-force"], default="staged",
                        help="Search mode: 'staged'/'decoupled' (factorized subspaces) or 'joint'/'brute-force' (all parameters at once)")
    parser.add_argument("--n-trials", type=int, default=24, help="Total optimization trials")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Timesteps per trial")
    parser.add_argument("--n-envs", type=int, default=16, help="Parallel CPU environments")
    parser.add_argument("--eval-sessions", type=int, default=35, help="Tournament evaluation sessions per trial")
    parser.add_argument("--params-path", type=str, default="configs/best_params_d.json")
    parser.add_argument("--smoke-test", action="store_true", help="Quick 2-trial validation run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logger("results/hpo_model_d_log.txt")

    if args.smoke_test:
        print("\n*** RUNNING MODEL D HPO SMOKE TEST (2 trials, 1,000 steps) ***\n")
        args.n_trials = 2
        args.timesteps = 1_000
        args.n_envs = 2
        args.eval_sessions = 4

    mode = args.mode.lower()
    if mode in ("staged", "decoupled"):
        run_staged_hpo(
            n_trials=args.n_trials,
            timesteps=args.timesteps,
            n_envs=args.n_envs,
            eval_sessions=args.eval_sessions,
            seed=args.seed,
            params_path=args.params_path,
            logger=logger,
        )
    else:
        run_joint_hpo(
            n_trials=args.n_trials,
            timesteps=args.timesteps,
            n_envs=args.n_envs,
            eval_sessions=args.eval_sessions,
            seed=args.seed,
            params_path=args.params_path,
            logger=logger,
        )
