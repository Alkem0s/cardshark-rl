"""
scripts/holdem/train_holdem.py — Training Pipeline for No-Limit Texas Hold'em (Model Holdem-A).

Key features:
1. Pure 6-action discrete betting abstraction.
2. 73-dimensional bounded [-1.0, 1.0] observation vector.
3. Checkpoint outputs strictly isolated to models/holdem/.
"""

from __future__ import annotations
import os
import math
import time
import shutil
import json
import argparse
from typing import Callable, Optional, List, Dict, Any

import numpy as np
import torch
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from games.holdem.gym_wrapper import HoldemPokerGymEnv, OBS_DIM_HOLDEM


def cosine_warm_restart_schedule(
    initial_lr: float,
    min_lr: float = 2.0e-5,
    n_cycles: int = 7,
    warmup_fraction: float = 0.05,
) -> Callable[[float], float]:
    def schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        cycle_len = 1.0 / max(1, n_cycles)
        current_cycle = min(int(progress / cycle_len), n_cycles - 1)
        cycle_progress = (progress - current_cycle * cycle_len) / cycle_len

        if cycle_progress < warmup_fraction:
            alpha = cycle_progress / max(1e-6, warmup_fraction)
            return min_lr + alpha * (initial_lr - min_lr)
        else:
            decay_progress = (cycle_progress - warmup_fraction) / max(1e-6, 1.0 - warmup_fraction)
            return min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + math.cos(math.pi * decay_progress))

    return schedule


class EntropyDecayCallback(BaseCallback):
    def __init__(
        self,
        initial_ent: float = 0.015,
        min_ent: float = 0.001,
        total_timesteps: int = 1_000_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.initial_ent = initial_ent
        self.min_ent = min_ent
        self.total_timesteps = max(1, total_timesteps)

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.total_timesteps)
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        self.model.ent_coef = float(self.min_ent + (self.initial_ent - self.min_ent) * decay)
        return True


class HoldemSessionCallback(BaseCallback):
    def __init__(
        self,
        log_freq: int = 5_000,
        window: int = 100,
        save_best_path: Optional[str] = "models/holdem/model_holdem_a.zip",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.window = window
        self.save_best_path = save_best_path
        self.best_win_rate = -1.0
        self.winner_history: List[int] = []
        self.total_sessions = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if done and i < len(infos):
                info = infos[i]
                if "is_winner" in info:
                    is_winner = 1 if info.get("is_winner", False) else 0
                    self.total_sessions += 1
                    self.winner_history.append(is_winner)
                    if len(self.winner_history) > self.window:
                        self.winner_history.pop(0)

        if self.num_timesteps % self.log_freq == 0 and len(self.winner_history) > 0:
            rolling_win = np.mean(self.winner_history) * 100.0
            if self.verbose > 0:
                print(
                    f"[{self.num_timesteps:,} steps] "
                    f"Hold'em Sessions: {self.total_sessions:,} | "
                    f"Win Rate (1st Place): {rolling_win:.1f}%"
                )

            if self.save_best_path and len(self.winner_history) >= min(self.window, 20) and rolling_win > self.best_win_rate:
                self.best_win_rate = rolling_win
                self.model.save(self.save_best_path)
                if self.verbose > 0:
                    print(f"      >>> [NEW BEST MODEL HOLDEM-A] Saved checkpoint: {self.save_best_path} ({rolling_win:.1f}% win rate) <<<")

        return True


def make_holdem_env(rank: int, seed: int = 42, fold_penalty: float = 0.02):
    def _init():
        e = HoldemPokerGymEnv(
            num_seats=5,
            starting_chips=200,
            small_blind=1,
            big_blind=2,
            hero_seat=0,
            max_hands_per_session=150,
            blind_escalation_interval=15,
            randomize_stacks=True,
            fold_penalty=fold_penalty,
            rng_seed=seed + rank * 1000,
        )
        return ActionMasker(e, lambda env: env.action_masks())
    return _init


def train_holdem(
    timesteps: int = 1_000_000,
    n_envs: int = 8,
    save_dir: str = "models/holdem",
    model_name: str = "model_holdem_a.zip",
    smoke_test: bool = False,
    seed: int = 42,
    params_path: str = "configs/holdem/best_params_holdem_a.json",
):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)

    # Defaults
    learning_rate = 2.0e-4
    min_lr = 2.0e-5
    n_restart_cycles = 7
    initial_ent_coef = 0.015
    min_ent_coef = 0.001
    n_steps = 1024
    batch_size = 128
    n_epochs = 7
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.20
    fold_penalty = 0.02
    net_arch = [128, 128]

    resolved_path = os.path.join(PROJECT_ROOT, params_path) if not os.path.isabs(params_path) else params_path
    if os.path.exists(resolved_path):
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            learning_rate = params.get("learning_rate", learning_rate)
            if "min_lr" in params:
                min_lr = params["min_lr"]
            elif "min_lr_ratio" in params:
                min_lr = learning_rate * params["min_lr_ratio"]
            n_restart_cycles = params.get("n_restart_cycles", n_restart_cycles)
            initial_ent_coef = params.get("initial_ent_coef", initial_ent_coef)
            min_ent_coef = params.get("min_ent_coef", min_ent_coef)
            n_steps = params.get("n_steps", n_steps)
            batch_size = params.get("batch_size", batch_size)
            n_epochs = params.get("n_epochs", n_epochs)
            gamma = params.get("gamma", gamma)
            gae_lambda = params.get("gae_lambda", gae_lambda)
            clip_range = params.get("clip_range", clip_range)
            fold_penalty = params.get("fold_penalty", fold_penalty)
            net_arch = params.get("net_arch", net_arch)
            print(f"  [Config] Loaded Hold'em params from: {resolved_path}")
        except Exception as e:
            print(f"  [Config] Warning: Could not parse {resolved_path} ({e}), using defaults.")

    if smoke_test:
        timesteps = 500
        n_envs = 2
        effective_n_steps = 64
        batch_size = 32
        n_epochs = 2
        print("\n*** RUNNING HOLDEM IN SMOKE-TEST MODE (500 steps) ***\n")
    else:
        effective_n_steps = n_steps
        print(f"=== STARTING NO-LIMIT TEXAS HOLD'EM TRAINING ({timesteps:,} steps) ===")
        print(f"  Observation Space: {OBS_DIM_HOLDEM} dims")
        print(f"  Architecture: MaskableActorCriticPolicy with net_arch={net_arch}")
        print(f"  Environments: {n_envs} | Steps/Env: {effective_n_steps}")

    env = DummyVecEnv([make_holdem_env(i, seed=seed, fold_penalty=fold_penalty) for i in range(n_envs)])

    lr_schedule = cosine_warm_restart_schedule(
        initial_lr=learning_rate,
        min_lr=min_lr,
        n_cycles=n_restart_cycles,
    )

    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, vf=net_arch),
    )

    model = MaskablePPO(
        policy=MaskableActorCriticPolicy,
        env=env,
        learning_rate=lr_schedule,
        n_steps=effective_n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=initial_ent_coef,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=1,
    )

    callbacks = CallbackList([
        EntropyDecayCallback(
            initial_ent=initial_ent_coef,
            min_ent=min_ent_coef,
            total_timesteps=timesteps,
        ),
        HoldemSessionCallback(
            log_freq=100 if smoke_test else 5_000,
            save_best_path=save_path,
        ),
    ])

    start_time = time.time()
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)
    elapsed = time.time() - start_time

    model.save(save_path)
    print(f"\n[Training Complete] Elapsed time: {elapsed / 60:.1f} mins | Saved to: {save_path}")

    env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Texas Hold'em RL Agent")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--config", type=str, default="configs/holdem/best_params_holdem_a.json")
    args = parser.parse_args()

    train_holdem(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        smoke_test=args.smoke_test,
        params_path=args.config,
    )
