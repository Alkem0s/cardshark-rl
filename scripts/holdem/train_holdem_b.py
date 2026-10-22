"""
scripts/holdem/train_holdem_b.py — Training Pipeline for Model Holdem-B Crusher.

Features:
1. 125-dimensional observation space:
   - 7 cards (2 hole + 5 board) processed via 7-card CardAttentionExtractor
   - 17-feature Board Texture & Draws Engine (monotone, two-tone, paired, gutshots, OESD, flush draws, overcards, equity)
   - 32-feature Dual-Timescale Bayesian Opponent Profiler (VPIP, PFR, AF, Fold-to-CBet, Tilt)
   - 3-feature ICM & Stack Geometry (share, effective stack, short stack ratio)
2. Permutation-invariant 7-token Multi-Head Self-Attention.
3. Cosine warm-restart learning rate schedule + Cosine entropy annealing.
4. Checkpoint outputs isolated to models/holdem/model_holdem_b.zip.
"""

from __future__ import annotations
import os
import sys

# Prevent OpenMP / Intel MKL library collision under SubprocVecEnv
os.environ["MKL_THREADING_LAYER"] = "GNU"

import math
import time
import json
import argparse
import shutil
from typing import Callable, Optional, List, Dict, Any

import numpy as np
import torch
torch.set_num_threads(4)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from core.neural.card_attention import CardAttentionExtractor
import core.neural.card_attention
sys.modules["card_attention"] = core.neural.card_attention

from games.holdem.gym_wrapper import HoldemPokerGymEnv, OBS_DIM_HOLDEM_B


def pin_to_pcores(n_cores: int = 12):
    """Pins execution and worker processes to P-cores (cores 0..15 on Intel 13th Gen)."""
    try:
        if hasattr(os, "sched_setaffinity"):
            target_cores = set(range(min(n_cores, 16)))
            os.sched_setaffinity(0, target_cores)
            print(f"  [CPU Affinity] Pinned process & workers to P-cores: {sorted(list(target_cores))}")
    except Exception as e:
        print(f"  [CPU Affinity] Notice: Could not set CPU affinity ({e})")



def cosine_warm_restart_schedule(
    initial_lr: float,
    min_lr: float = 2.0e-5,
    n_cycles: int = 5,
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
        initial_ent: float = 0.025,
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


class HoldemBSessionCallback(BaseCallback):
    def __init__(
        self,
        log_freq: int = 5_000,
        window: int = 100,
        save_best_path: Optional[str] = "models/holdem/model_holdem_b_best.zip",
        primary_save_path: Optional[str] = "models/holdem/model_holdem_b.zip",
        stats_log_path: Optional[str] = "logs/holdem/win_rate_history_b.json",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.window = window
        self.save_best_path = save_best_path
        self.primary_save_path = primary_save_path
        self.stats_log_path = stats_log_path
        self.best_win_rate = -1.0
        self.winner_history: List[int] = []
        self.total_sessions = 0
        self.trajectory: List[dict] = []

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
            rolling_win = float(np.mean(self.winner_history) * 100.0)
            self.trajectory.append({
                "timestep": int(self.num_timesteps),
                "total_sessions": int(self.total_sessions),
                "win_rate": round(rolling_win, 2),
            })

            if self.stats_log_path:
                try:
                    os.makedirs(os.path.dirname(self.stats_log_path), exist_ok=True)
                    with open(self.stats_log_path, "w", encoding="utf-8") as f:
                        json.dump(self.trajectory, f, indent=2)
                except Exception as e:
                    if self.verbose > 1:
                        print(f"      [Warning] Failed to write trajectory log: {e}")

            if self.verbose > 0:
                print(
                    f"[{self.num_timesteps:,} steps] "
                    f"Hold'em Sessions: {self.total_sessions:,} | "
                    f"Win Rate (1st Place): {rolling_win:.1f}%"
                )

            if len(self.winner_history) >= min(self.window, 20) and rolling_win > self.best_win_rate:
                self.best_win_rate = rolling_win
                if self.save_best_path:
                    os.makedirs(os.path.dirname(self.save_best_path), exist_ok=True)
                    self.model.save(self.save_best_path)
                if self.primary_save_path:
                    os.makedirs(os.path.dirname(self.primary_save_path), exist_ok=True)
                    self.model.save(self.primary_save_path)
                if self.verbose > 0:
                    print(f"      >>> [NEW BEST MODEL HOLDEM-B] Saved checkpoint: ({rolling_win:.1f}% win rate) <<<")

        return True


def make_holdem_b_env(
    rank: int,
    seed: int = 42,
    fold_penalty: float = 0.0,
    elimination_penalty: float = 0.0,
    win_bonus: float = 0.0,
):
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
            elimination_penalty=elimination_penalty,
            win_bonus=win_bonus,
            dominated_masking=True,
            obs_version="b",
            rng_seed=seed + rank * 1000,
        )
        return ActionMasker(e, lambda env: env.action_masks())
    return _init


def train_holdem_b(
    timesteps: int = 1_000_000,
    n_envs: int = 12,
    save_dir: str = "models/holdem",
    model_name: str = "model_holdem_b.zip",
    smoke_test: bool = False,
    seed: int = 42,
    params_path: str = "configs/holdem/best_params_holdem_b.json",
    resume_path: Optional[str] = None,
    pin_cores: bool = True,
):
    if pin_cores and not smoke_test:
        pin_to_pcores(n_envs)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)
    best_save_path = os.path.join(save_dir, "model_holdem_b_best.zip")
    final_save_path = os.path.join(save_dir, "model_holdem_b_final.zip")
    stats_path = os.path.join("logs", "holdem", "win_rate_history_b.json")

    # Defaults
    learning_rate = 2.5e-4
    min_lr = 5.0e-5
    n_restart_cycles = 5
    initial_ent_coef = 0.025
    min_ent_coef = 0.001
    n_steps = 1024
    batch_size = 128
    n_epochs = 7
    gamma = 0.992
    gae_lambda = 0.93
    clip_range = 0.20
    fold_penalty = 0.025
    embed_dim = 32
    num_heads = 4
    pooling = "mean"
    card_features_dim = 32
    game_features_dim = 64
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
            elimination_penalty = params.get("elimination_penalty", 0.0)
            win_bonus = params.get("win_bonus", 0.0)
            embed_dim = params.get("embed_dim", embed_dim)
            num_heads = params.get("num_heads", num_heads)
            pooling = params.get("pooling", pooling)
            card_features_dim = params.get("card_features_dim", card_features_dim)
            game_features_dim = params.get("game_features_dim", game_features_dim)
            net_arch = params.get("net_arch", net_arch)
            print(f"  [Config] Loaded Hold'em Model B params from: {resolved_path}")
        except Exception as e:
            print(f"  [Config] Warning: Could not parse {resolved_path} ({e}), using defaults.")
    else:
        elimination_penalty = 0.0
        win_bonus = 0.0

    if smoke_test:
        timesteps = 500
        n_envs = 2
        effective_n_steps = 64
        batch_size = 32
        n_epochs = 2
        vec_env_cls = DummyVecEnv
        print("\n*** RUNNING HOLDEM MODEL B IN SMOKE-TEST MODE (500 steps) ***\n")
    else:
        effective_n_steps = n_steps
        vec_env_cls = SubprocVecEnv
        print(f"=== STARTING NO-LIMIT TEXAS HOLD'EM MODEL B TRAINING ({timesteps:,} steps) ===")
        print(f"  Observation Space: {OBS_DIM_HOLDEM_B} dims (Model B)")
        print(f"  Backbone: CardAttentionExtractor(7 cards, dim={embed_dim}, heads={num_heads}, pool={pooling})")
        print(f"  Policy MLP: net_arch={net_arch}")
        print(f"  Environments: {n_envs} (SubprocVecEnv) | Steps/Env: {effective_n_steps}")

    env = vec_env_cls([
        make_holdem_b_env(
            i,
            seed=seed,
            fold_penalty=fold_penalty,
            elimination_penalty=elimination_penalty,
            win_bonus=win_bonus,
        ) for i in range(n_envs)
    ])

    lr_schedule = cosine_warm_restart_schedule(
        initial_lr=learning_rate,
        min_lr=min_lr,
        n_cycles=n_restart_cycles,
    )

    policy_kwargs = dict(
        features_extractor_class=CardAttentionExtractor,
        features_extractor_kwargs=dict(
            num_cards=7,
            embed_dim=embed_dim,
            num_heads=num_heads,
            pooling=pooling,
            card_features_dim=card_features_dim,
            game_features_dim=game_features_dim,
            slot_features_dim=0,
        ),
        net_arch=dict(pi=net_arch, vf=net_arch),
    )

    if resume_path and os.path.exists(resume_path):
        print(f"  [Resume] Loading existing checkpoint from: {resume_path}")
        model = MaskablePPO.load(
            resume_path,
            env=env,
            learning_rate=lr_schedule,
            custom_objects={"learning_rate": lr_schedule},
        )
    else:
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

    session_callback = HoldemBSessionCallback(
        log_freq=100 if smoke_test else 5_000,
        save_best_path=best_save_path,
        primary_save_path=save_path,
        stats_log_path=stats_path,
    )

    callbacks = CallbackList([
        EntropyDecayCallback(
            initial_ent=initial_ent_coef,
            min_ent=min_ent_coef,
            total_timesteps=timesteps,
        ),
        session_callback,
    ])

    start_time = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        progress_bar=False,
        reset_num_timesteps=False if (resume_path and os.path.exists(resume_path)) else True,
    )
    elapsed = time.time() - start_time

    # Save final model
    model.save(final_save_path)
    # Ensure primary save path is present (copying best or final)
    if os.path.exists(best_save_path) and session_callback.best_win_rate > 0:
        shutil.copyfile(best_save_path, save_path)
        print(f"  [Primary] Updated {save_path} with best checkpoint ({session_callback.best_win_rate:.1f}% win rate)")
    else:
        model.save(save_path)

    print(f"\n[Training Complete] Elapsed time: {elapsed / 60:.1f} mins | Saved to: {save_path}")

    env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Texas Hold'em Model B Crusher")
    parser.add_argument("--timesteps", type=int, default=2_500_000)
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--no-pin", action="store_true", help="Disable CPU pinning")
    parser.add_argument("--config", type=str, default="configs/holdem/best_params_holdem_b.json")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    train_holdem_b(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        smoke_test=args.smoke_test,
        params_path=args.config,
        resume_path=args.resume,
        pin_cores=not args.no_pin,
    )

