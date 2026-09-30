"""
train_model_d.py — Training Pipeline for CardShark-RL Model D (Next-Generation Champion).

Key Architectural Innovations over Model C:
1. Permutation-Invariant Card Transformer (CardAttentionExtractor):
   Extracts the 5 private cards as an unordered set of tokens via Multi-Head Self-Attention
   and invariant pooling, eliminating the permutation memorization bottleneck.
2. Cosine Annealing with Warm Restarts (SGDR):
   Synchronized learning rate cycling that warms up when new sparring champions are introduced
   to the League pool, preventing late-stage policy freezing.
3. Multi-Agent Co-Evolutionary League Sparring:
   Fictitious play against frozen Model B, historical Model C checkpoints, rolling Model D
   snapshots, and the active AdversarialExploiter archetype.
4. Scale-Invariant 87-Dimensional Observation Vector:
   Full intra-hand sequence trajectories and dual-timescale Bayesian opponent profiling.
"""

from __future__ import annotations
import os
import math
import json
import argparse
from typing import Callable, Optional, List, Dict, Any

import numpy as np
import torch

import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.multi_gym_wrapper import (
    make_superhuman_multi_env,
    multi_mask_fn,
    SUPERHUMAN_OBS_DIM,
)
from rl.league import LeaguePool
from rl.card_attention import CardAttentionExtractor


# ---------------------------------------------------------------------------
# Cosine Annealing with Warm Restarts (SGDR)
# ---------------------------------------------------------------------------

def cosine_warm_restart_schedule(
    initial_lr: float,
    min_lr: float = 2.0e-5,
    n_cycles: int = 5,
    warmup_fraction: float = 0.05,
) -> Callable[[float], float]:
    """
    Cosine Annealing with Warm Restarts (SGDR) for Stable-Baselines3.
    progress_remaining decays from 1.0 (start) down to 0.0 (end).
    """
    def schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining  # 0.0 -> 1.0
        cycle_len = 1.0 / max(1, n_cycles)
        current_cycle = min(int(progress / cycle_len), n_cycles - 1)
        cycle_progress = (progress - current_cycle * cycle_len) / cycle_len  # 0.0 -> 1.0

        if cycle_progress < warmup_fraction:
            # Linear warmup from min_lr to initial_lr
            alpha = cycle_progress / max(1e-6, warmup_fraction)
            return min_lr + alpha * (initial_lr - min_lr)
        else:
            # Cosine decay from initial_lr down to min_lr
            decay_progress = (cycle_progress - warmup_fraction) / max(1e-6, 1.0 - warmup_fraction)
            return min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + math.cos(math.pi * decay_progress))

    return schedule


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class ModelDLeagueSnapshotCallback(BaseCallback):
    """Saves rolling Model D checkpoints at regular intervals and adds them to the sparring pool."""

    def __init__(
        self,
        league_pool: LeaguePool,
        snapshot_interval: int = 250_000,
        save_dir: str = "models/league",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.league_pool = league_pool
        self.snapshot_interval = snapshot_interval
        self.save_dir = save_dir
        self.last_snapshot_step = 0
        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_snapshot_step >= self.snapshot_interval:
            self.last_snapshot_step = self.num_timesteps
            snapshot_name = f"model_d_step_{self.num_timesteps}.zip"
            snapshot_path = os.path.join(self.save_dir, snapshot_name)
            self.model.save(snapshot_path)
            self.league_pool.add_snapshot(snapshot_path)
            if self.verbose > 0:
                print(f"\n  >>> [MODEL D SNAPSHOT] Saved & registered sparring agent: {snapshot_path} <<<")
                print(f"      Active League Pool Size: {len(self.league_pool.checkpoints)} agents\n")
        return True


class ModelDSessionCallback(BaseCallback):
    """Logs rolling tournament metrics: survival rate, 1st-place rate, hands survived, and returns."""

    def __init__(
        self,
        league_pool: LeaguePool,
        log_freq: int = 5_000,
        window: int = 100,
        is_freezeout: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.league_pool = league_pool
        self.log_freq = log_freq
        self.window = window
        self.is_freezeout = is_freezeout

        self.session_returns: list[float] = []
        self.survived_history: list[int] = []
        self.winner_history: list[int] = []
        self.hands_history: list[int] = []
        self.total_sessions = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if done and i < len(infos):
                info = infos[i]
                hero_chips = info.get("hero_chips", 0)
                hands = info.get("hands_played", 1)
                is_winner = 1 if info.get("is_winner", False) else 0

                self.total_sessions += 1
                survived = 1 if hero_chips > 0 else 0
                self.survived_history.append(survived)
                self.winner_history.append(is_winner)
                self.hands_history.append(hands)

                if len(self.survived_history) > self.window:
                    self.survived_history.pop(0)
                    self.winner_history.pop(0)
                    self.hands_history.pop(0)

                ep_info = info.get("episode")
                if ep_info:
                    self.session_returns.append(ep_info["r"])
                    if len(self.session_returns) > self.window:
                        self.session_returns.pop(0)

        if self.num_timesteps % self.log_freq == 0 and len(self.survived_history) > 0:
            rolling_survival = np.mean(self.survived_history) * 100.0
            rolling_win = np.mean(self.winner_history) * 100.0
            rolling_hands = np.mean(self.hands_history)
            avg_return = np.mean(self.session_returns) if self.session_returns else 0.0

            if self.verbose > 0:
                pool_size = len(self.league_pool.checkpoints)
                print(
                    f"[{self.num_timesteps:,} steps] "
                    f"Sessions: {self.total_sessions:,} | "
                    f"Freezeout Win Rate (1st Place): {rolling_win:.1f}% | "
                    f"Avg Hands: {rolling_hands:.1f} | "
                    f"Mean Return: {avg_return:+.3f} | "
                    f"League Pool: {pool_size} bots"
                )
        return True


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def train(
    timesteps: int = 1_500_000,
    n_envs: int = 16,
    n_steps: Optional[int] = None,
    starting_chips: int = 200,
    learning_rate: float = 2.5e-4,
    min_lr: float = 2.5e-5,
    n_restart_cycles: int = 5,
    save_dir: str = "models",
    league_dir: str = "models/league",
    model_name: str = "model_d_champion.zip",
    smoke_test: bool = False,
    seed: int = 42,
    freezeout: bool = True,
    blind_escalation_interval: Optional[int] = 15,
    randomize_stacks: bool = True,
    max_hands: int = 200,
    snapshot_interval: int = 250_000,
    params_path: Optional[str] = "best_params_d.json",
    embed_dim: int = 32,
    num_heads: int = 2,
    pooling: str = "mean",
    card_features_dim: int = 32,
    game_features_dim: int = 64,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(league_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)

    # Defaults
    batch_size = 64
    n_epochs = 7
    gamma = 0.9958
    gae_lambda = 0.9227
    clip_range = 0.1851
    ent_coef = 0.00764
    vf_coef = 0.5316
    max_grad_norm = 0.5723
    fold_penalty = 0.215
    net_arch = [256, 256, 256]

    # Load custom tuned parameters if available
    loaded_params = False
    resolved_path = None
    if params_path:
        for candidate in [params_path, os.path.join("configs", params_path), os.path.join("configs", os.path.basename(params_path))]:
            if os.path.exists(candidate):
                resolved_path = candidate
                break

    if resolved_path:
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            learning_rate = params.get("learning_rate", learning_rate)
            min_lr = params.get("min_lr", min_lr)
            n_restart_cycles = params.get("n_restart_cycles", n_restart_cycles)
            batch_size = params.get("batch_size", batch_size)
            n_epochs = params.get("n_epochs", n_epochs)
            gamma = params.get("gamma", gamma)
            gae_lambda = params.get("gae_lambda", gae_lambda)
            clip_range = params.get("clip_range", clip_range)
            ent_coef = params.get("ent_coef", ent_coef)
            vf_coef = params.get("vf_coef", vf_coef)
            max_grad_norm = params.get("max_grad_norm", max_grad_norm)
            fold_penalty = params.get("fold_penalty", fold_penalty)
            blind_escalation_interval = params.get("blind_escalation", blind_escalation_interval)
            net_arch = params.get("net_arch", net_arch)
            embed_dim = params.get("embed_dim", embed_dim)
            num_heads = params.get("num_heads", num_heads)
            pooling = params.get("pooling", pooling)
            card_features_dim = params.get("card_features_dim", card_features_dim)
            game_features_dim = params.get("game_features_dim", game_features_dim)
            if n_steps is None:
                n_steps = params.get("n_steps", 1536)
            loaded_params = True
            print(f"  [Config] Successfully loaded Model D hyperparameters from: {resolved_path}")
        except Exception as e:
            print(f"  [Config] Warning: Could not parse {resolved_path} ({e}), using defaults.")

    if n_steps is None:
        n_steps = 1536

    if not freezeout:
        blind_escalation_interval = None

    # Multi-Agent League Pool
    league_pool = LeaguePool(
        base_model_path="models/model_b_multiplayer.zip",
        league_dir=league_dir,
        neural_opponent_prob=0.60,
    )

    if smoke_test:
        timesteps = 500
        n_envs = 2
        effective_n_steps = 64
        batch_size = 32
        n_epochs = 2
        snapshot_interval = 250
        print("\n*** RUNNING MODEL D IN SMOKE-TEST MODE (500 steps) ***\n")
    else:
        effective_n_steps = n_steps
        total_rollout = effective_n_steps * n_envs
        print(f"=== STARTING MODEL D CHAMPION TRAINING ({timesteps:,} steps) ===")
        print(f"  Architecture: CardAttentionExtractor (dim={embed_dim}, heads={num_heads}, pool={pooling}) + MLP {net_arch}")
        print(f"  Observation: {SUPERHUMAN_OBS_DIM} dims (10 Card tokens + 77 Table/Tracker/Sequence state)")
        print(f"  LR Dynamics: Cosine Annealing with {n_restart_cycles} Warm Restarts (peak={learning_rate:.6f}, min={min_lr:.6f})")
        print(f"  Environments: {n_envs} | Steps/Env: {effective_n_steps} | Rollout Buffer: {total_rollout:,} steps")
        print(f"  League Sparring Pool: {len(league_pool.checkpoints)} active checkpoints (Snapshot every {snapshot_interval:,} steps)")

    # Vectorized Environment
    env = DummyVecEnv([
        make_superhuman_multi_env(
            num_seats=5,
            starting_chips=starting_chips,
            blind_escalation_interval=blind_escalation_interval,
            randomize_stacks=randomize_stacks,
            fold_penalty=fold_penalty,
            max_hands_per_session=max_hands,
            seed=seed + i * 100,
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
            game_features_dim=game_features_dim,
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
        n_steps=effective_n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1 if smoke_test else 0,
        seed=seed,
    )

    callbacks = [
        ModelDSessionCallback(
            league_pool=league_pool,
            log_freq=250 if smoke_test else 5_000,
            is_freezeout=bool(blind_escalation_interval),
        ),
        ModelDLeagueSnapshotCallback(
            league_pool=league_pool,
            snapshot_interval=snapshot_interval,
            save_dir=league_dir,
            verbose=1,
        ),
    ]

    model.learn(
        total_timesteps=timesteps,
        callback=CallbackList(callbacks),
        progress_bar=False,
    )

    model.save(save_path)
    print(f"\n============================================================")
    print(f"  MODEL D CHAMPION SAVED SUCCESSFULLY TO: {save_path}")
    print(f"  Total League Snapshots Created: {len(league_pool.checkpoints)}")
    print(f"============================================================\n")

    env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CardShark-RL Model D (Next-Generation Champion)")
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--chips", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--min-lr", type=float, default=2.5e-5)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--params-path", type=str, default="configs/best_params_d.json")
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--league-dir", type=str, default="models/league")
    parser.add_argument("--model-name", type=str, default="model_d_champion.zip")
    parser.add_argument("--freezeout", action="store_true", default=True)
    parser.add_argument("--no-freezeout", dest="freezeout", action="store_false")
    parser.add_argument("--blind-escalation", type=int, default=15)
    parser.add_argument("--randomize-stacks", action="store_true", default=True)
    parser.add_argument("--no-randomize-stacks", dest="randomize_stacks", action="store_false")
    parser.add_argument("--max-hands", type=int, default=200)
    parser.add_argument("--snapshot-interval", type=int, default=250_000)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        starting_chips=args.chips,
        learning_rate=args.lr,
        min_lr=args.min_lr,
        n_restart_cycles=args.cycles,
        save_dir=args.save_dir,
        league_dir=args.league_dir,
        model_name=args.model_name,
        smoke_test=args.smoke_test,
        seed=args.seed,
        freezeout=args.freezeout,
        blind_escalation_interval=args.blind_escalation,
        randomize_stacks=args.randomize_stacks,
        max_hands=args.max_hands,
        snapshot_interval=args.snapshot_interval,
        params_path=args.params_path,
    )
