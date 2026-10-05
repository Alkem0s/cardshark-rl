"""
train_model_e.py — Training Pipeline for CardShark-RL Model E (The Superhuman Crusher).

Key Architectural Innovations over Model D:
1. Dominated Discard Action Masking Engine:
   Mathematically eliminates suicidal discards (e.g. breaking a made flush or pair),
   guaranteeing 0% draw blunders from Step 0 and putting discard mechanics on parity with heuristics.
2. 91-Dimensional ICM & Blind-Aware Observation Space:
   Injects Big Blind depth (Stack/BB, Pot/BB, Blind Pressure, Escalation Clock),
   empowering the policy to navigate deep-stack vs push/fold tournament phases.
3. Anti-Heuristic League Sparring Inoculation:
   Heavy 35% AdversarialExploiter + 20% TAG/LAG heuristic sparring combined with historical
   Model D and Model C champions, training the agent to check-raise trap aggressive probe bets.
4. Cosine Annealing with 7 Warm Restarts (SGDR):
   Synchronized learning rate cycling to absorb new self-play snapshots without policy collapse.
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.multi_gym_wrapper import (
    make_model_e_multi_env,
    multi_mask_fn,
    MODEL_E_OBS_DIM,
)
from rl.league import LeaguePool
from rl.card_attention import CardAttentionExtractor
import rl.card_attention
sys.modules["card_attention"] = rl.card_attention


# ---------------------------------------------------------------------------
# Cosine Annealing with Warm Restarts (SGDR)
# ---------------------------------------------------------------------------

def cosine_warm_restart_schedule(
    initial_lr: float,
    min_lr: float = 2.0e-5,
    n_cycles: int = 7,
    warmup_fraction: float = 0.05,
) -> Callable[[float], float]:
    """Cosine Annealing with Warm Restarts (SGDR) for Stable-Baselines3."""
    def schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining  # 0.0 -> 1.0
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


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class ModelELeagueSnapshotCallback(BaseCallback):
    """Saves rolling Model E checkpoints at regular intervals and adds them to the sparring pool."""

    def __init__(
        self,
        league_pool: LeaguePool,
        first_snapshot_step: int = 125_000,
        snapshot_interval: int = 250_000,
        save_dir: str = "models/league",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.league_pool = league_pool
        self.first_snapshot_step = first_snapshot_step
        self.snapshot_interval = snapshot_interval
        self.save_dir = save_dir
        self.next_snapshot_step = first_snapshot_step
        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        self.league_pool.set_step(self.num_timesteps)

        if self.num_timesteps >= self.next_snapshot_step:
            snapshot_name = f"model_e_step_{self.num_timesteps}.zip"
            snapshot_path = os.path.join(self.save_dir, snapshot_name)
            self.model.save(snapshot_path)
            self.league_pool.add_snapshot(snapshot_path)
            if self.verbose > 0:
                print(f"\n  >>> [MODEL E SNAPSHOT] Saved & registered sparring agent: {snapshot_path} <<<")
                active_count = len(getattr(self.league_pool, "active_snapshots", []))
                print(f"      League Pool: {len(self.league_pool.checkpoints)} agents ({active_count} active self-play snapshots)\n")

            if self.next_snapshot_step == self.first_snapshot_step and self.first_snapshot_step < self.snapshot_interval:
                self.next_snapshot_step = self.snapshot_interval
            else:
                self.next_snapshot_step += self.snapshot_interval

        return True


class ModelESessionCallback(BaseCallback):
    """Logs rolling tournament metrics and saves the best model based on rolling 1st-place rate."""

    def __init__(
        self,
        league_pool: LeaguePool,
        log_freq: int = 5_000,
        window: int = 100,
        save_best_path: Optional[str] = "models/model_e.zip",
        initial_best_win_rate: float = -1.0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.league_pool = league_pool
        self.log_freq = log_freq
        self.window = window
        self.save_best_path = save_best_path

        self.session_returns: list[float] = []
        self.survived_history: list[int] = []
        self.winner_history: list[int] = []
        self.hands_history: list[int] = []
        self.total_sessions = 0
        self.best_win_rate = initial_best_win_rate

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards")

        if not hasattr(self, "env_returns") or len(self.env_returns) != len(dones):
            self.env_returns = np.zeros(len(dones), dtype=np.float32)

        if rewards is not None:
            self.env_returns += rewards

        for i, done in enumerate(dones):
            if done:
                self.session_returns.append(float(self.env_returns[i]))
                self.env_returns[i] = 0.0
                if len(self.session_returns) > self.window:
                    self.session_returns.pop(0)

                if i < len(infos):
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

        if self.num_timesteps % self.log_freq == 0 and len(self.survived_history) > 0:
            rolling_survival = np.mean(self.survived_history) * 100.0
            rolling_win = np.mean(self.winner_history) * 100.0
            rolling_hands = np.mean(self.hands_history)
            avg_return = np.mean(self.session_returns) if self.session_returns else 0.0

            if self.verbose > 0:
                pool_size = len(self.league_pool.checkpoints)
                active_self = len(getattr(self.league_pool, "active_snapshots", []))
                neural_p = self.league_pool.neural_opponent_prob * 100.0
                print(
                    f"[{self.num_timesteps:,} steps] "
                    f"Sessions: {self.total_sessions:,} | "
                    f"Freezeout Win Rate (1st Place): {rolling_win:.1f}% | "
                    f"Avg Hands: {rolling_hands:.1f} | "
                    f"Mean Return: {avg_return:+.3f} | "
                    f"League: {pool_size} bots ({active_self} self-play, {neural_p:.0f}% neural)"
                )

            if self.save_best_path and len(self.winner_history) >= min(self.window, 30) and rolling_win > self.best_win_rate:
                self.best_win_rate = rolling_win
                self.model.save(self.save_best_path)
                if self.verbose > 0:
                    print(f"      >>> [NEW BEST MODEL E] Saved peak champion checkpoint: {self.save_best_path} ({rolling_win:.1f}% win rate) <<<")
        return True


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def train_model_e(
    timesteps: int = 2_000_000,
    n_envs: int = 14,
    starting_chips: int = 200,
    save_dir: str = "models",
    league_dir: str = "models/league",
    model_name: str = "model_e.zip",
    smoke_test: bool = False,
    seed: int = 42,
    blind_escalation: int = 15,
    snapshot_interval: int = 250_000,
    params_path: str = "configs/best_params_e.json",
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(league_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)

    # Defaults
    learning_rate = 1.65e-4
    min_lr = 2.0e-5
    n_restart_cycles = 7
    n_steps = 1024
    batch_size = 128
    n_epochs = 7
    gamma = 0.994
    gae_lambda = 0.913
    clip_range = 0.185
    ent_coef = 0.0065
    vf_coef = 0.52
    max_grad_norm = 0.65
    fold_penalty = 0.075
    embed_dim = 32
    num_heads = 4
    pooling = "attention"
    card_features_dim = 32
    slot_features_dim = 24
    game_features_dim = 64
    net_arch = [256, 256, 256]
    exploiter_prob = 0.35

    # Load custom tuned parameters if available
    resolved_path = None
    for cand in [params_path, os.path.join("configs", params_path), os.path.join("configs", os.path.basename(params_path))]:
        if os.path.exists(cand):
            resolved_path = cand
            break

    if resolved_path:
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            learning_rate = params.get("learning_rate", learning_rate)
            min_lr = params.get("min_lr", min_lr)
            n_restart_cycles = params.get("n_restart_cycles", n_restart_cycles)
            n_steps = params.get("n_steps", n_steps)
            batch_size = params.get("batch_size", batch_size)
            n_epochs = params.get("n_epochs", n_epochs)
            gamma = params.get("gamma", gamma)
            gae_lambda = params.get("gae_lambda", gae_lambda)
            clip_range = params.get("clip_range", clip_range)
            ent_coef = params.get("ent_coef", ent_coef)
            vf_coef = params.get("vf_coef", vf_coef)
            max_grad_norm = params.get("max_grad_norm", max_grad_norm)
            fold_penalty = params.get("fold_penalty", fold_penalty)
            blind_escalation = params.get("blind_escalation", blind_escalation)
            embed_dim = params.get("embed_dim", embed_dim)
            num_heads = params.get("num_heads", num_heads)
            pooling = params.get("pooling", pooling)
            card_features_dim = params.get("card_features_dim", card_features_dim)
            slot_features_dim = params.get("slot_features_dim", slot_features_dim)
            game_features_dim = params.get("game_features_dim", game_features_dim)
            net_arch = params.get("net_arch", net_arch)
            exploiter_prob = params.get("exploiter_prob", exploiter_prob)
            print(f"  [Config] Successfully loaded Model E hyperparameters from: {resolved_path}")
        except Exception as e:
            print(f"  [Config] Warning: Could not parse {resolved_path} ({e}), using defaults.")

    # Multi-Agent League Pool with Anti-Heuristic Inoculation
    league_pool = LeaguePool(
        base_model_path="models/model_b.zip",
        league_dir=league_dir,
        neural_opponent_prob=0.50,
        min_neural_prob=0.30,
        curriculum_warmup_steps=250_000 if not smoke_test else 250,
        self_play_prob=0.40,
        exploiter_prob=exploiter_prob,
        exclude_prefix="model_e_step_",
    )

    if smoke_test:
        timesteps = 500
        n_envs = 2
        effective_n_steps = 64
        batch_size = 32
        n_epochs = 2
        snapshot_interval = 250
        print("\n*** RUNNING MODEL E IN SMOKE-TEST MODE (500 steps) ***\n")
    else:
        effective_n_steps = n_steps
        total_rollout = effective_n_steps * n_envs
        print(f"=== STARTING MODEL E CHAMPION TRAINING ({timesteps:,} steps) ===")
        print(f"  Architecture: CardAttentionExtractor (dim={embed_dim}, heads={num_heads}, pool={pooling}, slots={slot_features_dim}x5) + MLP {net_arch}")
        print(f"  Observation: {MODEL_E_OBS_DIM} dims (10 Card tokens + 77 Table/Sequence + 4 Tournament ICM features)")
        print(f"  Discard Policy: Dominated Discard Action Masking Engine (Strict GTO Draw Pruning)")
        print(f"  LR Dynamics: Cosine Annealing with {n_restart_cycles} Warm Restarts (peak={learning_rate:.6f}, min={min_lr:.6f})")
        print(f"  Environments: {n_envs} | Steps/Env: {effective_n_steps} | Rollout Buffer: {total_rollout:,} steps")
        print(f"  Curriculum Sparring: 35% Adversarial Exploiter | 50% Neural (Model D & Self-Play) | 15% Heuristics")
        print(f"  League Sparring Pool: {len(league_pool.checkpoints)} baseline checkpoints (1st snapshot at 125,000 steps, then every {snapshot_interval:,} steps)")

    # Vectorized Environment with Model E 91-dim state and Discard Masking
    env = DummyVecEnv([
        make_model_e_multi_env(
            num_seats=5,
            starting_chips=starting_chips,
            small_blind=1,
            big_blind=2,
            hero_seat=None,
            max_hands_per_session=200,
            blind_escalation_interval=blind_escalation,
            randomize_stacks=True,
            fold_penalty=fold_penalty,
            seed=seed + i * 100,
            league_pool=league_pool,
        )
        for i in range(n_envs)
    ])

    lr_schedule = cosine_warm_restart_schedule(
        initial_lr=learning_rate,
        min_lr=min_lr,
        n_cycles=n_restart_cycles,
    )

    policy_kwargs = dict(
        features_extractor_class=CardAttentionExtractor,
        features_extractor_kwargs=dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            pooling=pooling,
            card_features_dim=card_features_dim,
            game_features_dim=game_features_dim,
            slot_features_dim=slot_features_dim,
        ),
        net_arch=dict(pi=net_arch, vf=net_arch),
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
        verbose=1,
        seed=seed,
    )

    snapshot_callback = ModelELeagueSnapshotCallback(
        league_pool=league_pool,
        first_snapshot_step=125_000 if not smoke_test else 250,
        snapshot_interval=snapshot_interval,
        save_dir=league_dir,
    )

    session_callback = ModelESessionCallback(
        league_pool=league_pool,
        log_freq=5_000 if not smoke_test else 250,
        window=100,
        save_best_path=save_path if not smoke_test else "models/smoke_test_model_e.zip",
    )

    callbacks = CallbackList([snapshot_callback, session_callback])

    start_time = time.time()
    try:
        model.learn(total_timesteps=timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print("\n[Training Interrupted] Saving current model state...")

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    print(f"\nTraining completed in {hours}h {mins}m ({elapsed:.1f}s)")

    # Final model save
    if not smoke_test:
        model.save(save_path)
        print(f"\n============================================================")
        print(f"  MODEL E SAVED SUCCESSFULLY TO: {save_path}")

        # Training Manifest
        manifest = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timesteps": timesteps,
            "n_envs": n_envs,
            "effective_n_steps": effective_n_steps,
            "learning_rate": learning_rate,
            "min_lr": min_lr,
            "n_restart_cycles": n_restart_cycles,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "fold_penalty": fold_penalty,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "pooling": pooling,
            "card_features_dim": card_features_dim,
            "slot_features_dim": slot_features_dim,
            "game_features_dim": game_features_dim,
            "net_arch": net_arch,
            "exploiter_prob": exploiter_prob,
            "model_e_obs_dim": MODEL_E_OBS_DIM,
            "discard_masking": True,
            "device": "cpu",
            "league_snapshots_count": len(snapshot_callback.league_pool.checkpoints),
        }
        os.makedirs("results", exist_ok=True)
        manifest_path = "results/model_e_training_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"  TRAINING RUN MANIFEST WRITTEN TO: {manifest_path}")
        print(f"  Total League Snapshots Created: {len(snapshot_callback.league_pool.checkpoints)}")
        print(f"============================================================\n")

    env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CardShark-RL Model E Champion")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=14, help="Parallel environments")
    parser.add_argument("--chips", type=int, default=200, help="Starting stack per seat")
    parser.add_argument("--blind-escalation", type=int, default=15, help="Hands between blind escalations")
    parser.add_argument("--smoke-test", action="store_true", help="Quick 500-step smoke test")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    train_model_e(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        starting_chips=args.chips,
        blind_escalation=args.blind_escalation,
        smoke_test=args.smoke_test,
        seed=args.seed,
    )
