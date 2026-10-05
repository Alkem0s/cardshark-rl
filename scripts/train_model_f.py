"""
train_model_f.py — Training Pipeline for CardShark-RL Model F (The Co-Evolutionary Champion).

Key Architectural Innovations over Model E:
1. Decoupled Dual-Stream Actor Architecture:
   Separates betting decisions (6 logits) and discard choices (32 logits) into dedicated
   sub-branches, completely eliminating cross-phase gradient interference.
2. 97-Dimensional State Geometry Vector:
   Injects exact mathematical game theory features (pot odds, minimum defense frequency,
   aggressor draw count, escalation ratio, showdown percentile, probe indicator)
   without imposing any rigid decision heuristics.
3. Auxiliary Supervised Belief Head:
   Trains the Card Attention backbone to predict opponent hand ranges and bluff status
   using privileged environment data during rollouts, shaping rich latent representations.
4. True Neural Co-Evolution (PFSP):
   Fully verified native self-play sparring with rolling historical snapshots,
   curriculum-balanced with Adversarial Exploiter and Model E baselines.
5. Cosine Annealing Schedules for LR and Entropy:
   Smoothly transitions from broad early exploration to razor-sharp late-stage exploitation.
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
    make_model_f_multi_env,
    multi_mask_fn,
    ModelObsDim,
    OBS_DIM_F,
)
from rl.league import LeaguePool
from rl.card_attention import CardAttentionExtractor
from rl.decoupled_actor import DecoupledMaskableActorCriticPolicy
from rl.auxiliary_belief import AuxiliaryBeliefHead, AuxiliaryBeliefCallback
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


class EntropyDecayCallback(BaseCallback):
    """Dynamically decays entropy coefficient using a smooth cosine schedule."""

    def __init__(
        self,
        initial_ent: float = 0.015,
        min_ent: float = 0.001,
        total_timesteps: int = 2_000_000,
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


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class ModelFLeagueSnapshotCallback(BaseCallback):
    """Saves rolling Model F checkpoints and registers them for self-play sparring."""

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
            snapshot_name = f"model_f_step_{self.num_timesteps}.zip"
            snapshot_path = os.path.join(self.save_dir, snapshot_name)
            self.model.save(snapshot_path)
            self.league_pool.add_snapshot(snapshot_path)
            if self.verbose > 0:
                print(f"\n  >>> [MODEL F SNAPSHOT] Saved & registered sparring agent: {snapshot_path} <<<")
                active_count = len(getattr(self.league_pool, "active_snapshots", []))
                print(f"      League Pool: {len(self.league_pool.checkpoints)} agents ({active_count} active self-play snapshots)\n")

            if self.next_snapshot_step == self.first_snapshot_step and self.first_snapshot_step < self.snapshot_interval:
                self.next_snapshot_step = self.snapshot_interval
            else:
                self.next_snapshot_step += self.snapshot_interval

        return True


class ModelFSessionCallback(BaseCallback):
    """Logs rolling tournament metrics and saves the best model based on rolling 1st-place rate."""

    def __init__(
        self,
        league_pool: LeaguePool,
        log_freq: int = 5_000,
        window: int = 100,
        save_best_path: Optional[str] = "models/model_f.zip",
        initial_best_win_rate: float = -1.0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.league_pool = league_pool
        self.log_freq = log_freq
        self.window = window
        self.save_best_path = save_best_path
        self.best_win_rate = initial_best_win_rate
        self.verbose = verbose

        self.survived_history: List[int] = []
        self.winner_history: List[int] = []
        self.hands_history: List[int] = []
        self.session_returns: List[float] = []
        self.total_sessions = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if done and i < len(infos):
                info = infos[i]
                if "is_winner" in info:
                    hands = info.get("hands_played", 0)
                    hero_chips = info.get("hero_chips", 0)
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

            if self.verbose > 0:
                pool_size = len(self.league_pool.checkpoints)
                active_self = len(getattr(self.league_pool, "active_snapshots", []))
                neural_p = self.league_pool.neural_opponent_prob * 100.0
                print(
                    f"[{self.num_timesteps:,} steps] "
                    f"Sessions: {self.total_sessions:,} | "
                    f"Win Rate (1st Place): {rolling_win:.1f}% | "
                    f"Survival: {rolling_survival:.1f}% | "
                    f"Avg Hands: {rolling_hands:.1f} | "
                    f"League: {pool_size} bots ({active_self} self-play, {neural_p:.0f}% neural)"
                )

            if self.save_best_path and len(self.winner_history) >= min(self.window, 30) and rolling_win > self.best_win_rate:
                self.best_win_rate = rolling_win
                self.model.save(self.save_best_path)
                if self.verbose > 0:
                    print(f"      >>> [NEW BEST MODEL F] Saved peak checkpoint: {self.save_best_path} ({rolling_win:.1f}% win rate) <<<")
        return True


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def train_model_f(
    timesteps: int = 2_000_000,
    n_envs: int = 14,
    starting_chips: int = 200,
    save_dir: str = "models",
    league_dir: str = "models/league",
    model_name: str = "model_f.zip",
    smoke_test: bool = False,
    seed: int = 42,
    blind_escalation: int = 15,
    snapshot_interval: int = 250_000,
    params_path: str = "configs/best_params_f.json",
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(league_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)

    # Defaults
    learning_rate = 1.65e-4
    min_lr = 2.0e-5
    n_restart_cycles = 7
    initial_ent_coef = 0.015
    min_ent_coef = 0.001
    n_steps = 1024
    batch_size = 128
    n_epochs = 7
    gamma = 0.994
    gae_lambda = 0.913
    clip_range = 0.185
    vf_coef = 0.5
    max_grad_norm = 0.5
    fold_penalty = 0.075
    exploiter_prob = 0.30
    self_play_prob = 0.40
    aux_lr = 1.0e-4
    aux_bluff_weight = 0.5

    embed_dim = 32
    num_heads = 4
    pooling = "attention"
    card_features_dim = 32
    slot_features_dim = 24
    game_features_dim = 64
    net_arch = [128, 128]

    resolved_path = os.path.join(PROJECT_ROOT, params_path) if not os.path.isabs(params_path) else params_path
    if os.path.exists(resolved_path):
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            learning_rate = params.get("learning_rate", learning_rate)
            min_lr = params.get("min_lr", min_lr)
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
            blind_escalation = params.get("blind_escalation", blind_escalation)
            embed_dim = params.get("embed_dim", embed_dim)
            num_heads = params.get("num_heads", num_heads)
            pooling = params.get("pooling", pooling)
            card_features_dim = params.get("card_features_dim", card_features_dim)
            slot_features_dim = params.get("slot_features_dim", slot_features_dim)
            game_features_dim = params.get("game_features_dim", game_features_dim)
            net_arch = params.get("net_arch", net_arch)
            exploiter_prob = params.get("exploiter_prob", exploiter_prob)
            self_play_prob = params.get("self_play_prob", self_play_prob)
            aux_lr = params.get("aux_lr", aux_lr)
            aux_bluff_weight = params.get("aux_bluff_weight", aux_bluff_weight)
            print(f"  [Config] Successfully loaded Model F hyperparameters from: {resolved_path}")
        except Exception as e:
            print(f"  [Config] Warning: Could not parse {resolved_path} ({e}), using defaults.")

    # Archive previous Model F runs if starting from scratch
    if not smoke_test and os.path.exists(save_path):
        ts = int(time.time())
        archive_dir = os.path.join(save_dir, "archive", f"previous_model_f_{ts}")
        os.makedirs(archive_dir, exist_ok=True)
        shutil.copy2(save_path, os.path.join(archive_dir, model_name))
        print(f"  [Archive] Backed up prior Model F checkpoint to: {archive_dir}/{model_name}")

    # Multi-Agent League Pool with Prioritized Fictitious Self-Play (PFSP)
    league_pool = LeaguePool(
        base_model_path="models/model_e.zip",
        league_dir=league_dir,
        neural_opponent_prob=0.55,
        min_neural_prob=0.35,
        curriculum_warmup_steps=250_000 if not smoke_test else 250,
        self_play_prob=self_play_prob,
        exploiter_prob=exploiter_prob,
        exclude_prefix="model_f_step_",
    )

    if smoke_test:
        timesteps = 500
        n_envs = 2
        effective_n_steps = 64
        batch_size = 32
        n_epochs = 2
        snapshot_interval = 250
        print("\n*** RUNNING MODEL F IN SMOKE-TEST MODE (500 steps) ***\n")
    else:
        effective_n_steps = n_steps
        total_rollout = effective_n_steps * n_envs
        print(f"=== STARTING MODEL F CHAMPION TRAINING ({timesteps:,} steps) ===")
        print(f"  Architecture: CardAttentionExtractor (dim={embed_dim}, heads={num_heads}, pool={pooling}, slots={slot_features_dim}x5)")
        print(f"  Policy Head: Decoupled Dual-Stream Actor (Betting: 6 logits, Draw: 32 logits, Gradient Isolated)")
        print(f"  Observation: {OBS_DIM_F} dims (10 Card + 77 Table/Sequence + 4 ICM + 6 Game Geometry features)")
        print(f"  Auxiliary Head: Opponent Range & Bluff Self-Supervised Learning (LR={aux_lr:.1e}, weight={aux_bluff_weight})")
        print(f"  Entropy Dynamics: Cosine Decay ({initial_ent_coef:.3f} -> {min_ent_coef:.3f})")
        print(f"  LR Dynamics: Cosine Annealing with {n_restart_cycles} Warm Restarts (peak={learning_rate:.6f}, min={min_lr:.6f})")
        print(f"  Environments: {n_envs} | Steps/Env: {effective_n_steps} | Rollout Buffer: {total_rollout:,} steps")
        print(f"  Sparring Distribution: {int(self_play_prob*100)}% Self-Play (PFSP) | {int(exploiter_prob*100)}% Adversarial Exploiter | 15% Model E | 15% Heuristics")

    # Vectorized Environment with Model F 97-dim state and Discard Masking
    env = DummyVecEnv([
        make_model_f_multi_env(
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
        policy=DecoupledMaskableActorCriticPolicy,
        env=env,
        learning_rate=lr_schedule,
        n_steps=effective_n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=initial_ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
    )

    # Feature extractor dimension: embed_dim (32) + 5 * slot_features_dim (120) + game_features_dim (64) = 216
    extracted_feature_dim = embed_dim + (5 * slot_features_dim) + game_features_dim
    belief_head = AuxiliaryBeliefHead(feature_dim=extracted_feature_dim, hidden_dim=128)

    aux_callback = AuxiliaryBeliefCallback(
        belief_head=belief_head,
        lr=aux_lr,
        bluff_weight=aux_bluff_weight,
        verbose=1 if smoke_test else 0,
    )

    entropy_callback = EntropyDecayCallback(
        initial_ent=initial_ent_coef,
        min_ent=min_ent_coef,
        total_timesteps=timesteps,
        verbose=0,
    )

    session_callback = ModelFSessionCallback(
        league_pool=league_pool,
        log_freq=5_000 if not smoke_test else 100,
        window=100 if not smoke_test else 10,
        save_best_path=save_path,
        verbose=1,
    )

    snapshot_callback = ModelFLeagueSnapshotCallback(
        league_pool=league_pool,
        first_snapshot_step=125_000 if not smoke_test else 250,
        snapshot_interval=snapshot_interval,
        save_dir=league_dir,
        verbose=1,
    )

    callbacks = CallbackList([aux_callback, entropy_callback, session_callback, snapshot_callback])

    t0 = time.time()
    try:
        model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)
    except KeyboardInterrupt:
        print("\n[Training] Interrupted by user. Saving current checkpoint...")
    finally:
        model.save(save_path)
        elapsed = time.time() - t0
        print(f"\n[Training Complete] Elapsed time: {elapsed/60:.1f} mins | Saved to: {save_path}")

    env.close()
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CardShark-RL Model F (Co-Evolutionary Champion)")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=14, help="Parallel Gymnasium worker environments")
    parser.add_argument("--chips", type=int, default=200, help="Starting chips per player")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--smoke-test", action="store_true", help="Quick 500-step verification run")
    parser.add_argument("--blind-escalation", type=int, default=15, help="Hands between blind raises")
    parser.add_argument("--snapshot-interval", type=int, default=250_000, help="Timesteps between self-play snapshots")
    parser.add_argument("--config", type=str, default="configs/best_params_f.json", help="Path to JSON hyperparameters")
    args = parser.parse_args()

    train_model_f(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        starting_chips=args.chips,
        seed=args.seed,
        smoke_test=args.smoke_test,
        blind_escalation=args.blind_escalation,
        snapshot_interval=args.snapshot_interval,
        params_path=args.config,
    )
