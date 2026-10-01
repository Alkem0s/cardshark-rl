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
        # Sync step to league pool for dynamic curriculum opponent probability
        self.league_pool.set_step(self.num_timesteps)

        if self.num_timesteps >= self.next_snapshot_step:
            snapshot_name = f"model_d_step_{self.num_timesteps}.zip"
            snapshot_path = os.path.join(self.save_dir, snapshot_name)
            self.model.save(snapshot_path)
            self.league_pool.add_snapshot(snapshot_path)
            if self.verbose > 0:
                print(f"\n  >>> [MODEL D SNAPSHOT] Saved & registered sparring agent: {snapshot_path} <<<")
                active_count = len(getattr(self.league_pool, "active_snapshots", []))
                print(f"      League Pool: {len(self.league_pool.checkpoints)} agents ({active_count} active self-play snapshots)\n")

            if self.next_snapshot_step == self.first_snapshot_step and self.first_snapshot_step < self.snapshot_interval:
                self.next_snapshot_step = self.snapshot_interval
            else:
                self.next_snapshot_step += self.snapshot_interval

        return True


class ModelDSessionCallback(BaseCallback):
    """Logs rolling tournament metrics and saves the best model based on rolling 1st-place rate."""

    def __init__(
        self,
        league_pool: LeaguePool,
        log_freq: int = 5_000,
        window: int = 100,
        is_freezeout: bool = True,
        save_best_path: Optional[str] = "models/model_d.zip",
        initial_best_win_rate: float = -1.0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.league_pool = league_pool
        self.log_freq = log_freq
        self.window = window
        self.is_freezeout = is_freezeout
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
                    print(f"      >>> [NEW BEST MODEL] Saved peak champion checkpoint: {self.save_best_path} ({rolling_win:.1f}% win rate) <<<")
        return True


# ---------------------------------------------------------------------------
# League Archiving for Clean Scratch Runs
# ---------------------------------------------------------------------------

def archive_previous_run(save_dir: str = "models", league_dir: str = "models/league") -> Optional[str]:
    """Archives previous Model D checkpoints and snapshots so training from scratch starts with a clean pool."""
    d_snapshots = []
    if os.path.exists(league_dir):
        d_snapshots = [
            f for f in os.listdir(league_dir)
            if f.startswith("model_d_step_") and f.endswith(".zip")
        ]
    model_d_path = os.path.join(save_dir, "model_d.zip")
    best_path = os.path.join(save_dir, "model_d_best.zip")
    champ_path = os.path.join(save_dir, "model_d_champion.zip")
    has_artifacts = bool(d_snapshots) or os.path.exists(model_d_path) or os.path.exists(best_path) or os.path.exists(champ_path)
    if not has_artifacts:
        return None

    # Check if this includes the 1.5M champion to give it a descriptive name
    is_1_5m = any("1500000" in s for s in d_snapshots)
    tag = "run_1.5m_champion" if is_1_5m else f"previous_run_{int(time.time())}"
    archive_dir = os.path.join(save_dir, "archive", tag)
    os.makedirs(archive_dir, exist_ok=True)

    archived_count = 0
    for snap in d_snapshots:
        src = os.path.join(league_dir, snap)
        dst = os.path.join(archive_dir, snap)
        shutil.copy2(src, dst)
        try:
            os.remove(src)
        except OSError:
            pass
        archived_count += 1

    # Archive previous model checkpoint (prefer model_d.zip or fallback to _best/_champion)
    active_model = None
    for candidate in [model_d_path, best_path, champ_path]:
        if os.path.exists(candidate):
            active_model = candidate
            break

    if active_model:
        shutil.copy2(active_model, os.path.join(archive_dir, "model_d.zip"))
        for candidate in [model_d_path, best_path, champ_path]:
            if os.path.exists(candidate):
                try:
                    os.remove(candidate)
                except OSError:
                    pass

    print(f"\n  [League Setup] Clean scratch run detected:")
    print(f"    - Archived {archived_count} previous Model D snapshots to: {archive_dir}")
    if active_model:
        print(f"    - Backed up previous model checkpoint to: {os.path.join(archive_dir, 'model_d.zip')}")
    print(f"    - Resetting sparring league to baseline opponents\n")
    return archive_dir


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def train(
    timesteps: int = 1_500_000,
    n_envs: int = 16,
    n_steps: Optional[int] = None,
    starting_chips: int = 200,
    learning_rate: Optional[float] = None,
    min_lr: Optional[float] = None,
    n_restart_cycles: Optional[int] = None,
    save_dir: str = "models",
    league_dir: str = "models/league",
    model_name: str = "model_d.zip",
    smoke_test: bool = False,
    seed: int = 42,
    freezeout: bool = True,
    blind_escalation_interval: Optional[int] = 15,
    randomize_stacks: bool = True,
    max_hands: int = 200,
    snapshot_interval: int = 250_000,
    params_path: Optional[str] = "best_params_d.json",
    resume_path: Optional[str] = None,
    preserve_league: bool = False,
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
            if learning_rate is None:
                learning_rate = params.get("learning_rate", 2.15e-4)
            if min_lr is None:
                min_lr = params.get("min_lr", 2.5e-5)
            if n_restart_cycles is None:
                n_restart_cycles = params.get("n_restart_cycles", 7 if timesteps >= 2_000_000 else 5)
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

    # Scratch run clean-up: Ensure newborn agent starts against baseline bots without late-stage champions
    if resume_path is None and not preserve_league and not smoke_test:
        archive_previous_run(save_dir=save_dir, league_dir=league_dir)

    # In smoke test, clean up any transient test snapshots
    if smoke_test and os.path.exists(league_dir):
        for f in os.listdir(league_dir):
            if f.startswith("model_d_step_250.") or f.startswith("model_d_step_500."):
                try:
                    os.remove(os.path.join(league_dir, f))
                except OSError:
                    pass

    # Multi-Agent League Pool (PFSP with Dynamic Curriculum Warmup)
    league_pool = LeaguePool(
        base_model_path="models/model_b.zip",
        league_dir=league_dir,
        neural_opponent_prob=0.65,
        min_neural_prob=0.35,
        curriculum_warmup_steps=250_000 if not smoke_test else 250,
        self_play_prob=0.50,
        exclude_prefix="model_d_step_" if (resume_path is None and not preserve_league) else None,
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
        print(f"  Curriculum Sparring: Neural density ramps 35% -> 65% over 250k steps | 50% Recency-Weighted Self-Play")
        print(f"  League Sparring Pool: {len(league_pool.checkpoints)} baseline checkpoints (1st snapshot at 125,000 steps, then every {snapshot_interval:,} steps)")

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

    if resume_path and os.path.exists(resume_path):
        print(f"  [Resume] Loading pre-trained checkpoint to continue training: {resume_path}")
        model = MaskablePPO.load(
            resume_path,
            env=env,
            custom_objects={
                "learning_rate": lr_schedule,
                "clip_range": clip_range,
                "ent_coef": ent_coef,
                "vf_coef": vf_coef,
                "max_grad_norm": max_grad_norm,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "n_steps": effective_n_steps,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
            },
            seed=seed,
        )
    else:
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

    first_snapshot_step = 250 if smoke_test else 125_000
    callbacks = [
        ModelDSessionCallback(
            league_pool=league_pool,
            log_freq=250 if smoke_test else 5_000,
            is_freezeout=bool(blind_escalation_interval),
            save_best_path=save_path,
            initial_best_win_rate=37.0 if (resume_path and os.path.exists(resume_path)) else -1.0,
        ),
        ModelDLeagueSnapshotCallback(
            league_pool=league_pool,
            first_snapshot_step=first_snapshot_step,
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

    if smoke_test:
        save_path = os.path.join(save_dir, "smoke_test_model_d.zip")
        model.save(save_path)
    else:
        # If no rolling peak model was saved during training, save the final model state
        if not os.path.exists(save_path):
            model.save(save_path)

    # Save complete training run manifest
    manifest_path = "results/model_d_training_manifest.json"
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
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
        "game_features_dim": game_features_dim,
        "net_arch": net_arch,
        "device": str(model.device),
        "league_snapshots_count": len(league_pool.checkpoints),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n============================================================")
    print(f"  MODEL D SAVED SUCCESSFULLY TO: {save_path}")
    print(f"  TRAINING RUN MANIFEST WRITTEN TO: {manifest_path}")
    print(f"  Total League Snapshots Created: {len(league_pool.checkpoints)}")
    print(f"============================================================\n")

    env.close()
    return model


train_model_d = train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CardShark-RL Model D (Next-Generation Champion)")
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--chips", type=int, default=200)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--params-path", type=str, default="configs/best_params_d.json")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to continue training from")
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--league-dir", type=str, default="models/league")
    parser.add_argument("--model-name", type=str, default="model_d.zip")
    parser.add_argument("--freezeout", action="store_true", default=True)
    parser.add_argument("--no-freezeout", dest="freezeout", action="store_false")
    parser.add_argument("--blind-escalation", type=int, default=15)
    parser.add_argument("--randomize-stacks", action="store_true", default=True)
    parser.add_argument("--no-randomize-stacks", dest="randomize_stacks", action="store_false")
    parser.add_argument("--max-hands", type=int, default=200)
    parser.add_argument("--snapshot-interval", type=int, default=250_000)
    parser.add_argument("--preserve-league", action="store_true", default=False, help="Do not archive previous Model D snapshots before starting training")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cycles = args.cycles
    if cycles is None and args.timesteps >= 2_000_000:
        cycles = 7

    train(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        starting_chips=args.chips,
        learning_rate=args.lr,
        min_lr=args.min_lr,
        n_restart_cycles=cycles,
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
        resume_path=args.resume_from,
        preserve_league=args.preserve_league,
    )
