"""
train.py — Training pipeline for Model A (Explicit) and Model B (Implicit).

Uses MaskablePPO from sb3-contrib with action masking.
Trains against randomised opponent rotations.
"""

from __future__ import annotations
import os
import time
import numpy as np
from pathlib import Path
from typing import Optional

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import configure

from gym_wrapper import DrawPokerGymEnv, mask_fn


# ---------------------------------------------------------------------------
# Custom callback for tracking training metrics
# ---------------------------------------------------------------------------

class RewardTrackingCallback(BaseCallback):
    """Tracks per-episode rewards during training for plotting."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.timestamps: list[int] = []
        self._current_rewards: dict[int, float] = {}

    def _on_step(self) -> bool:
        # Check for episode completions across all envs
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                infos = self.locals.get("infos", [])
                if i < len(infos):
                    ep_info = infos[i].get("episode")
                    if ep_info:
                        self.episode_rewards.append(ep_info["r"])
                        self.episode_lengths.append(ep_info["l"])
                        self.timestamps.append(self.num_timesteps)
        return True


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_model(
    model_name: str,
    use_implicit: bool,
    total_timesteps: int = 500_000,
    save_dir: str = "models",
    log_dir: str = "logs",
    seed: int = 42,
    net_arch: list[int] | None = None,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 256,
    n_epochs: int = 10,
    gamma: float = 0.99,
    device: str = "auto",
    verbose: int = 1,
) -> tuple[MaskablePPO, RewardTrackingCallback]:
    """Train a MaskablePPO model on the Draw Poker environment.

    Args:
        model_name: Name for saving (e.g., 'model_a_explicit')
        use_implicit: If True, use implicit (rolling stats) observation.
                      If False, use explicit (one-hot opponent ID).
        total_timesteps: Number of training steps.
        save_dir: Directory to save trained models.
        log_dir: Directory for tensorboard logs.
        seed: Random seed.
        net_arch: MLP architecture (default [256, 256]).
        learning_rate: PPO learning rate.
        n_steps: Steps per rollout.
        batch_size: Minibatch size.
        n_epochs: PPO epochs per update.
        gamma: Discount factor.
        device: 'auto', 'cpu', or 'cuda'.
        verbose: Verbosity level.

    Returns:
        Tuple of (trained model, reward callback).
    """
    if net_arch is None:
        net_arch = [256, 256]

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    mode_str = "Implicit" if use_implicit else "Explicit"
    print(f"\n{'='*60}")
    print(f"  Training {model_name} ({mode_str} Modeling)")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"  Net arch: {net_arch}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Create environment (opponent_id=None → random rotation)
    env = DrawPokerGymEnv(
        use_implicit_modeling=use_implicit,
        opponent_id=None,  # Random opponent each episode
        rolling_window=50,
        rng_seed=seed,
    )

    # Wrap with action masker
    env = ActionMasker(env, mask_fn)

    # Configure logger
    model_log_dir = os.path.join(log_dir, model_name)
    new_logger = configure(model_log_dir, ["stdout", "csv", "tensorboard"])

    # Create model
    policy_kwargs = dict(net_arch=net_arch)
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=verbose,
        seed=seed,
        device=device,
        policy_kwargs=policy_kwargs,
    )
    model.set_logger(new_logger)

    # Callback for tracking
    reward_callback = RewardTrackingCallback(verbose=0)

    # Train
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=reward_callback,
        progress_bar=True,
    )
    elapsed = time.time() - start_time

    # Save model
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)

    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Model saved to: {model_path}.zip")
    print(f"  Episodes completed: {len(reward_callback.episode_rewards)}")
    if reward_callback.episode_rewards:
        last_100 = reward_callback.episode_rewards[-100:]
        print(f"  Last 100 episodes avg reward: {np.mean(last_100):.3f}")
    print(f"{'='*60}\n")

    env.close()
    return model, reward_callback


# ---------------------------------------------------------------------------
# Quick training convenience
# ---------------------------------------------------------------------------

def train_both_models(
    total_timesteps: int = 500_000,
    save_dir: str = "models",
    log_dir: str = "logs",
    seed: int = 42,
    device: str = "auto",
) -> dict:
    """Train both Model A (Explicit) and Model B (Implicit).

    Returns dict with keys 'model_a', 'model_b', 'callback_a', 'callback_b'.
    """
    results = {}

    # Model A — Explicit (one-hot opponent ID)
    model_a, cb_a = train_model(
        model_name="model_a_explicit",
        use_implicit=False,
        total_timesteps=total_timesteps,
        save_dir=save_dir,
        log_dir=log_dir,
        seed=seed,
        device=device,
    )
    results["model_a"] = model_a
    results["callback_a"] = cb_a

    # Model B — Implicit (rolling stats)
    model_b, cb_b = train_model(
        model_name="model_b_implicit",
        use_implicit=True,
        total_timesteps=total_timesteps,
        save_dir=save_dir,
        log_dir=log_dir,
        seed=seed + 1,  # Different seed for variety
        device=device,
    )
    results["model_b"] = model_b
    results["callback_b"] = cb_b

    return results
