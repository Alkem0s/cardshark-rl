"""
train.py — Training pipeline for Model A (Explicit) and Model B (Implicit).

Uses MaskablePPO from sb3-contrib with action masking.
Supports:
  - VecEnv (n_envs parallel environments)
  - Linear LR decay schedule
  - Per-opponent evaluation callback during training
  - Separate hyperparameter sets for Model A and Model B
  - Full PPO params: clip_range, gae_lambda, vf_coef, max_grad_norm
"""

from __future__ import annotations
import os
import time
import numpy as np
from typing import Callable

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_wrapper import DrawPokerGymEnv, make_env, mask_fn
from evaluate import run_tournament, OPP_NAMES


# ---------------------------------------------------------------------------
# Linear LR schedule
# ---------------------------------------------------------------------------

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Return a schedule that linearly decays from initial_value to 0."""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


# ---------------------------------------------------------------------------
# Reward tracking callback
# ---------------------------------------------------------------------------

class RewardTrackingCallback(BaseCallback):
    """Tracks per-episode rewards during training for plotting."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.timestamps: list[int] = []

    def _on_step(self) -> bool:
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
# Per-opponent evaluation callback
# ---------------------------------------------------------------------------

class OpponentEvalCallback(BaseCallback):
    """Periodically evaluates the model against each archetype separately.

    Logs per-opponent BB/100 so you can see when each matchup is learned
    and detect trade-offs between opponent strategies.
    """

    def __init__(
        self,
        use_implicit: bool,
        eval_interval: int = 50_000,
        eval_hands: int = 500,
        seed: int = 0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.use_implicit = use_implicit
        self.eval_interval = eval_interval
        self.eval_hands = eval_hands
        self.seed = seed
        self._last_eval_step = 0
        # History: {opp_name: [(timestep, bb_per_100)]}
        self.eval_history: dict[str, list[tuple[int, float]]] = {n: [] for n in OPP_NAMES}

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step >= self.eval_interval:
            self._last_eval_step = self.num_timesteps
            self._run_eval()
        return True

    def _run_eval(self):
        results = []
        for opp_id, opp_name in enumerate(OPP_NAMES):
            res = run_tournament(
                self.model,
                use_implicit=self.use_implicit,
                opponent_id=opp_id,
                num_hands=self.eval_hands,
                seed=self.seed + opp_id,
            )
            bb = res["bb_per_100"]
            self.eval_history[opp_name].append((self.num_timesteps, bb))
            results.append(f"{opp_name}: {bb:+.1f}")

        if self.verbose >= 1:
            avg = np.mean([v[-1][1] for v in self.eval_history.values() if v])
            print(
                f"  [EvalCallback @{self.num_timesteps:,}] "
                + " | ".join(results)
                + f" | avg={avg:+.1f}"
            )


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
    n_steps: int = 1024,
    batch_size: int = 128,
    n_epochs: int = 10,
    gamma: float = 1.0,
    ent_coef: float = 0.01,
    clip_range: float = 0.2,
    gae_lambda: float = 0.95,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    lr_schedule: str = "linear",  # "linear" or "constant"
    device: str = "auto",
    verbose: int = 1,
    rolling_window: int | None = None,
    opponent_schedule: str | None = None,
    block_size: int = 200,
    hybrid_switch_episodes: int | None = None,
    fold_penalty: float = 0.3,
    steal_bonus: float = 0.2,
    n_envs: int = 1,
    eval_during_training: bool = True,
    eval_interval: int = 50_000,
    eval_hands_callback: int = 500,
) -> tuple[MaskablePPO, RewardTrackingCallback, OpponentEvalCallback | None]:
    """Train a MaskablePPO model on the Draw Poker environment.

    Args:
        model_name: Name for saving (e.g., 'model_a_explicit')
        use_implicit: If True, use implicit (rolling stats) observation.
                      If False, use explicit (one-hot opponent ID).
        total_timesteps: Number of training steps.
        save_dir: Directory to save trained models.
        log_dir: Directory for tensorboard logs.
        seed: Random seed.
        net_arch: MLP architecture (default [256, 256, 256]).
        learning_rate: PPO learning rate (base value for schedule).
        n_steps: Steps per rollout (per env).
        batch_size: Minibatch size.
        n_epochs: PPO epochs per update.
        gamma: Discount factor.
        ent_coef: Entropy coefficient.
        clip_range: PPO clip range.
        gae_lambda: GAE lambda.
        vf_coef: Value function loss coefficient.
        max_grad_norm: Gradient clipping norm.
        lr_schedule: "linear" (decay to 0) or "constant".
        device: 'auto', 'cpu', or 'cuda'.
        n_envs: Number of parallel environments (DummyVecEnv).
        eval_during_training: Whether to run per-opponent eval callbacks.
        eval_interval: Timestep interval between eval callbacks.
        eval_hands_callback: Hands per opponent per eval callback.

    Returns:
        Tuple of (trained model, reward callback, eval callback or None).
    """
    if net_arch is None:
        net_arch = [256, 256, 256]

    # Defaults: hybrid scheduling for implicit (block first, random second half)
    # random scheduling for explicit (already knows opponent identity)
    if rolling_window is None:
        rolling_window = 50 if use_implicit else 50
    if opponent_schedule is None:
        if use_implicit:
            opponent_schedule = "hybrid"
            if hybrid_switch_episodes is None:
                # Switch halfway through training (approx episodes = timesteps/3 steps avg)
                hybrid_switch_episodes = total_timesteps // 3
        else:
            opponent_schedule = "random"

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Ensure batch_size is valid: must not exceed the full rollout buffer
    # (SB3 collects n_steps * n_envs steps per rollout)
    total_rollout = n_steps * n_envs
    if batch_size > total_rollout:
        batch_size = total_rollout
        print(f"  [WARN] batch_size clamped to {batch_size} (= n_steps * n_envs)")

    mode_str = "Implicit" if use_implicit else "Explicit"
    print(f"\n{'='*60}")
    print(f"  Training {model_name} ({mode_str} Modeling)")
    print(f"  Timesteps: {total_timesteps:,} | n_envs: {n_envs}")
    print(f"  Net arch: {net_arch} | LR: {learning_rate:.5f} ({lr_schedule})")
    print(f"  Schedule: {opponent_schedule} | fold_penalty: {fold_penalty} | steal_bonus: {steal_bonus}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Build vectorised environment
    # ------------------------------------------------------------------
    def _make_single_env(env_seed: int):
        """Factory closure: returns a fully-wrapped env for DummyVecEnv."""
        def _init():
            raw = make_env(
                use_implicit=use_implicit,
                opponent_id=None,  # Random rotation
                rolling_window=rolling_window,
                seed=env_seed,
                opponent_schedule=opponent_schedule,
                block_size=block_size,
                hybrid_switch_episodes=hybrid_switch_episodes,
                fold_penalty=fold_penalty,
                steal_bonus=steal_bonus,
            )()  # call the inner closure to get the raw DrawPokerGymEnv
            wrapped = ActionMasker(raw, mask_fn)
            return Monitor(wrapped)
        return _init

    # DummyVecEnv: each factory already returns ActionMasker(Monitor(raw_env))
    # MaskablePPO will call env.env_method("action_masks") which propagates
    # through DummyVecEnv → ActionMasker → DrawPokerGymEnv.action_mask()
    env = DummyVecEnv([_make_single_env(seed + i) for i in range(n_envs)])

    # Configure logger
    model_log_dir = os.path.join(log_dir, model_name)
    try:
        import tensorboard  # noqa: F401
        log_formats = ["stdout", "csv", "tensorboard"]
    except ImportError:
        log_formats = ["stdout", "csv"]
    new_logger = configure(model_log_dir, log_formats)

    # LR schedule
    if lr_schedule == "linear":
        lr_param = linear_schedule(learning_rate)
    else:
        lr_param = learning_rate

    # Create model
    policy_kwargs = dict(net_arch=net_arch)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=lr_param,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        ent_coef=ent_coef,
        clip_range=clip_range,
        gae_lambda=gae_lambda,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=verbose,
        seed=seed,
        device=device,
        policy_kwargs=policy_kwargs,
    )
    model.set_logger(new_logger)

    # Build callbacks
    reward_callback = RewardTrackingCallback(verbose=0)
    callbacks: list[BaseCallback] = [reward_callback]

    eval_callback: OpponentEvalCallback | None = None
    if eval_during_training:
        eval_callback = OpponentEvalCallback(
            use_implicit=use_implicit,
            eval_interval=eval_interval,
            eval_hands=eval_hands_callback,
            seed=seed,
            verbose=1,
        )
        callbacks.append(eval_callback)

    # Train
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
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
    return model, reward_callback, eval_callback


# ---------------------------------------------------------------------------
# Quick training convenience
# ---------------------------------------------------------------------------

def train_both_models(
    total_timesteps: int = 500_000,
    save_dir: str = "models",
    log_dir: str = "logs",
    seed: int = 42,
    device: str = "auto",
    n_envs: int = 1,
    model_a_params: dict | None = None,
    model_b_params: dict | None = None,
    shared_params: dict | None = None,
    eval_during_training: bool = True,
) -> dict:
    """Train both Model A (Explicit) and Model B (Implicit).

    Accepts separate param dicts for each model plus a shared base.
    model_a_params / model_b_params override shared_params.

    Returns dict with keys 'model_a', 'model_b', 'callback_a', 'callback_b',
    'eval_callback_a', 'eval_callback_b'.
    """
    shared = shared_params or {}
    a_kwargs = {**shared, **(model_a_params or {})}
    b_kwargs = {**shared, **(model_b_params or {})}

    results = {}

    # Model A — Explicit (one-hot opponent ID)
    model_a, cb_a, eval_cb_a = train_model(
        model_name="model_a_explicit",
        use_implicit=False,
        total_timesteps=total_timesteps,
        save_dir=save_dir,
        log_dir=log_dir,
        seed=seed,
        device=device,
        n_envs=n_envs,
        eval_during_training=eval_during_training,
        **a_kwargs,
    )
    results["model_a"] = model_a
    results["callback_a"] = cb_a
    results["eval_callback_a"] = eval_cb_a

    # Model B — Implicit (rolling stats)
    model_b, cb_b, eval_cb_b = train_model(
        model_name="model_b_implicit",
        use_implicit=True,
        total_timesteps=total_timesteps,
        save_dir=save_dir,
        log_dir=log_dir,
        seed=seed + 1,  # Different seed for variety (different card deals)
        device=device,
        n_envs=n_envs,
        eval_during_training=eval_during_training,
        **b_kwargs,
    )
    results["model_b"] = model_b
    results["callback_b"] = cb_b
    results["eval_callback_b"] = eval_cb_b

    return results
