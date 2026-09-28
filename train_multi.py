"""
train_multi.py — MaskablePPO Training Pipeline for 5-Seat Multi-Player CardShark-RL.

Trains the Model B Bayesian implicit adaptation policy across multi-hand tournament sessions.
Scale-invariant, position-agnostic, and accounts for player elimination.
"""

from __future__ import annotations
import os
import json
import argparse
import numpy as np
from typing import Callable, Optional

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from multi_gym_wrapper import make_multi_env, multi_mask_fn


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate decay schedule."""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


class MultiSessionCallback(BaseCallback):
    """Logs rolling tournament metrics: survival rate, 1st-place rate, hands survived, and returns."""

    def __init__(self, log_freq: int = 5_000, window: int = 100, is_freezeout: bool = False, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.window = window
        self.is_freezeout = is_freezeout

        self.session_returns: list[float] = []
        self.survived_history: list[int] = []       # 1 if survived, 0 if busted
        self.winner_history: list[int] = []         # 1 if 1st place champion, 0 otherwise
        self.hands_history: list[int] = []          # hands played before bust/end
        self.net_chips_history: list[int] = []      # net chip profit
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

                # Keep rolling window
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
                if self.is_freezeout:
                    print(
                        f"[{self.num_timesteps:,} steps] "
                        f"Sessions: {self.total_sessions:,} | "
                        f"Freezeout Win Rate (1st Place): {rolling_win:.1f}% | "
                        f"Avg Hands: {rolling_hands:.1f} | "
                        f"Mean Return: {avg_return:+.3f}"
                    )
                else:
                    print(
                        f"[{self.num_timesteps:,} steps] "
                        f"Sessions: {self.total_sessions:,} | "
                        f"Survival: {rolling_survival:.1f}% | "
                        f"1st Place: {rolling_win:.1f}% | "
                        f"Avg Hands: {rolling_hands:.1f} | "
                        f"Mean Return: {avg_return:+.3f}"
                    )

        return True


def create_env(
    n_envs: int = 4,
    starting_chips: int = 200,
    blind_escalation_interval: Optional[int] = None,
    randomize_stacks: bool = False,
    max_hands_per_session: int = 150,
    seed: int = 42,
):
    """Creates vectorized environments with ActionMasker and Monitor."""
    def _make(s: int):
        raw = make_multi_env(
            num_seats=5,
            starting_chips=starting_chips,
            max_hands_per_session=max_hands_per_session,
            blind_escalation_interval=blind_escalation_interval,
            randomize_stacks=randomize_stacks,
            seed=seed + s * 100,
        )()
        wrapped = ActionMasker(raw, multi_mask_fn)
        return Monitor(wrapped)

    return DummyVecEnv([lambda s=i: _make(s) for i in range(n_envs)])


def train(
    timesteps: int = 500_000,
    n_envs: int = 4,
    n_steps: Optional[int] = None,
    starting_chips: int = 200,
    learning_rate: float = 3e-4,
    save_dir: str = "models",
    model_name: str = "model_b_multiplayer.zip",
    smoke_test: bool = False,
    seed: int = 42,
    freezeout: bool = False,
    blind_escalation_interval: Optional[int] = None,
    randomize_stacks: bool = False,
    max_hands: int = 150,
    params_path: Optional[str] = None,
):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)

    # Configure tournament freezeout mode defaults
    if freezeout:
        if blind_escalation_interval is None:
            blind_escalation_interval = 15
        if max_hands == 150:
            max_hands = 200

    # Hyperparameter defaults
    net_arch = [256, 256, 256]
    batch_size = 64
    n_epochs = 10
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    ent_coef = 0.008
    vf_coef = 0.5
    max_grad_norm = 0.5
    effective_n_steps = n_steps if n_steps is not None else max(512, 16384 // n_envs)

    # Load tuned parameters from JSON if provided (overriding defaults before printing banner)
    if params_path and os.path.exists(params_path):
        with open(params_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        learning_rate = cfg.get("learning_rate", learning_rate)
        if n_steps is None and "n_steps" in cfg:
            effective_n_steps = cfg["n_steps"]
        batch_size = cfg.get("batch_size", batch_size)
        n_epochs = cfg.get("n_epochs", n_epochs)
        gamma = cfg.get("gamma", gamma)
        gae_lambda = cfg.get("gae_lambda", gae_lambda)
        clip_range = cfg.get("clip_range", clip_range)
        ent_coef = cfg.get("ent_coef", ent_coef)
        vf_coef = cfg.get("vf_coef", vf_coef)
        max_grad_norm = cfg.get("max_grad_norm", max_grad_norm)
        if "blind_escalation" in cfg and blind_escalation_interval is None:
            blind_escalation_interval = cfg["blind_escalation"]
        if "net_arch" in cfg:
            net_arch = cfg["net_arch"]

    if smoke_test:
        print("=== RUNNING MULTI-PLAYER SMOKE TEST (500 steps) ===")
        save_path = os.path.join(save_dir, "smoke_test_model.zip")
        timesteps = 500
        n_envs = 2
        effective_n_steps = 128
        batch_size = 32
        n_epochs = 3
    else:
        total_rollout = effective_n_steps * n_envs
        print(f"=== STARTING MULTI-PLAYER TRAINING ({timesteps:,} steps) ===")
        print(f"  Environments: {n_envs} | Steps/Env: {effective_n_steps} | Rollout Buffer: {total_rollout:,} steps")
        mode_desc = f"Freezeout (Blind Escalation every {blind_escalation_interval} hands, Max hands: {max_hands})" if blind_escalation_interval else f"Cash Table (Static Blinds, Max hands: {max_hands})"
        print(f"  Mode: {mode_desc} | Randomized Stacks: {'Enabled' if randomize_stacks else 'Disabled'}")
        if params_path and os.path.exists(params_path):
            print(f"  Loaded Tuned Config: {params_path} | Architecture: {net_arch}")
            print(f"  LR: {learning_rate:.6f} | Batch: {batch_size} | Epochs: {n_epochs} | Gamma: {gamma:.4f} | Ent: {ent_coef:.5f}")

    env = create_env(
        n_envs=n_envs,
        starting_chips=starting_chips,
        blind_escalation_interval=blind_escalation_interval,
        randomize_stacks=randomize_stacks,
        max_hands_per_session=max_hands,
        seed=seed,
    )

    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, vf=net_arch),
    )

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=linear_schedule(learning_rate),
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
        MultiSessionCallback(
            log_freq=250 if smoke_test else 5_000,
            is_freezeout=bool(blind_escalation_interval),
        )
    ]

    model.learn(
        total_timesteps=timesteps,
        callback=CallbackList(callbacks),
        progress_bar=False,
    )

    model.save(save_path)
    print(f"Model saved successfully to: {save_path}")
    env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Player CardShark-RL Model B")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel CPU environments")
    parser.add_argument("--n-steps", type=int, default=None, help="Steps collected per environment before each PPO update")
    parser.add_argument("--chips", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--params-path", type=str, default=None, help="Path to tuned hyperparameters JSON (e.g. best_params_multi.json)")
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--model-name", type=str, default="model_b_multiplayer.zip")
    parser.add_argument("--freezeout", action="store_true", help="Escalate blinds periodically to guarantee 100% elimination down to 1 winner")
    parser.add_argument("--blind-escalation", type=int, default=None, help="Hands between blind doubling (defaults to 15 when --freezeout is set)")
    parser.add_argument("--randomize-stacks", action="store_true", help="Randomize starting stack depths across players (short, medium, deep)")
    parser.add_argument("--max-hands", type=int, default=150, help="Max hands per session before truncation")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        starting_chips=args.chips,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        model_name=args.model_name,
        smoke_test=args.smoke_test,
        seed=args.seed,
        freezeout=args.freezeout,
        blind_escalation_interval=args.blind_escalation,
        randomize_stacks=args.randomize_stacks,
        max_hands=args.max_hands,
        params_path=args.params_path,
    )
