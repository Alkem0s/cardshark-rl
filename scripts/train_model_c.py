"""
train_superhuman.py — MaskablePPO Training Pipeline for Model C (Superhuman Autonomous Agent).

Key Capabilities:
1. Multi-Agent League Sparring (Fictitious Play): table seats populated with frozen Model B
   and rolling historical checkpoints of Model C, alongside heuristic archetypes.
2. Intra-Hand Sequence Memory: 87-dimensional observation vector capturing pre-draw and
   post-draw action trajectories and deceptive trap lines.
3. Adaptive Recency Bayesian Profiler: dual-timescale Beta-Binomial tracking with exponential
   forgetting (lambda=0.90) for rapid tilt and gear-shift detection.
4. Scale-invariant reward and state representations across arbitrary stack sizes.
"""

from __future__ import annotations
import os
import json
import argparse
import numpy as np
from typing import Callable, Optional, List

import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.multi_gym_wrapper import make_superhuman_multi_env, multi_mask_fn, SUPERHUMAN_OBS_DIM
from rl.league import LeaguePool


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate decay schedule."""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


class LeagueSnapshotCallback(BaseCallback):
    """Saves rolling Model C checkpoints at regular intervals and adds them to the sparring LeaguePool."""

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
            snapshot_name = f"model_c_step_{self.num_timesteps}.zip"
            snapshot_path = os.path.join(self.save_dir, snapshot_name)
            self.model.save(snapshot_path)
            self.league_pool.add_snapshot(snapshot_path)
            if self.verbose > 0:
                print(f"\n  >>> [LEAGUE SNAPSHOT] Saved & registered new sparring agent: {snapshot_path} <<<")
                print(f"      Active League Pool Size: {len(self.league_pool.checkpoints)} agents\n")
        return True


class SuperhumanSessionCallback(BaseCallback):
    """Logs rolling tournament metrics: survival rate, 1st-place rate, hands survived, and returns."""

    def __init__(
        self,
        league_pool: LeaguePool,
        log_freq: int = 5_000,
        window: int = 100,
        is_freezeout: bool = False,
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
                if self.is_freezeout:
                    print(
                        f"[{self.num_timesteps:,} steps] "
                        f"Sessions: {self.total_sessions:,} | "
                        f"Freezeout Win Rate (1st Place): {rolling_win:.1f}% | "
                        f"Avg Hands: {rolling_hands:.1f} | "
                        f"Mean Return: {avg_return:+.3f} | "
                        f"League Pool: {pool_size} bots"
                    )
                else:
                    print(
                        f"[{self.num_timesteps:,} steps] "
                        f"Sessions: {self.total_sessions:,} | "
                        f"Survival: {rolling_survival:.1f}% | "
                        f"1st Place: {rolling_win:.1f}% | "
                        f"Avg Hands: {rolling_hands:.1f} | "
                        f"Mean Return: {avg_return:+.3f} | "
                        f"League Pool: {pool_size} bots"
                    )

        return True


def create_superhuman_env(
    n_envs: int = 4,
    starting_chips: int = 200,
    blind_escalation_interval: Optional[int] = None,
    randomize_stacks: bool = False,
    max_hands_per_session: int = 150,
    seed: int = 42,
    league_pool: Optional[LeaguePool] = None,
):
    """Creates vectorized environments configured with 87-dim Superhuman observation space and League sparring."""
    def _make(s: int):
        raw = make_superhuman_multi_env(
            num_seats=5,
            starting_chips=starting_chips,
            max_hands_per_session=max_hands_per_session,
            blind_escalation_interval=blind_escalation_interval,
            randomize_stacks=randomize_stacks,
            seed=seed + s * 100,
            league_pool=league_pool,
        )()
        wrapped = ActionMasker(raw, multi_mask_fn)
        return Monitor(wrapped)

    return DummyVecEnv([lambda s=i: _make(s) for i in range(n_envs)])


def train(
    timesteps: int = 1_500_000,
    n_envs: int = 16,
    n_steps: Optional[int] = None,
    starting_chips: int = 200,
    learning_rate: float = 3e-4,
    save_dir: str = "models",
    league_dir: str = "models/league",
    model_name: str = "model_c.zip",
    smoke_test: bool = False,
    seed: int = 42,
    freezeout: bool = True,
    blind_escalation_interval: Optional[int] = None,
    randomize_stacks: bool = True,
    max_hands: int = 200,
    snapshot_interval: int = 250_000,
    params_path: Optional[str] = "best_params_multi.json",
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(league_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)

    if freezeout:
        if blind_escalation_interval is None:
            blind_escalation_interval = 15
        if max_hands == 150:
            max_hands = 200

    # Default architecture: 4-Layer Dense Network (256x256x256x256) for Model C
    net_arch = [256, 256, 256, 256]
    batch_size = 64
    n_epochs = 10
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    ent_coef = 0.008
    vf_coef = 0.5
    max_grad_norm = 0.5
    effective_n_steps = n_steps if n_steps is not None else max(512, 16384 // n_envs)

    # Load tuned parameters if available
    resolved_path = None
    if params_path:
        for candidate in [params_path, os.path.join("configs", params_path), os.path.join("configs", os.path.basename(params_path))]:
            if os.path.exists(candidate):
                resolved_path = candidate
                break

    if resolved_path:
        with open(resolved_path, "r", encoding="utf-8") as f:
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
        print(f"  [Config] Loaded hyperparameters from: {resolved_path}")

    # Initialize League Pool
    league_pool = LeaguePool(
        base_model_path=os.path.join(save_dir, "model_b_multiplayer.zip"),
        league_dir=league_dir,
        neural_opponent_prob=0.50,
    )

    if smoke_test:
        print("=== RUNNING MODEL C SUPERHUMAN SMOKE TEST (500 steps) ===")
        save_path = os.path.join(save_dir, "smoke_test_model_c.zip")
        timesteps = 500
        n_envs = 2
        effective_n_steps = 128
        batch_size = 32
        n_epochs = 3
        snapshot_interval = 250
    else:
        total_rollout = effective_n_steps * n_envs
        print(f"=== STARTING MODEL C SUPERHUMAN TRAINING ({timesteps:,} steps) ===")
        print(f"  Architecture: 4-Layer Dense {net_arch} | Observation Dims: {SUPERHUMAN_OBS_DIM}")
        print(f"  Environments: {n_envs} | Steps/Env: {effective_n_steps} | Rollout Buffer: {total_rollout:,} steps")
        mode_desc = f"Freezeout (Blind Escalation every {blind_escalation_interval} hands, Max hands: {max_hands})" if blind_escalation_interval else f"Cash Table (Static Blinds, Max hands: {max_hands})"
        print(f"  Mode: {mode_desc} | Randomized Stacks: {'Enabled' if randomize_stacks else 'Disabled'}")
        print(f"  League Sparring Pool: {len(league_pool.checkpoints)} active checkpoints (Snapshot every {snapshot_interval:,} steps)")
        print(f"  LR: {learning_rate:.6f} | Batch: {batch_size} | Epochs: {n_epochs} | Gamma: {gamma:.4f} | Ent: {ent_coef:.5f}")

    env = create_superhuman_env(
        n_envs=n_envs,
        starting_chips=starting_chips,
        blind_escalation_interval=blind_escalation_interval,
        randomize_stacks=randomize_stacks,
        max_hands_per_session=max_hands,
        seed=seed,
        league_pool=league_pool,
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
        SuperhumanSessionCallback(
            league_pool=league_pool,
            log_freq=250 if smoke_test else 5_000,
            is_freezeout=bool(blind_escalation_interval),
        ),
        LeagueSnapshotCallback(
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
    print(f"  MODEL C SUPERHUMAN SAVED SUCCESSFULLY TO: {save_path}")
    print(f"  Total League Snapshots Created: {len(league_pool.checkpoints)}")
    print(f"============================================================\n")

    env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CardShark-RL Model C (Superhuman Autonomous Agent)")
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--n-envs", type=int, default=16, help="Parallel CPU environments")
    parser.add_argument("--n-steps", type=int, default=None, help="Steps collected per environment before update")
    parser.add_argument("--chips", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--params-path", type=str, default="configs/best_params_c.json")
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--league-dir", type=str, default="models/league")
    parser.add_argument("--model-name", type=str, default="model_c.zip")
    parser.add_argument("--freezeout", action="store_true", default=True, help="Enable tournament blind escalation")
    parser.add_argument("--no-freezeout", dest="freezeout", action="store_false", help="Disable tournament blind escalation")
    parser.add_argument("--blind-escalation", type=int, default=15)
    parser.add_argument("--randomize-stacks", action="store_true", default=True, help="Randomize starting stack depths")
    parser.add_argument("--no-randomize-stacks", dest="randomize_stacks", action="store_false")
    parser.add_argument("--max-hands", type=int, default=200)
    parser.add_argument("--snapshot-interval", type=int, default=250_000, help="Timesteps between saving league sparring snapshots")
    parser.add_argument("--smoke-test", action="store_true", help="Run 500-step smoke test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        starting_chips=args.chips,
        learning_rate=args.lr,
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
