"""
scripts/holdem/train_holdem_d.py — Training Pipeline for Texas Hold'em Model D (Targeted Adaptive Exploiter).

Features:
1. 152-dimensional observation space:
   - 7 cards (2 hole + 5 board) via 7-card CardAttentionExtractor (14 floats)
   - 17-feature Board Texture & Draws Engine
   - 32-feature Dual-Timescale Bayesian Opponent Tracking
   - 3-feature ICM Stack Geometry
   - 6-feature Preflop GTO Positional Metrics
   - 4-feature Postflop Pot Control & SPR Metrics
   - 3-feature Malmuth-Harville ICM Equities
   - 14-feature Targeted Adversary Exploitation Vector (archetype probabilities, bluff/value/steal directives, fold equity)
2. Decoupled Hierarchical Policy (HoldemDecoupledActorCriticPolicy):
   - Core Action Stream: Fold (0), Call (1), Raise Base (2), All-In (5)
   - Bet Sizing Stream: Min-Raise (2), Half-Pot (3), Full-Pot (4)
   - Eliminates flat-logit competition and bet-sizing starvation.
3. Auxiliary Supervised Belief Head (HoldemAuxiliaryBeliefHead & Callback):
   - Multi-task backpropagation directly through Card Attention backbone predicting villain archetype, hand strength tier, and fold probability.
4. Adaptive Exploitative Action Masking (AEAM):
   - Anti-Station Shield (no betting air into CallingStation)
   - Anti-Station Max-Value (forces Half-Pot/Pot/Shove with monsters)
   - TAG Blind Steal (unmasks open-raises from BTN/CO)
   - LAG Counter-Attack (unmasks 3-bets/shoves against wide LAG opens)
5. Targeted Exploitative Reward Shaping:
   - exploit_value_bonus (+0.05 BB)
   - exploit_steal_bonus (+0.03 BB)
   - bluff_station_penalty (-0.10 BB)
6. Isolated checkpointing to models/holdem/model_holdem_d.zip.
7. P-Core CPU Affinity Pinning (12 parallel envs pinned to P-cores 0-11).
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
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from core.neural.card_attention import CardAttentionExtractor
import core.neural.card_attention
sys.modules["card_attention"] = core.neural.card_attention

from games.holdem.gym_wrapper import HoldemPokerGymEnv, OBS_DIM_HOLDEM_D
from rl.holdem_decoupled_actor import HoldemDecoupledActorCriticPolicy
from rl.holdem_auxiliary_belief import HoldemAuxiliaryBeliefHead, HoldemAuxiliaryBeliefCallback


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
    n_cycles: int = 6,
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
        initial_ent: float = 0.030,
        min_ent: float = 0.008,
        total_timesteps: int = 2_500_000,
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


class HoldemDSessionCallback(BaseCallback):
    def __init__(
        self,
        log_freq: int = 5_000,
        window: int = 100,
        save_best_path: Optional[str] = "models/holdem/model_holdem_d_best.zip",
        primary_save_path: Optional[str] = "models/holdem/model_holdem_d.zip",
        stats_log_path: Optional[str] = "logs/holdem/win_rate_history_d.json",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.window = window
        self.save_best_path = save_best_path
        self.primary_save_path = primary_save_path
        self.stats_log_path = stats_log_path
        self.session_outcomes: List[int] = []
        self.best_win_rate = 0.0
        self.last_log_step = 0
        self.win_rate_history: List[Dict[str, Any]] = []

        if stats_log_path:
            os.makedirs(os.path.dirname(stats_log_path), exist_ok=True)
            if os.path.exists(stats_log_path):
                try:
                    with open(stats_log_path, "r") as f:
                        data = json.load(f)
                        self.win_rate_history = data.get("history", [])
                        self.best_win_rate = data.get("best_win_rate", 0.0)
                except Exception:
                    pass

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for i, info in enumerate(infos):
            is_done = dones[i] if i < len(dones) else False
            if is_done and "is_winner" in info:
                self.session_outcomes.append(1 if info["is_winner"] else 0)

        if len(self.session_outcomes) > 2000:
            self.session_outcomes = self.session_outcomes[-self.window * 2 :]

        if self.num_timesteps - self.last_log_step >= self.log_freq:
            self.last_log_step = self.num_timesteps
            recent = self.session_outcomes[-self.window :]
            if recent:
                win_rate = float(np.mean(recent) * 100.0)
                n_sessions = len(recent)

                entry = {
                    "step": self.num_timesteps,
                    "win_rate": win_rate,
                    "window_size": n_sessions,
                    "ent_coef": getattr(self.model, "ent_coef", 0.0),
                }
                self.win_rate_history.append(entry)

                if self.stats_log_path:
                    try:
                        with open(self.stats_log_path, "w") as f:
                            json.dump(
                                {
                                    "best_win_rate": max(self.best_win_rate, win_rate),
                                    "history": self.win_rate_history,
                                },
                                f,
                                indent=2,
                            )
                    except Exception:
                        pass

                is_new_best = win_rate > self.best_win_rate and n_sessions >= min(30, self.window)
                marker = " [NEW BEST]" if is_new_best else ""
                print(
                    f"  [Model D Rollout {self.num_timesteps:,} steps] "
                    f"Rolling Win Rate: {win_rate:5.1f}% (over {n_sessions} sessions) | "
                    f"Best: {self.best_win_rate:5.1f}% | ent_coef: {self.model.ent_coef:.5f}{marker}"
                )

                if is_new_best:
                    self.best_win_rate = win_rate
                    if self.save_best_path:
                        os.makedirs(os.path.dirname(self.save_best_path), exist_ok=True)
                        self.model.save(self.save_best_path)
                    if self.primary_save_path:
                        self.model.save(self.primary_save_path)
        return True


def make_holdem_d_env(
    rank: int,
    seed: int = 42,
    fold_penalty: float = -0.008,
    elimination_penalty: float = 5.5,
    win_bonus: float = 2.5,
    icm_equity_weight: float = 0.95,
    pot_control_bonus: float = 0.02,
    exploit_value_bonus: float = 0.05,
    exploit_steal_bonus: float = 0.03,
    bluff_station_penalty: float = 0.10,
    league_mode: bool = True,
    model_b_path: str = "models/holdem/model_holdem_b.zip",
) -> Callable[[], HoldemPokerGymEnv]:
    def _thunk():
        env_seed = seed + rank * 1000
        env = HoldemPokerGymEnv(
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
            pos_gto_masking=True,
            exploitative_masking=True,
            obs_version="d",
            icm_equity_weight=icm_equity_weight,
            pot_control_bonus=pot_control_bonus,
            exploit_value_bonus=exploit_value_bonus,
            exploit_steal_bonus=exploit_steal_bonus,
            bluff_station_penalty=bluff_station_penalty,
            league_mode=league_mode,
            model_b_path=model_b_path,
            rng_seed=env_seed,
        )
        return ActionMasker(env, lambda e: e.action_masks())
    return _thunk


def train_holdem_d(
    timesteps: int = 2_500_000,
    n_envs: int = 12,
    seed: int = 42,
    smoke_test: bool = False,
    resume_path: Optional[str] = None,
    save_path: str = "models/holdem/model_holdem_d.zip",
    best_save_path: str = "models/holdem/model_holdem_d_best.zip",
    final_save_path: str = "models/holdem/model_holdem_d_final.zip",
    stats_path: str = "logs/holdem/win_rate_history_d.json",
    params_path: Optional[str] = "configs/holdem/best_params_holdem_d.json",
    pin_cores: bool = True,
):
    if pin_cores and not smoke_test:
        pin_to_pcores(n_envs)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)

    # Default Hyperparameters
    learning_rate = 1.2e-4
    min_lr = 3.6e-5
    n_restart_cycles = 6
    initial_ent_coef = 0.030
    min_ent_coef = 0.008
    n_steps = 256
    batch_size = 256
    n_epochs = 5
    gamma = 0.992
    gae_lambda = 0.92
    clip_range = 0.20
    fold_penalty = -0.008
    elimination_penalty = 5.5
    win_bonus = 2.5
    icm_equity_weight = 0.95
    pot_control_bonus = 0.02
    exploit_value_bonus = 0.05
    exploit_steal_bonus = 0.03
    bluff_station_penalty = 0.10
    league_mode = True

    embed_dim = 32
    num_heads = 4
    pooling = "mean"
    card_features_dim = 32
    game_features_dim = 96
    net_arch = [256, 256]

    resolved_path = None
    if params_path:
        candidates = [
            params_path,
            os.path.join(PROJECT_ROOT, params_path),
            os.path.join(PROJECT_ROOT, "configs", params_path),
        ]
        for c in candidates:
            if os.path.exists(c):
                resolved_path = c
                break

    if resolved_path:
        try:
            with open(resolved_path, "r") as f:
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
            elimination_penalty = params.get("elimination_penalty", elimination_penalty)
            win_bonus = params.get("win_bonus", win_bonus)
            icm_equity_weight = params.get("icm_equity_weight", icm_equity_weight)
            pot_control_bonus = params.get("pot_control_bonus", pot_control_bonus)
            exploit_value_bonus = params.get("exploit_value_bonus", exploit_value_bonus)
            exploit_steal_bonus = params.get("exploit_steal_bonus", exploit_steal_bonus)
            bluff_station_penalty = params.get("bluff_station_penalty", bluff_station_penalty)
            league_mode = params.get("league_mode", league_mode)
            embed_dim = params.get("embed_dim", embed_dim)
            num_heads = params.get("num_heads", num_heads)
            pooling = params.get("pooling", pooling)
            card_features_dim = params.get("card_features_dim", card_features_dim)
            game_features_dim = params.get("game_features_dim", game_features_dim)
            net_arch = params.get("net_arch", net_arch)
            print(f"  [Config] Loaded Hold'em Model D params from: {resolved_path}")
        except Exception as e:
            print(f"  [Config] Warning: Could not parse {resolved_path} ({e}), using defaults.")

    if smoke_test:
        timesteps = 500
        n_envs = 2
        effective_n_steps = 64
        batch_size = 32
        n_epochs = 2
        vec_env_cls = DummyVecEnv
        save_path = "models/holdem/model_holdem_d_smoke.zip"
        best_save_path = "models/holdem/model_holdem_d_smoke_best.zip"
        final_save_path = "models/holdem/model_holdem_d_smoke_final.zip"
        stats_path = "logs/holdem/win_rate_history_d_smoke.json"
        print("\n*** RUNNING HOLDEM MODEL D IN SMOKE-TEST MODE (500 steps) ***\n")
    else:
        effective_n_steps = n_steps
        vec_env_cls = SubprocVecEnv
        print(f"=== STARTING NO-LIMIT TEXAS HOLD'EM MODEL D TRAINING ({timesteps:,} steps) ===")
        print(f"  Observation Space: {OBS_DIM_HOLDEM_D} dims (Model D Exploiter + AEAM)")
        print(f"  Backbone: CardAttentionExtractor(7 cards, dim={embed_dim}, heads={num_heads}, pool={pooling})")
        print(f"  Policy: HoldemDecoupledActorCriticPolicy (Core 4 + Sizing 3)")
        print(f"  Auxiliary Belief: Supervised representation learning (Archetype, Tier, Fold)")
        print(f"  Environments: {n_envs} (SubprocVecEnv) | Steps/Env: {effective_n_steps}")
        print(f"  Exploit Bonuses: Val={exploit_value_bonus}, Steal={exploit_steal_bonus}, BluffPen={bluff_station_penalty}")
        print(f"  League Mode: {league_mode} (sparring vs Model B)")

    env = vec_env_cls([
        make_holdem_d_env(
            i,
            seed=seed,
            fold_penalty=fold_penalty,
            elimination_penalty=elimination_penalty,
            win_bonus=win_bonus,
            icm_equity_weight=icm_equity_weight,
            pot_control_bonus=pot_control_bonus,
            exploit_value_bonus=exploit_value_bonus,
            exploit_steal_bonus=exploit_steal_bonus,
            bluff_station_penalty=bluff_station_penalty,
            league_mode=league_mode,
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
            policy=HoldemDecoupledActorCriticPolicy,
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

    # Initialize Auxiliary Belief Head
    total_feat_dim = card_features_dim + game_features_dim
    belief_head = HoldemAuxiliaryBeliefHead(feature_dim=total_feat_dim, hidden_dim=128)
    belief_callback = HoldemAuxiliaryBeliefCallback(
        belief_head=belief_head,
        lr=learning_rate * 0.8,
        tier_weight=0.5,
        fold_weight=0.5,
        max_batch_size=2048,
        verbose=1 if smoke_test else 0,
    )

    session_callback = HoldemDSessionCallback(
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
        belief_callback,
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
    print(f"\n[Training Finished] Elapsed: {elapsed/60.0:.2f} mins. Saving final model to: {final_save_path}")
    model.save(final_save_path)
    if not os.path.exists(save_path):
        shutil.copyfile(final_save_path, save_path)
    if not os.path.exists(best_save_path):
        shutil.copyfile(final_save_path, best_save_path)

    env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Texas Hold'em Model D (Targeted Exploiter)")
    parser.add_argument("--timesteps", type=int, default=2_500_000, help="Total timesteps to train")
    parser.add_argument("--envs", type=int, default=12, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke-test", action="store_true", help="Quick 500-step smoke test")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--save-path", type=str, default="models/holdem/model_holdem_d.zip")
    parser.add_argument("--best-path", type=str, default="models/holdem/model_holdem_d_best.zip")
    parser.add_argument("--final-path", type=str, default="models/holdem/model_holdem_d_final.zip")
    parser.add_argument("--stats-path", type=str, default="logs/holdem/win_rate_history_d.json")
    parser.add_argument("--params", type=str, default="configs/holdem/best_params_holdem_d.json")
    parser.add_argument("--no-pin", action="store_true", help="Disable CPU P-core pinning")
    args = parser.parse_args()

    train_holdem_d(
        timesteps=args.timesteps,
        n_envs=args.envs,
        seed=args.seed,
        smoke_test=args.smoke_test,
        resume_path=args.resume,
        save_path=args.save_path,
        best_save_path=args.best_path,
        final_save_path=args.final_path,
        stats_path=args.stats_path,
        params_path=args.params,
        pin_cores=not args.no_pin,
    )
