"""
scripts/holdem/train_holdem_f.py — Dedicated Production Training for Texas Hold'em Model F.
The Superhuman Closer ("Endgame Finisher").

Architectural Enhancements:
1. 168-Dimensional Observation Geometry (OBS_DIM_HOLDEM_F):
   - Adds tournament blind level timer, clock urgency, chip leaderboard rank,
     Harrington M-factor & zone classification, heads-up endgame regime flag,
     podium bubble factor, Sklansky-Chubukov push urgency, c-bet opportunity,
     and closer stack dominance ratio.
2. 8-Action Fine-Grained Betting Abstraction (NUM_ACTIONS_HOLDEM_F = 8):
   - Introduces 33% pot micro-cbet and 150% overbet/3-bet sizings.
   - Decoupled Hierarchical Policy seamlessly maps 4 core intentions and 5 geometric sizings.
3. Uncapped Tournament Blind Escalation:
   - Escalates blinds through Level 7 (128x multiplier), forcing heads-up resolution
     and eliminating 150-hand timeout stalls.
4. Strategic Air-Bluff Unshackling:
   - Permits postflop air bets on dry textures against over-folding adversaries.
5. Uncontested Pot Showdown-Avoidance Incentive:
   - +0.015 reward shaping bonus for winning pots before showdown.
6. Co-Evolutionary League Sparring:
   - Self-play against Model E Champion baselines and past Model F league snapshots.
"""

from __future__ import annotations
import os
import sys

# Prevent OpenMP / Intel MKL library collision under SubprocVecEnv
os.environ["MKL_THREADING_LAYER"] = "GNU"

import json
import math
import time
import shutil
import argparse
from typing import Callable, Optional, List, Dict, Any

import numpy as np
import torch
torch.set_num_threads(4)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from core.neural.card_attention import CardAttentionExtractor
import core.neural.card_attention
sys.modules["card_attention"] = core.neural.card_attention

from games.holdem.gym_wrapper import HoldemPokerGymEnv, OBS_DIM_HOLDEM_F
from games.holdem.action_abstraction import NUM_ACTIONS_HOLDEM_F
from rl.holdem_decoupled_actor import HoldemDecoupledActorCriticPolicy
from rl.holdem_auxiliary_belief import HoldemAuxiliaryBeliefHead, HoldemAuxiliaryBeliefCallback
from rl.holdem_league import HoldemLeaguePool
from rl.holdem_curriculum import HoldemCurriculumRewardCallback


def pin_to_pcores(n_cores: int = 12):
    """Pins execution and worker processes to P-cores (cores 0..15 on Intel 13th Gen)."""
    try:
        if hasattr(os, "sched_setaffinity"):
            target_cores = set(range(min(max(n_cores, 12), 16)))
            os.sched_setaffinity(0, target_cores)
            print(f"  [CPU Affinity] Pinned process & workers to P-cores: {sorted(list(target_cores))}")
    except Exception as e:
        print(f"  [CPU Affinity] Notice: Could not set CPU affinity ({e})")


def linear_warmup_cosine_decay(
    initial_lr: float,
    min_lr: float = 1.5e-5,
    warmup_fraction: float = 0.02,
) -> Callable[[float], float]:
    """Single-cycle smooth schedule: linear warmup then monotonic cosine decay to min_lr."""
    def schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        if progress < warmup_fraction:
            alpha = progress / max(1e-6, warmup_fraction)
            return min_lr + alpha * (initial_lr - min_lr)
        else:
            decay_progress = (progress - warmup_fraction) / max(1e-6, 1.0 - warmup_fraction)
            return min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + math.cos(math.pi * decay_progress))

    return schedule


class HoldemEntropyDecayCallback(BaseCallback):
    """Monotonic cosine decay of entropy coefficient from initial_ent to min_ent."""
    def __init__(
        self,
        initial_ent: float = 0.025,
        min_ent: float = 0.005,
        total_timesteps: int = 2_500_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.initial_ent = float(initial_ent)
        self.min_ent = float(min_ent)
        self.total_timesteps = max(1, int(total_timesteps))

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.total_timesteps)
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        self.model.ent_coef = float(self.min_ent + (self.initial_ent - self.min_ent) * decay)
        return True


class HoldemLeagueSnapshotCallback(BaseCallback):
    """Periodically snapshots model checkpoint into the league directory for dynamic self-play."""
    def __init__(
        self,
        league_pool: HoldemLeaguePool,
        snapshot_freq: int = 250_000,
        league_dir: str = "models/holdem/league",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.league_pool = league_pool
        self.snapshot_freq = int(snapshot_freq)
        self.league_dir = league_dir
        self.last_snapshot_step = 0
        os.makedirs(league_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_snapshot_step >= self.snapshot_freq:
            self.last_snapshot_step = self.num_timesteps
            path = os.path.join(self.league_dir, f"model_holdem_f_step_{self.num_timesteps}.zip")
            self.model.save(path)
            self.league_pool.add_snapshot(path)
            if self.verbose > 0:
                print(f"  [Holdem League] Registered new Model F self-play snapshot: {path}")
        return True


class HoldemFSessionCallback(BaseCallback):
    """Tracks tournament rolling win rate and saves the best model checkpoint."""
    def __init__(
        self,
        log_freq: int = 5_000,
        window: int = 100,
        save_best_path: Optional[str] = "models/holdem/model_holdem_f_best.zip",
        primary_save_path: Optional[str] = "models/holdem/model_holdem_f.zip",
        stats_log_path: Optional[str] = "logs/holdem/win_rate_history_f.json",
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
                    f"  [Model F Rollout {self.num_timesteps:,} steps] "
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


def make_holdem_f_env(
    rank: int,
    seed: int = 42,
    league_pool: Optional[HoldemLeaguePool] = None,
    fold_penalty: float = -0.015,
    elimination_penalty: float = 3.14,
    win_bonus: float = 1.50,
    icm_equity_weight: float = 0.75,
    pot_control_bonus: float = 0.025,
    exploit_value_bonus: float = 0.07,
    exploit_steal_bonus: float = 0.04,
    bluff_station_penalty: float = 0.14,
    uncontested_pot_bonus: float = 0.015,
) -> Callable[[], HoldemPokerGymEnv]:
    def _thunk():
        import torch
        torch.set_num_threads(1)
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
            obs_version="f",
            icm_equity_weight=icm_equity_weight,
            pot_control_bonus=pot_control_bonus,
            exploit_value_bonus=exploit_value_bonus,
            exploit_steal_bonus=exploit_steal_bonus,
            bluff_station_penalty=bluff_station_penalty,
            uncontested_pot_bonus=uncontested_pot_bonus,
            league_mode=True,
            league_pool=league_pool,
            rng_seed=env_seed,
        )
        return env

    return _thunk


def run_training_holdem_f(
    total_timesteps: int = 2_500_000,
    n_envs: int = 12,
    seed: int = 42,
    scaffold_steps: Optional[int] = None,
    lr: float = 9.10e-5,
    min_lr: float = 1.82e-5,
    initial_ent: float = 0.025,
    min_ent: float = 0.005,
    clip_range: float = 0.20,
    params_path: Optional[str] = "configs/holdem/best_params_holdem_f.json",
    output_dir: str = "models/holdem",
    league_dir: str = "models/holdem/league",
    anchor_model_path: str = "models/holdem/model_holdem_e_best.zip",
    smoke_test: bool = False,
    model_name: str = "model_holdem_f",
):
    pin_to_pcores(n_envs)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(league_dir, exist_ok=True)

    # Reward shaping defaults
    min_shaping_mult = 0.0
    fold_penalty = -0.015
    elimination_penalty = 3.14
    win_bonus = 1.50
    icm_equity_weight = 0.75
    pot_control_bonus = 0.025
    exploit_value_bonus = 0.07
    exploit_steal_bonus = 0.04
    bluff_station_penalty = 0.14
    uncontested_pot_bonus = 0.015

    # Load parameters if available
    loaded_params = {}
    if params_path and os.path.exists(params_path):
        try:
            with open(params_path, "r", encoding="utf-8") as f:
                loaded_params = json.load(f)
            print(f"  [Config Loaded] Successfully loaded Model F parameters from {params_path}")
            if scaffold_steps is None:
                scaffold_steps = int(loaded_params.get("scaffold_steps", 110_000))
            min_shaping_mult = float(loaded_params.get("min_shaping_mult", min_shaping_mult))
            exploit_value_bonus = float(loaded_params.get("exploit_value_bonus", exploit_value_bonus))
            exploit_steal_bonus = float(loaded_params.get("exploit_steal_bonus", exploit_steal_bonus))
            bluff_station_penalty = float(loaded_params.get("bluff_station_penalty", bluff_station_penalty))
            icm_equity_weight = float(loaded_params.get("icm_equity_weight", icm_equity_weight))
            fold_penalty = float(loaded_params.get("fold_penalty", fold_penalty))
            pot_control_bonus = float(loaded_params.get("pot_control_bonus", pot_control_bonus))
            elimination_penalty = float(loaded_params.get("elimination_penalty", elimination_penalty))
            win_bonus = float(loaded_params.get("win_bonus", win_bonus))
            uncontested_pot_bonus = float(loaded_params.get("uncontested_pot_bonus", uncontested_pot_bonus))
            lr = float(loaded_params.get("learning_rate", lr))
            min_lr = lr * 0.20
            initial_ent = float(loaded_params.get("initial_ent_coef", initial_ent))
            min_ent = float(loaded_params.get("min_ent_coef", min_ent))
            clip_range = float(loaded_params.get("clip_range", clip_range))
        except Exception as e:
            print(f"  [Config Warning] Could not parse {params_path}: {e}")

    if scaffold_steps is None:
        scaffold_steps = 110_000

    print("\n" + "=" * 80)
    print("      CARDSHARK-RL: TEXAS HOLD'EM MODEL F (THE SUPERHUMAN CLOSER)")
    print("=" * 80)
    print(f"  Observation Dimension:     {OBS_DIM_HOLDEM_F} (obs_version='f')")
    print(f"  Action Space:              {NUM_ACTIONS_HOLDEM_F} Actions (Discrete(8) with Micro-Cbet & Overbet)")
    print(f"  Policy Backbone:           HoldemDecoupledActorCriticPolicy + CardAttention")
    print(f"  Curriculum Scaffolding:    {scaffold_steps:,} steps -> Shaping decayed to {min_shaping_mult:.4f}")
    print(f"  Self-Play League:          PFSP via HoldemLeaguePool ({league_dir})")
    print(f"  Anchor Baseline:           {anchor_model_path}")
    print(f"  Total Timesteps:           {total_timesteps:,}")
    print(f"  Parallel Workers:          {n_envs}")
    print(f"  LR Schedule:               Warmup to {lr:.2e} -> Cosine Decay to {min_lr:.2e}")
    print(f"  Entropy Schedule:          {initial_ent:.4f} -> Cosine Decay to {min_ent:.4f}")
    print(f"  Uncontested Steal Bonus:   +{uncontested_pot_bonus:.4f}")
    print(f"  Uncapped Blinds:           True (Up to Level 7: 128x multiplier)")
    print(f"  Air-Bluff Unshackling:     True")
    print("=" * 80 + "\n")

    # If anchor model doesn't exist, fall back to model_holdem_d_best.zip if present
    if not os.path.exists(anchor_model_path):
        fallback = "models/holdem/model_holdem_d_best.zip"
        if os.path.exists(fallback):
            anchor_model_path = fallback

    # Initialize dynamic league pool
    league_pool = HoldemLeaguePool(
        league_dir=league_dir,
        anchor_model_path=anchor_model_path if os.path.exists(anchor_model_path) else None,
        max_snapshots=15,
    )

    env_fns = [
        make_holdem_f_env(
            rank=i,
            seed=seed,
            league_pool=league_pool,
            fold_penalty=fold_penalty,
            elimination_penalty=elimination_penalty,
            win_bonus=win_bonus,
            icm_equity_weight=icm_equity_weight,
            pot_control_bonus=pot_control_bonus,
            exploit_value_bonus=exploit_value_bonus,
            exploit_steal_bonus=exploit_steal_bonus,
            bluff_station_penalty=bluff_station_penalty,
            uncontested_pot_bonus=uncontested_pot_bonus,
        )
        for i in range(n_envs)
    ]

    vec_env = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)

    lr_schedule = linear_warmup_cosine_decay(
        initial_lr=lr,
        min_lr=min_lr,
        warmup_fraction=0.02 if not smoke_test else 0.10,
    )

    policy_kwargs = dict(
        features_extractor_class=CardAttentionExtractor,
        features_extractor_kwargs=dict(
            num_cards=7,
            embed_dim=64,
            num_heads=4,
            pooling="mean",
            card_features_dim=64,
            game_features_dim=104,  # 168 total - 64 card extractor = 104 game features
            slot_features_dim=0,
        ),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    model = MaskablePPO(
        policy=HoldemDecoupledActorCriticPolicy,
        env=vec_env,
        learning_rate=lr_schedule,
        n_steps=512,
        batch_size=256,
        n_epochs=5,
        gamma=0.992,
        gae_lambda=0.92,
        clip_range=clip_range,
        ent_coef=initial_ent,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=1 if smoke_test else 0,
    )

    belief_head = HoldemAuxiliaryBeliefHead(feature_dim=64 + 104, hidden_dim=128)
    belief_callback = HoldemAuxiliaryBeliefCallback(
        belief_head=belief_head,
        lr=lr * 0.8,
        tier_weight=0.5,
        fold_weight=0.5,
        max_batch_size=2048,
        verbose=0,
    )

    curriculum_callback = HoldemCurriculumRewardCallback(
        scaffold_steps=scaffold_steps if not smoke_test else 500,
        total_timesteps=total_timesteps,
        min_shaping_mult=min_shaping_mult,
        log_freq=25_000 if not smoke_test else 500,
        verbose=1,
    )

    entropy_callback = HoldemEntropyDecayCallback(
        initial_ent=initial_ent,
        min_ent=min_ent,
        total_timesteps=total_timesteps,
    )

    league_snapshot_callback = HoldemLeagueSnapshotCallback(
        league_pool=league_pool,
        snapshot_freq=250_000 if not smoke_test else 1000,
        league_dir=league_dir,
    )

    session_callback = HoldemFSessionCallback(
        log_freq=5_000 if not smoke_test else 500,
        window=100 if not smoke_test else 20,
        save_best_path=os.path.join(output_dir, f"{model_name}_best.zip"),
        primary_save_path=os.path.join(output_dir, f"{model_name}.zip"),
        stats_log_path=os.path.join("logs/holdem", f"win_rate_history_{model_name}.json"),
        verbose=1,
    )

    callbacks = CallbackList([
        curriculum_callback,
        entropy_callback,
        belief_callback,
        league_snapshot_callback,
        session_callback,
    ])

    t0 = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
    finally:
        vec_env.close()

    dur = time.time() - t0
    final_path = os.path.join(output_dir, f"{model_name}_final.zip")
    best_path = os.path.join(output_dir, f"{model_name}_best.zip")
    primary_path = os.path.join(output_dir, f"{model_name}.zip")
    model.save(final_path)
    if not os.path.exists(best_path):
        shutil.copyfile(final_path, best_path)
    if not os.path.exists(primary_path):
        shutil.copyfile(final_path, primary_path)
    print(f"\n[{model_name} Complete] Finished {total_timesteps:,} steps in {dur/60:.2f} mins. Saved -> {final_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Texas Hold'em Model F (The Superhuman Closer)")
    parser.add_argument("--timesteps", type=int, default=2_500_000, help="Total timesteps to train")
    parser.add_argument("--n-envs", type=int, default=12, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--scaffold-steps", type=int, default=None, help="Steps for Stage 1 shaping")
    parser.add_argument("--params", type=str, default="configs/holdem/best_params_holdem_f.json", help="Path to best hyperparameters JSON")
    parser.add_argument("--model-name", type=str, default="model_holdem_f", help="Checkpoint model name prefix")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 2000-step smoke test")

    args = parser.parse_args()

    if args.smoke_test:
        run_training_holdem_f(
            total_timesteps=2000,
            n_envs=2,
            seed=args.seed,
            scaffold_steps=500,
            params_path=None,
            smoke_test=True,
            model_name=args.model_name,
        )
    else:
        run_training_holdem_f(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            seed=args.seed,
            scaffold_steps=args.scaffold_steps,
            params_path=args.params,
            smoke_test=False,
            model_name=args.model_name,
        )
