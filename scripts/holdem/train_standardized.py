"""
scripts/holdem/train_standardized.py — Standardized Training Pipeline for Texas Hold'em.

Supports all model generations (Model A through Model F) on the unified, bug-free engine:
- Model A: 73-dim obs, flat 6 actions, basic PPO, basic rule masking, static bots.
- Model B: 125-dim obs, flat 6 actions, Bayesian opponent tracking, static bots.
- Model C: 138-dim obs, flat 6 actions, ICM equity, GTO pre-flop masking.
- Model D: 152-dim obs, flat 6 actions, opponent exploiter, exploitative masking.
- Model E: 158-dim obs, decoupled hierarchical policy head, self-play league pool.
- Model F: 168-dim obs, 8-action space (33% cbet, 150% overbet), uncapped blinds, uncontested pot bonus.

All models benefit from:
- Multiprocessing via SubprocVecEnv pinned to P-cores.
- Unified CardAttentionExtractor backbone.
- Two-phase curriculum reward annealing.
- Standardized tournament validation benchmarks.
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
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from core.neural.card_attention import CardAttentionExtractor
import core.neural.card_attention
sys.modules["card_attention"] = core.neural.card_attention

from games.holdem.gym_wrapper import (
    HoldemPokerGymEnv,
    OBS_DIM_HOLDEM_A,
    OBS_DIM_HOLDEM_B,
    OBS_DIM_HOLDEM_C,
    OBS_DIM_HOLDEM_D,
    OBS_DIM_HOLDEM_E,
    OBS_DIM_HOLDEM_F,
)
from games.holdem.action_abstraction import NUM_ACTIONS, NUM_ACTIONS_HOLDEM_F
from games.holdem.opponents import TAGBot, LAGBot, CallingStationBot, RandomHoldemBot
from rl.holdem_decoupled_actor import HoldemDecoupledActorCriticPolicy
from rl.holdem_league import HoldemLeaguePool
from rl.holdem_curriculum import HoldemCurriculumRewardCallback


def get_model_architecture_spec(model_letter: str) -> Dict[str, Any]:
    """Return the architectural specifications for the given model generation."""
    m = model_letter.lower()
    specs = {
        "a": {
            "obs_version": "a",
            "obs_dim": OBS_DIM_HOLDEM_A,
            "num_actions": NUM_ACTIONS,
            "policy_type": "flat",
            "use_league": False,
            "pos_gto_masking": False,
            "exploitative_masking": False,
            "unshackle_bluffs": False,
            "default_icm_weight": 0.0,
            "default_exploit_bonus": 0.0,
            "default_uncontested_bonus": 0.0,
        },
        "b": {
            "obs_version": "b",
            "obs_dim": OBS_DIM_HOLDEM_B,
            "num_actions": NUM_ACTIONS,
            "policy_type": "flat",
            "use_league": False,
            "pos_gto_masking": False,
            "exploitative_masking": False,
            "unshackle_bluffs": False,
            "default_icm_weight": 0.0,
            "default_exploit_bonus": 0.0,
            "default_uncontested_bonus": 0.0,
        },
        "c": {
            "obs_version": "c",
            "obs_dim": OBS_DIM_HOLDEM_C,
            "num_actions": NUM_ACTIONS,
            "policy_type": "flat",
            "use_league": False,
            "pos_gto_masking": True,
            "exploitative_masking": False,
            "unshackle_bluffs": False,
            "default_icm_weight": 0.40,
            "default_exploit_bonus": 0.0,
            "default_uncontested_bonus": 0.0,
        },
        "d": {
            "obs_version": "d",
            "obs_dim": OBS_DIM_HOLDEM_D,
            "num_actions": NUM_ACTIONS,
            "policy_type": "flat",
            "use_league": False,
            "pos_gto_masking": True,
            "exploitative_masking": True,
            "unshackle_bluffs": False,
            "default_icm_weight": 0.50,
            "default_exploit_bonus": 0.08,
            "default_uncontested_bonus": 0.0,
        },
        "e": {
            "obs_version": "e",
            "obs_dim": OBS_DIM_HOLDEM_E,
            "num_actions": NUM_ACTIONS,
            "policy_type": "decoupled",
            "use_league": True,
            "pos_gto_masking": True,
            "exploitative_masking": True,
            "unshackle_bluffs": False,
            "default_icm_weight": 0.67,
            "default_exploit_bonus": 0.10,
            "default_uncontested_bonus": 0.0,
        },
        "f": {
            "obs_version": "f",
            "obs_dim": OBS_DIM_HOLDEM_F,
            "num_actions": NUM_ACTIONS_HOLDEM_F,
            "policy_type": "decoupled",
            "use_league": True,
            "pos_gto_masking": True,
            "exploitative_masking": True,
            "unshackle_bluffs": True,
            "default_icm_weight": 0.67,
            "default_exploit_bonus": 0.10,
            "default_uncontested_bonus": 0.015,
        },
    }
    if m not in specs:
        raise ValueError(f"Unknown model identifier '{model_letter}'. Choose from: a, b, c, d, e, f")
    return specs[m]


def pin_to_pcores(n_cores: int = 12):
    """Pin execution and worker processes to P-cores when supported."""
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
        total_timesteps: int = 250_000,
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
        snapshot_freq: int = 100_000,
        league_dir: str = "models/holdem/league",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.league_pool = league_pool
        self.snapshot_freq = snapshot_freq
        self.league_dir = league_dir
        self.last_snapshot_step = 0
        os.makedirs(self.league_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_snapshot_step >= self.snapshot_freq:
            self.last_snapshot_step = self.num_timesteps
            snapshot_path = os.path.join(self.league_dir, f"snapshot_{self.num_timesteps}.zip")
            self.model.save(snapshot_path)
            self.league_pool.register_checkpoint(snapshot_path)
            if self.verbose > 0:
                print(f"  [League] Snapshotted checkpoint at {self.num_timesteps:,} steps -> {snapshot_path}")
        return True


def make_env_creator(
    rank: int,
    spec: Dict[str, Any],
    params: Dict[str, Any],
    seed: int = 42,
    league_pool: Optional[HoldemLeaguePool] = None,
):
    """Creates a wrapped HoldemPokerGymEnv instance."""
    def _init():
        env = HoldemPokerGymEnv(
            num_seats=5,
            starting_chips=200,
            small_blind=1,
            big_blind=2,
            hero_seat=0,
            max_hands_per_session=150,
            blind_escalation_interval=15,
            obs_version=spec["obs_version"],
            pos_gto_masking=spec["pos_gto_masking"],
            exploitative_masking=spec["exploitative_masking"],
            uncontested_pot_bonus=float(params.get("uncontested_pot_bonus", spec["default_uncontested_bonus"])),
            icm_equity_weight=float(params.get("icm_equity_weight", spec["default_icm_weight"])),
            exploit_value_bonus=float(params.get("exploit_value_bonus", spec["default_exploit_bonus"])),
            exploit_steal_bonus=float(params.get("exploit_steal_bonus", spec["default_exploit_bonus"] * 0.4)),
            bluff_station_penalty=float(params.get("bluff_station_penalty", spec["default_exploit_bonus"] * 1.0)),
            pot_control_bonus=float(params.get("pot_control_bonus", 0.02 if spec["obs_version"] in ("c", "d", "e", "f") else 0.0)),
            fold_penalty=float(params.get("fold_penalty", 0.0)),
            elimination_penalty=float(params.get("elimination_penalty", 1.5 if spec["obs_version"] in ("c", "d", "e", "f") else 0.0)),
            win_bonus=float(params.get("win_bonus", 1.5 if spec["obs_version"] in ("c", "d", "e", "f") else 0.0)),
            curriculum_shaping_mult=1.0,
            league_pool=league_pool if spec["use_league"] else None,
            rng_seed=seed + rank * 1000,
        )
        return ActionMasker(env, lambda e: e.action_masks())

    return _init


def evaluate_model_tournament(
    model: MaskablePPO,
    spec: Dict[str, Any],
    n_sessions: int = 25,
    seed: int = 10000,
) -> Dict[str, Any]:
    """Standardized tournament evaluation against benchmark bot field."""
    env = HoldemPokerGymEnv(
        num_seats=5,
        starting_chips=200,
        small_blind=1,
        big_blind=2,
        hero_seat=0,
        max_hands_per_session=150,
        blind_escalation_interval=15,
        obs_version=spec["obs_version"],
        pos_gto_masking=spec["pos_gto_masking"],
        exploitative_masking=spec["exploitative_masking"],
        uncontested_pot_bonus=spec["default_uncontested_bonus"],
        rng_seed=seed,
    )

    wins = 0
    podiums = 0
    eliminations = 0
    net_chips = []
    finish_ranks = []

    for s_idx in range(n_sessions):
        obs, _ = env.reset(seed=seed + s_idx * 77)
        done = False
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        hero_chips = env.env.seats[0].chips
        chip_delta = hero_chips - 200
        net_chips.append(chip_delta)

        # Rank determination
        alive_seats = sorted([s for s in env.env.seats], key=lambda x: x.chips, reverse=True)
        hero_rank = [s.seat_idx for s in alive_seats].index(0) + 1
        finish_ranks.append(hero_rank)

        if hero_rank == 1:
            wins += 1
        if hero_rank <= 3:
            podiums += 1
        if hero_chips == 0:
            eliminations += 1

    win_rate = float(wins) / float(n_sessions)
    avg_chips = float(np.mean(net_chips))
    score = (win_rate * 100.0) + (avg_chips / 10.0)

    return {
        "n_sessions": n_sessions,
        "wins": wins,
        "win_rate": win_rate,
        "podiums": podiums,
        "podium_rate": float(podiums) / float(n_sessions),
        "eliminations": eliminations,
        "elimination_rate": float(eliminations) / float(n_sessions),
        "avg_chips": avg_chips,
        "avg_rank": float(np.mean(finish_ranks)),
        "score": score,
    }


def train_standardized(
    model_letter: str = "f",
    timesteps: int = 250_000,
    n_envs: int = 12,
    seed: int = 42,
    params_path: Optional[str] = None,
    output_dir: str = "models/holdem",
    output_name: Optional[str] = None,
    smoke_test: bool = False,
    eval_sessions: int = 25,
) -> Dict[str, Any]:
    """Execute standardized training for any Texas Hold'em model generation."""
    spec = get_model_architecture_spec(model_letter)
    print("=" * 80)
    print(f"  CARDSHARK-RL STANDARDIZED TRAINER: MODEL {model_letter.upper()}")
    print(f"  Obs Dims: {spec['obs_dim']} | Actions: {spec['num_actions']} | Policy: {spec['policy_type']}")
    print(f"  League: {spec['use_league']} | Uncapped Blinds: True")
    print("=" * 80)

    if smoke_test:
        timesteps = 2_000
        n_envs = 2
        eval_sessions = 3
        print(f"  [Smoke Test Active] Timesteps: {timesteps}, Envs: {n_envs}")

    # Load parameters
    params: Dict[str, Any] = {}
    if params_path and os.path.exists(params_path):
        with open(params_path, "r") as f:
            params = json.load(f)
        print(f"  [Config] Loaded parameters from: {params_path}")
    else:
        # Check default paths
        default_cfg = os.path.join(PROJECT_ROOT, "configs", "holdem", f"best_params_holdem_{model_letter.lower()}.json")
        if os.path.exists(default_cfg):
            with open(default_cfg, "r") as f:
                params = json.load(f)
            print(f"  [Config] Loaded default champion parameters from: {default_cfg}")
        else:
            print("  [Config] No config found. Using standard default parameters.")

    lr = float(params.get("learning_rate", 1.0e-4 if spec["policy_type"] == "decoupled" else 2.0e-4))
    min_lr = float(params.get("min_lr", 1.5e-5))
    initial_ent = float(params.get("initial_ent_coef", 0.025))
    min_ent = float(params.get("min_ent_coef", 0.005))
    clip_range = float(params.get("clip_range", 0.20))
    scaffold_steps = int(params.get("scaffold_steps", int(timesteps * 0.4)))
    min_shaping_mult = float(params.get("min_shaping_mult", 0.01))

    # CPU Affinity
    pin_to_pcores(n_envs)

    # League Pool
    league_pool = None
    if spec["use_league"]:
        anchor_path = os.path.join(output_dir, "model_holdem_e_best.zip")
        if not os.path.exists(anchor_path):
            anchor_path = os.path.join(output_dir, "model_holdem_d_best.zip")
        league_pool = HoldemLeaguePool(
            league_dir=os.path.join(output_dir, "league"),
            anchor_model_path=anchor_path,
            max_snapshots=10,
        )

    # Environments
    env_fns = [
        make_env_creator(rank=i, spec=spec, params=params, seed=seed, league_pool=league_pool)
        for i in range(n_envs)
    ]
    vec_env = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)

    # Policy Configuration
    if spec["policy_type"] == "decoupled":
        policy_cls = HoldemDecoupledActorCriticPolicy
        policy_kwargs = {
            "features_extractor_class": CardAttentionExtractor,
            "features_extractor_kwargs": {"embed_dim": 64},
        }
    else:
        policy_cls = MaskableActorCriticPolicy
        policy_kwargs = {
            "features_extractor_class": CardAttentionExtractor,
            "features_extractor_kwargs": {"embed_dim": 64},
            "net_arch": dict(pi=[128, 64], vf=[128, 64]),
        }

    lr_schedule = linear_warmup_cosine_decay(initial_lr=lr, min_lr=min_lr)

    n_steps = int(params.get("n_steps", 512))
    batch_size = int(params.get("batch_size", 256))
    n_epochs = int(params.get("n_epochs", 5))

    model = MaskablePPO(
        policy=policy_cls,
        env=vec_env,
        learning_rate=lr_schedule,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=clip_range,
        ent_coef=initial_ent,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=1,
    )

    # Callbacks
    callbacks = [
        HoldemEntropyDecayCallback(initial_ent=initial_ent, min_ent=min_ent, total_timesteps=timesteps),
        HoldemCurriculumRewardCallback(
            scaffold_steps=scaffold_steps,
            total_timesteps=timesteps,
            min_shaping_mult=min_shaping_mult,
        ),
    ]
    if spec["use_league"] and league_pool is not None and not smoke_test:
        callbacks.append(HoldemLeagueSnapshotCallback(league_pool=league_pool, snapshot_freq=max(50_000, timesteps // 5)))

    # Train
    start_time = time.time()
    print(f"\n  [Training] Launching {timesteps:,} timesteps across {n_envs} environments...")
    model.learn(total_timesteps=timesteps, callback=CallbackList(callbacks))
    elapsed_mins = (time.time() - start_time) / 60.0
    print(f"  [Training Complete] Elapsed: {elapsed_mins:.2f} minutes.")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    if not output_name:
        output_name = f"model_holdem_{model_letter.lower()}_standardized.zip"
    save_path = os.path.join(output_dir, output_name)
    model.save(save_path)
    print(f"  [Model Saved] -> {save_path}")

    # Standardized Tournament Evaluation
    print(f"\n  [Evaluation] Benchmarking across {eval_sessions} tournament sessions...")
    eval_results = evaluate_model_tournament(model, spec=spec, n_sessions=eval_sessions, seed=seed + 5555)
    print(f"  Win Rate:     {eval_results['win_rate']*100:.1f}% ({eval_results['wins']}/{eval_sessions})")
    print(f"  Podium Rate:  {eval_results['podium_rate']*100:.1f}%")
    print(f"  Elimination:  {eval_results['elimination_rate']*100:.1f}%")
    print(f"  Avg Net Chip: {eval_results['avg_chips']:+.1f} chips")
    print(f"  Score:        {eval_results['score']:.2f}")

    eval_results["model"] = model_letter.upper()
    eval_results["timesteps"] = timesteps
    eval_results["elapsed_minutes"] = elapsed_mins
    eval_results["saved_path"] = save_path

    # Close environments
    vec_env.close()
    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardized Texas Hold'em Model Trainer")
    parser.add_argument("--model", type=str, default="f", choices=["a", "b", "c", "d", "e", "f"], help="Model generation (a..f)")
    parser.add_argument("--timesteps", type=int, default=250_000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=12, help="Number of parallel environments (P-cores)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--params-path", type=str, default=None, help="Path to hyperparameters JSON")
    parser.add_argument("--output-dir", type=str, default="models/holdem", help="Directory to save model")
    parser.add_argument("--output-name", type=str, default=None, help="Custom filename for output model")
    parser.add_argument("--eval-sessions", type=int, default=25, help="Number of post-training tournament evaluation sessions")
    parser.add_argument("--smoke-test", action="store_true", help="Quick 2000-step smoke test")
    args = parser.parse_args()

    train_standardized(
        model_letter=args.model,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        params_path=args.params_path,
        output_dir=args.output_dir,
        output_name=args.output_name,
        smoke_test=args.smoke_test,
        eval_sessions=args.eval_sessions,
    )
