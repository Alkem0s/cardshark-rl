"""
scripts/holdem/hpo_standardized.py — Standardized Optuna Hyperparameter Optimization for Texas Hold'em.

Key Features:
1. Supports all model generations (Model A through Model F) on the unified engine.
2. Accepts a JSON file as the search space rather than hardcoding parameter bounds.
   Supports both '_min'/'_max' naming and nested dict/list range formats.
3. Automatically maps model specifications (action spaces, observation dimensions, policy classes).
4. Evaluates each trial against standardized tournament bot fields.
5. Continuously records trial history and saves the champion parameter set.
"""

from __future__ import annotations
import os
import sys

os.environ["MKL_THREADING_LAYER"] = "GNU"

import json
import time
import shutil
import argparse
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
torch.set_num_threads(4)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import optuna
from optuna.samplers import TPESampler

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from core.neural.card_attention import CardAttentionExtractor
import core.neural.card_attention
sys.modules["card_attention"] = core.neural.card_attention

from games.holdem.gym_wrapper import HoldemPokerGymEnv
from rl.holdem_decoupled_actor import HoldemDecoupledActorCriticPolicy
from rl.holdem_league import HoldemLeaguePool
from rl.holdem_curriculum import HoldemCurriculumRewardCallback

from scripts.holdem.train_standardized import (
    get_model_architecture_spec,
    pin_to_pcores,
    linear_warmup_cosine_decay,
    HoldemEntropyDecayCallback,
    evaluate_model_tournament,
    make_env_creator,
)


def evaluate_model_heads_up(
    model: MaskablePPO,
    spec: Dict[str, Any],
    n_sessions: int = 30,
    seed: int = 10000,
) -> Dict[str, Any]:
    """Standardized 1v1 heads-up evaluation against benchmark opponents."""
    env = HoldemPokerGymEnv(
        num_seats=5,
        starting_chips=200,
        small_blind=1,
        big_blind=2,
        hero_seat=0,
        max_hands_per_session=150,
        blind_escalation_interval=15,
        heads_up_mode=True,
        obs_version=spec["obs_version"],
        pos_gto_masking=spec["pos_gto_masking"],
        exploitative_masking=spec["exploitative_masking"],
        uncontested_pot_bonus=spec["default_uncontested_bonus"],
        rng_seed=seed,
    )
    wins = 0
    net_chips = []
    hands_played = []
    for s_idx in range(n_sessions):
        obs, _ = env.reset(seed=seed + s_idx * 77)
        done = False
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
        hero_chips = env.env.seats[0].chips
        net_chips.append(hero_chips - 200)
        hands_played.append(env.env.hands_played)
        if hero_chips > 200:
            wins += 1

    win_rate = float(wins) / float(n_sessions)
    avg_chips = float(np.mean(net_chips))
    score = (win_rate * 100.0) + (avg_chips / 5.0)
    return {
        "n_sessions": n_sessions,
        "wins": wins,
        "win_rate": win_rate,
        "avg_chips": avg_chips,
        "avg_hands": float(np.mean(hands_played)),
        "score": score,
    }


def parse_search_space_json(search_space_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Parses a search space JSON file into a standardized parameter definition dict:
    {
        "param_name": {"type": "float"|"int", "low": ..., "high": ..., "log": bool, "step": ...}
    }
    Supports:
    - Format A: {"learning_rate_min": 1e-4, "learning_rate_max": 3e-4, ...}
    - Format B: {"learning_rate": {"low": 1e-4, "high": 3e-4, "log": true}, ...}
    - Format C: {"clip_range": [0.18, 0.24], ...}
    """
    if not os.path.exists(search_space_path):
        raise FileNotFoundError(f"Search space JSON not found: {search_space_path}")

    with open(search_space_path, "r") as f:
        raw_data = json.load(f)

    parsed_space: Dict[str, Dict[str, Any]] = {}

    # Check for Format A (_min / _max pairs)
    min_keys = [k for k in raw_data.keys() if k.endswith("_min")]
    if min_keys:
        for k_min in min_keys:
            param_name = k_min[:-4]
            k_max = f"{param_name}_max"
            if k_max in raw_data:
                low = raw_data[k_min]
                high = raw_data[k_max]
                is_int = isinstance(low, int) and isinstance(high, int) and not isinstance(low, bool)
                is_log = (not is_int and low > 0 and (high / low) >= 5.0 and param_name in ("learning_rate", "min_lr"))
                parsed_space[param_name] = {
                    "type": "int" if is_int else "float",
                    "low": low,
                    "high": high,
                    "log": is_log,
                    "step": 10000 if (is_int and high > 10000) else (1 if is_int else None),
                }

    # Check for Format B / C
    for k, v in raw_data.items():
        if k.endswith("_min") or k.endswith("_max"):
            continue
        if isinstance(v, dict):
            low = v.get("low", v.get("min"))
            high = v.get("high", v.get("max"))
            if low is not None and high is not None:
                p_type = v.get("type", "int" if isinstance(low, int) and isinstance(high, int) else "float")
                log = v.get("log", False)
                step = v.get("step", 1 if p_type == "int" else None)
                parsed_space[k] = {"type": p_type, "low": low, "high": high, "log": log, "step": step}
        elif isinstance(v, (list, tuple)) and len(v) == 2:
            low, high = v
            p_type = "int" if isinstance(low, int) and isinstance(high, int) else "float"
            parsed_space[k] = {"type": p_type, "low": low, "high": high, "log": False, "step": 1 if p_type == "int" else None}

    if not parsed_space:
        raise ValueError(f"Could not parse any parameter boundaries from {search_space_path}")

    return parsed_space


def sample_trial_params(trial: optuna.Trial, parsed_space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Sample hyperparameters using Optuna trial suggestions based on parsed search space."""
    sampled: Dict[str, Any] = {}
    for param_name, cfg in parsed_space.items():
        p_type = cfg["type"]
        low = cfg["low"]
        high = cfg["high"]
        if p_type == "int":
            step = cfg.get("step") or 1
            sampled[param_name] = trial.suggest_int(param_name, int(low), int(high), step=step)
        else:
            log = cfg.get("log", False)
            sampled[param_name] = trial.suggest_float(param_name, float(low), float(high), log=log)
    return sampled


class StandardizedHPO:
    """Orchestrates Bayesian Hyperparameter Optimization over a user-supplied search space JSON."""

    def __init__(
        self,
        model_letter: str = "f",
        search_space_path: Optional[str] = None,
        n_trials: int = 25,
        timesteps_per_trial: int = 250_000,
        n_envs: int = 12,
        eval_sessions: int = 20,
        seed: int = 42,
        smoke_test: bool = False,
    ):
        self.model_letter = model_letter.lower()
        self.spec = get_model_architecture_spec(self.model_letter)
        self.n_trials = 1 if smoke_test else n_trials
        self.timesteps = 2_000 if smoke_test else timesteps_per_trial
        self.n_envs = 2 if smoke_test else n_envs
        self.eval_sessions = 3 if smoke_test else eval_sessions
        self.seed = seed
        self.smoke_test = smoke_test

        # Resolve search space path
        if not search_space_path:
            default_path = os.path.join(PROJECT_ROOT, "configs", "holdem", f"screened_bounds_holdem_{self.model_letter}.json")
            if os.path.exists(default_path):
                search_space_path = default_path
            else:
                raise ValueError(
                    f"No search space path supplied and default {default_path} does not exist. "
                    "Please specify --search-space <path_to_json>."
                )

        self.search_space_path = search_space_path
        self.parsed_space = parse_search_space_json(self.search_space_path)

        # Output paths
        self.results_dir = os.path.join(PROJECT_ROOT, "results", "holdem")
        self.configs_dir = os.path.join(PROJECT_ROOT, "configs", "holdem")
        self.models_dir = os.path.join(PROJECT_ROOT, "models", "holdem")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.configs_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        self.history_file = os.path.join(self.results_dir, f"hpo_history_holdem_{self.model_letter}_standardized.json")
        self.best_params_file = os.path.join(self.configs_dir, f"best_params_holdem_{self.model_letter}_standardized.json")
        self.trials_history: List[Dict[str, Any]] = []

        if os.path.exists(self.history_file) and not smoke_test:
            try:
                with open(self.history_file, "r") as f:
                    self.trials_history = json.load(f)
                print(f"  [Resume] Loaded {len(self.trials_history)} completed prior trials from {self.history_file}")
            except Exception as e:
                print(f"  [Resume Notice] Could not load prior history: {e}")

    def objective(self, trial: optuna.Trial) -> float:
        """Run a single training trial and evaluate its tournament score."""
        trial_idx = len(self.trials_history)
        trial_seed = self.seed + trial_idx * 100
        params = sample_trial_params(trial, self.parsed_space)
        if self.model_letter == "g":
            params["icm_equity_weight"] = 0.0

        print(f"\n{'='*70}")
        print(f"  [TRIAL {trial_idx+1}/{self.n_trials}] Model: {self.model_letter.upper()}")
        print(f"  Params: {json.dumps(params, indent=2)}")
        print(f"{'='*70}")

        lr = float(params.get("learning_rate", 1.0e-4 if self.spec["policy_type"] == "decoupled" else 2.0e-4))
        min_lr = float(params.get("min_lr", lr * 0.15))
        initial_ent = float(params.get("initial_ent_coef", 0.025))
        min_ent = float(params.get("min_ent_coef", 0.005))
        clip_range = float(params.get("clip_range", 0.20))
        scaffold_steps = int(params.get("scaffold_steps", int(self.timesteps * 0.4)))
        min_shaping_mult = float(params.get("min_shaping_mult", 0.01))

        # Setup environments
        league_pool = None
        if self.spec["use_league"]:
            anchor_candidate = self.spec.get("anchor_model_path")
            if anchor_candidate and os.path.exists(anchor_candidate):
                anchor_path = anchor_candidate
            else:
                anchor_path = os.path.join(self.models_dir, "model_holdem_f_champion_2.5m.zip")
                if not os.path.exists(anchor_path):
                    anchor_path = os.path.join(self.models_dir, "model_holdem_e_best.zip")
                    if not os.path.exists(anchor_path):
                        anchor_path = os.path.join(self.models_dir, "model_holdem_d_best.zip")
            league_sub = self.spec.get("league_dir", f"league_hpo_{self.model_letter}")
            league_pool = HoldemLeaguePool(
                league_dir=os.path.join(self.models_dir, league_sub),
                anchor_model_path=anchor_path,
                max_snapshots=5,
            )

        env_fns = [
            make_env_creator(rank=i, spec=self.spec, params=params, seed=trial_seed, league_pool=league_pool)
            for i in range(self.n_envs)
        ]
        vec_env = SubprocVecEnv(env_fns) if self.n_envs > 1 else DummyVecEnv(env_fns)

        # Policy setup
        if self.spec["policy_type"] == "decoupled":
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

        model = MaskablePPO(
            policy=policy_cls,
            env=vec_env,
            learning_rate=lr_schedule,
            n_steps=512,
            batch_size=256,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=clip_range,
            ent_coef=initial_ent,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            seed=trial_seed,
            verbose=0,
        )

        callbacks = [
            HoldemEntropyDecayCallback(initial_ent=initial_ent, min_ent=min_ent, total_timesteps=self.timesteps),
            HoldemCurriculumRewardCallback(
                scaffold_steps=scaffold_steps,
                total_timesteps=self.timesteps,
                min_shaping_mult=min_shaping_mult,
            ),
        ]

        # Train trial
        t_start = time.time()
        model.learn(total_timesteps=self.timesteps, callback=CallbackList(callbacks))
        elapsed_mins = (time.time() - t_start) / 60.0

        # Evaluate trial
        if self.spec.get("heads_up_mode", False):
            eval_metrics = evaluate_model_heads_up(
                model=model,
                spec=self.spec,
                n_sessions=self.eval_sessions,
                seed=trial_seed + 9999,
            )
        else:
            eval_metrics = evaluate_model_tournament(
                model=model,
                spec=self.spec,
                n_sessions=self.eval_sessions,
                seed=trial_seed + 9999,
            )
        score = float(eval_metrics["score"])

        print(f"  [Trial {trial.number+1} Result] Win Rate: {eval_metrics['win_rate']*100:.1f}% | "
              f"Chips: {eval_metrics['avg_chips']:+.1f} | Score: {score:.2f} | Time: {elapsed_mins:.1f}m")

        # Save record
        trial_record = {
            "trial_number": trial.number,
            "params": params,
            "eval_metrics": eval_metrics,
            "score": score,
            "elapsed_minutes": elapsed_mins,
        }
        self.trials_history.append(trial_record)
        with open(self.history_file, "w") as f:
            json.dump(self.trials_history, f, indent=2)

        # Check if best so far
        best_so_far = max(t["score"] for t in self.trials_history)
        if score >= best_so_far:
            print(f"  >>> NEW BEST MODEL! Score: {score:.2f} <<<")
            best_model_path = os.path.join(self.models_dir, f"model_holdem_{self.model_letter}_hpo_best.zip")
            model.save(best_model_path)
            with open(self.best_params_file, "w") as f:
                json.dump(params, f, indent=2)

        vec_env.close()
        return score

    def run(self):
        """Execute the full Optuna study."""
        print("=" * 80)
        print(f"  STANDARDIZED HPO: MODEL {self.model_letter.upper()}")
        print(f"  Search Space File: {self.search_space_path}")
        print(f"  Parameters in Space: {list(self.parsed_space.keys())}")
        print(f"  Trials: {self.n_trials} | Steps/Trial: {self.timesteps:,} | Envs: {self.n_envs}")
        print("=" * 80)

        pin_to_pcores(self.n_envs)

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.seed),
            study_name=f"hpo_holdem_{self.model_letter}_standardized",
        )

        if self.trials_history and not self.smoke_test:
            print(f"  [Resume] Seeding Optuna with {len(self.trials_history)} completed historical trials...")
            for rec in self.trials_history:
                try:
                    trial_params = rec["params"]
                    dist = {}
                    for p_name, p_val in trial_params.items():
                        if p_name in self.parsed_space:
                            cfg = self.parsed_space[p_name]
                            if cfg["type"] == "int":
                                dist[p_name] = optuna.distributions.IntDistribution(
                                    low=int(cfg["low"]), high=int(cfg["high"]), step=cfg.get("step") or 1
                                )
                            else:
                                dist[p_name] = optuna.distributions.FloatDistribution(
                                    low=float(cfg["low"]), high=float(cfg["high"]), log=cfg.get("log", False)
                                )
                    trial = optuna.trial.create_trial(
                        params={k: v for k, v in trial_params.items() if k in dist},
                        distributions=dist,
                        value=float(rec["score"]),
                    )
                    study.add_trial(trial)
                except Exception as e:
                    print(f"  [Resume Notice] Could not seed historical trial: {e}")

        remaining_trials = max(0, self.n_trials - len(self.trials_history))
        print(f"  [Execution Plan] Completed so far: {len(self.trials_history)} | Remaining to run: {remaining_trials}")

        if remaining_trials > 0:
            study.optimize(self.objective, n_trials=remaining_trials)

        print("\n" + "=" * 80)
        print(f"  HPO STUDY COMPLETE FOR MODEL {self.model_letter.upper()}")
        print(f"  Best Trial: #{study.best_trial.number + 1}")
        print(f"  Best Score: {study.best_value:.2f}")
        print(f"  Best Parameters saved to: {self.best_params_file}")
        print(f"  Trial history saved to:   {self.history_file}")
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardized Optuna HPO Runner for Texas Hold'em")
    parser.add_argument("--model", type=str, default="g", choices=["a", "b", "c", "d", "e", "f", "g"], help="Model generation (a..g)")
    parser.add_argument("--search-space", type=str, default=None, help="Path to search space JSON file")
    parser.add_argument("--n-trials", type=int, default=25, help="Number of Optuna trials")
    parser.add_argument("--timesteps-per-trial", type=int, default=250_000, help="Timesteps per trial")
    parser.add_argument("--n-envs", type=int, default=12, help="Number of parallel environments (P-cores)")
    parser.add_argument("--eval-sessions", type=int, default=20, help="Tournament sessions per trial evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke-test", action="store_true", help="Quick 1-trial 2000-step smoke test")
    args = parser.parse_args()

    hpo = StandardizedHPO(
        model_letter=args.model,
        search_space_path=args.search_space,
        n_trials=args.n_trials,
        timesteps_per_trial=args.timesteps_per_trial,
        n_envs=args.n_envs,
        eval_sessions=args.eval_sessions,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )
    hpo.run()
