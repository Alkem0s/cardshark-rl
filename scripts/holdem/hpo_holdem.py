"""
scripts/holdem/hpo_holdem.py — Optuna Hyperparameter Optimization for No-Limit Texas Hold'em.

Features:
1. Joint Multivariate TPE Search with Gaussian Mixture Model.
2. Common Random Numbers (CRN) variance mitigation across 25 evaluation sessions.
3. Complete trial history logging for statistical variance analysis.
4. Safe dictionary merge preserving full architecture parameters.
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import time
from typing import Dict, Any, List

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import optuna
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from games.holdem.gym_wrapper import HoldemPokerGymEnv
from scripts.holdem.train_holdem import make_holdem_env, cosine_warm_restart_schedule


def objective(trial: optuna.Trial, timesteps: int = 250_000, n_envs: int = 8, seed: int = 42) -> float:
    # 1. PPO dynamics
    lr = trial.suggest_float("learning_rate", 8e-5, 3.5e-4, log=True)
    min_lr_ratio = trial.suggest_float("min_lr_ratio", 0.10, 0.30)
    min_lr = lr * min_lr_ratio
    n_restart_cycles = trial.suggest_int("n_restart_cycles", 4, 8)
    clip_range = trial.suggest_float("clip_range", 0.16, 0.24)
    initial_ent_coef = trial.suggest_float("initial_ent_coef", 0.008, 0.022)
    gamma = trial.suggest_float("gamma", 0.985, 0.995)
    gae_lambda = trial.suggest_float("gae_lambda", 0.92, 0.97)

    # 2. Network architecture
    arch_choice = trial.suggest_categorical("net_arch", ["128x128", "256x256", "256x128x64", "128x128x128"])
    if arch_choice == "128x128":
        net_arch = [128, 128]
    elif arch_choice == "256x256":
        net_arch = [256, 256]
    elif arch_choice == "256x128x64":
        net_arch = [256, 128, 64]
    else:
        net_arch = [128, 128, 128]

    # 3. Tournament shaping
    fold_penalty = trial.suggest_float("fold_penalty", 0.005, 0.030)

    env = DummyVecEnv([make_holdem_env(i, seed=seed + trial.number * 100, fold_penalty=fold_penalty) for i in range(n_envs)])

    lr_schedule = cosine_warm_restart_schedule(
        initial_lr=lr,
        min_lr=min_lr,
        n_cycles=n_restart_cycles,
    )

    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, vf=net_arch),
    )

    n_steps = min(1024, max(64, timesteps // n_envs))
    batch_size = min(128, max(32, (n_steps * n_envs) // 8))

    model = MaskablePPO(
        policy=MaskableActorCriticPolicy,
        env=env,
        learning_rate=lr_schedule,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=5,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=initial_ent_coef,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )

    model.learn(total_timesteps=timesteps)

    # Evaluate tournament returns across 25 sessions with Common Random Numbers (CRN)
    num_eval_sessions = 25 if timesteps >= 10_000 else 5
    eval_env = HoldemPokerGymEnv(
        num_seats=5,
        starting_chips=200,
        hero_seat=0,
        max_hands_per_session=50,
        fold_penalty=fold_penalty,
        rng_seed=seed + 999,
    )

    total_rewards = []
    wins = 0

    for sess_i in range(num_eval_sessions):
        # CRN: identical seed sequences for all trials to isolate policy skill from card variance
        eval_seed = 200_000 + sess_i * 777
        obs, info = eval_env.reset(seed=eval_seed)
        sess_reward = 0.0
        done = False
        steps = 0
        while not done and steps < 150:
            mask = eval_env.action_masks()
            act, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, r, term, trunc, info = eval_env.step(int(act))
            sess_reward += r
            done = term or trunc
            steps += 1
        total_rewards.append(sess_reward)
        if info.get("is_winner", False):
            wins += 1

    env.close()
    eval_env.close()

    mean_reward = float(np.mean(total_rewards))
    win_rate = (wins / num_eval_sessions) * 100.0

    score = mean_reward + (win_rate * 2.5)

    trial.set_user_attr("mean_reward", round(mean_reward, 2))
    trial.set_user_attr("win_rate", round(win_rate, 2))
    print(f"  [Trial {trial.number:02d}] Score: {score:.2f} | Win Rate: {win_rate:.1f}% | Mean Reward: {mean_reward:.2f} (arch={arch_choice}, lr={lr:.1e}, ent={initial_ent_coef:.3f}, fold_pen={fold_penalty:.3f})")
    return score


def run_hpo_holdem(
    n_trials: int = 25,
    timesteps_per_trial: int = 250_000,
    n_envs: int = 8,
    smoke_test: bool = False,
    output_path: str = "configs/holdem/best_params_holdem_a.json",
    history_path: str = "configs/holdem/hpo_trials_history.json",
):
    if smoke_test:
        n_trials = 2
        timesteps_per_trial = 256
        n_envs = 2
        print("\n*** RUNNING TEXAS HOLD'EM HPO SMOKE TEST (2 trials, 256 steps) ***\n")
    else:
        print(f"=== STARTING TEXAS HOLD'EM MULTIVARIATE HPO ({n_trials} trials, {timesteps_per_trial:,} steps/trial) ===")
        print(f"  Sampler: TPESampler(multivariate=True, seed=42)")
        print(f"  Environments: {n_envs} | Target: {output_path}")

    sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    t0 = time.time()
    study.optimize(
        lambda trial: objective(trial, timesteps=timesteps_per_trial, n_envs=n_envs),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    elapsed = time.time() - t0

    best = study.best_trial
    print(f"\n=======================================================")
    print(f"[Hold'em HPO Complete] Elapsed: {elapsed/60:.1f} mins")
    print(f"Best Trial #{best.number} | Score: {best.value:.2f}")
    print(f"Best Hyperparameters: {json.dumps(best.params, indent=2)}")
    print(f"=======================================================\n")

    # Record full history
    trials_data = []
    for t in study.trials:
        trials_data.append({
            "trial_id": t.number,
            "value": t.value,
            "state": str(t.state),
            "params": t.params,
            "user_attrs": t.user_attrs,
        })
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(trials_data, f, indent=2)
    print(f"Saved complete trial history to: {history_path}")

    params_to_save = dict(best.params)
    if params_to_save.get("net_arch") == "128x128":
        params_to_save["net_arch"] = [128, 128]
    elif params_to_save.get("net_arch") == "256x256":
        params_to_save["net_arch"] = [256, 256]
    elif params_to_save.get("net_arch") == "256x128x64":
        params_to_save["net_arch"] = [256, 128, 64]
    elif params_to_save.get("net_arch") == "128x128x128":
        params_to_save["net_arch"] = [128, 128, 128]

    if "min_lr_ratio" in best.params and "learning_rate" in best.params:
        params_to_save["min_lr"] = best.params["learning_rate"] * best.params["min_lr_ratio"]

    merged_params = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                merged_params = json.load(f)
        except Exception:
            pass
    merged_params.update(params_to_save)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_params, f, indent=2)
    print(f"Saved optimal parameters to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Texas Hold'em HPO")
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    run_hpo_holdem(
        n_trials=args.trials,
        timesteps_per_trial=args.timesteps,
        n_envs=args.n_envs,
        smoke_test=args.smoke_test,
    )
