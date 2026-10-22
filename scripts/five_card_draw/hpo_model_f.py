"""
scripts/hpo_model_f.py — Optuna Hyperparameter Optimization for Model F Champion.

Searches:
1. Card Attention architecture: embed_dim, num_heads, pooling, card_features_dim.
2. Decoupled PPO dynamics: learning_rate, clip_range, ent_coef, gamma, gae_lambda.
3. Tournament shaping: fold_penalty, blind_escalation.

Outputs winning trial configuration to configs/best_params_f.json and configs/five_card_draw/best_params_f.json.
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import optuna
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from core.neural.card_attention import CardAttentionExtractor
from core.neural.decoupled_actor import DecoupledMaskableActorCriticPolicy
from rl.multi_gym_wrapper import (
    make_model_f_multi_env,
    MultiDrawPokerGymEnv,
    ModelObsDim,
)
try:
    from scripts.five_card_draw.train_model_f import cosine_warm_restart_schedule
except ImportError:
    from scripts.train_model_f import cosine_warm_restart_schedule


def objective(trial: optuna.Trial, timesteps: int = 5000, n_envs: int = 4, seed: int = 42) -> float:
    # 1. Architecture
    embed_dim = trial.suggest_categorical("embed_dim", [32, 64])
    num_heads = trial.suggest_categorical("num_heads", [2, 4])
    pooling = trial.suggest_categorical("pooling", ["attention", "both", "mean"])
    card_features_dim = trial.suggest_categorical("card_features_dim", [32, 64])

    # 2. PPO dynamics
    lr = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    min_lr_ratio = trial.suggest_float("min_lr_ratio", 0.05, 0.25)
    min_lr = lr * min_lr_ratio
    n_restart_cycles = trial.suggest_int("n_restart_cycles", 3, 7)
    clip_range = trial.suggest_float("clip_range", 0.15, 0.25)
    initial_ent_coef = trial.suggest_float("initial_ent_coef", 0.005, 0.030)
    gamma = trial.suggest_float("gamma", 0.985, 0.998)
    gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.96)

    # 3. Tournament shaping
    fold_penalty = trial.suggest_float("fold_penalty", 0.02, 0.08)

    env = DummyVecEnv([
        make_model_f_multi_env(
            num_seats=5,
            starting_chips=200,
            hero_seat=0,
            max_hands_per_session=50,
            fold_penalty=fold_penalty,
            seed=seed + trial.number * 100 + i * 10,
        )
        for i in range(n_envs)
    ])

    lr_schedule = cosine_warm_restart_schedule(
        initial_lr=lr,
        min_lr=min_lr,
        n_cycles=n_restart_cycles,
    )

    policy_kwargs = dict(
        features_extractor_class=CardAttentionExtractor,
        features_extractor_kwargs=dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            pooling=pooling,
            card_features_dim=card_features_dim,
            game_features_dim=64,
            slot_features_dim=24,
        ),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    model = MaskablePPO(
        policy=DecoupledMaskableActorCriticPolicy,
        env=env,
        learning_rate=lr_schedule,
        n_steps=min(256, timesteps // n_envs),
        batch_size=min(64, (timesteps // n_envs)),
        n_epochs=3,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=initial_ent_coef,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )

    model.learn(total_timesteps=timesteps)

    # Evaluate tournament returns
    eval_fn = make_model_f_multi_env(
        num_seats=5,
        starting_chips=200,
        hero_seat=0,
        max_hands_per_session=50,
        seed=seed + 999,
    )
    eval_env = eval_fn()

    total_rewards = []
    wins = 0
    num_eval_sessions = 5

    for sess_i in range(num_eval_sessions):
        obs, info = eval_env.reset(seed=seed + sess_i * 13)
        sess_reward = 0.0
        done = False
        steps = 0
        while not done and steps < 100:
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
    mean_reward = float(np.mean(total_rewards))
    win_rate = (wins / num_eval_sessions) * 100.0

    score = mean_reward + (win_rate * 2.0)
    return score


def run_hpo_f(
    n_trials: int = 15,
    timesteps_per_trial: int = 5000,
    n_envs: int = 4,
    smoke_test: bool = False,
    output_path: str = "configs/five_card_draw/best_params_f.json",
):
    if smoke_test:
        n_trials = 2
        timesteps_per_trial = 256
        n_envs = 2
        print("\n*** RUNNING MODEL F HPO SMOKE TEST (2 trials, 256 steps) ***\n")

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(
        lambda trial: objective(trial, timesteps=timesteps_per_trial, n_envs=n_envs),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\n[HPO Complete] Best Score: {best.value:.2f}")
    print(f"Best Hyperparameters: {best.params}")

    merged_params = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                merged_params = json.load(f)
        except Exception:
            pass
    merged_params.update(best.params)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_params, f, indent=2)

    print(f"Saved optimal parameters to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model F HPO")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    run_hpo_f(
        n_trials=args.trials,
        timesteps_per_trial=args.timesteps,
        n_envs=args.n_envs,
        smoke_test=args.smoke_test,
    )
