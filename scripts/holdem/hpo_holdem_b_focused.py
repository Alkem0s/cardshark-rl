"""
scripts/holdem/hpo_holdem_b_focused.py — Focused Hyperparameter Optimization for Texas Hold'em Model B.

Targets:
1. Freezing proven architecture: 7-card Attention (32d, 4 heads, mean pool) + [256, 256] MLP.
2. Searching critical tournament incentives & PPO parameters:
   - fold_penalty: [-0.015, 0.015] (testing neutral/fold-reward vs small penalty)
   - elimination_penalty: [5.0, 25.0] (ICM survival alignment)
   - win_bonus: [10.0, 30.0] (1st place championship incentive)
   - learning_rate: [8.0e-5, 2.2e-4]
   - initial_ent_coef: [0.015, 0.035]
   - clip_range: [0.18, 0.25]
3. Increased timesteps per trial (default: 400,000 steps) to escape local blind-stealing trap.
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from typing import Dict, Any

import numpy as np
import optuna
from optuna.samplers import TPESampler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from core.neural.card_attention import CardAttentionExtractor
import core.neural.card_attention
sys.modules["card_attention"] = core.neural.card_attention

from games.holdem.gym_wrapper import HoldemPokerGymEnv
from scripts.holdem.train_holdem_b import make_holdem_b_env, cosine_warm_restart_schedule


def objective(trial: optuna.Trial, timesteps: int = 400_000, n_envs: int = 16, seed: int = 42) -> float:
    # 1. Critical Tournament Incentives
    fold_penalty = trial.suggest_float("fold_penalty", -0.015, 0.015)
    elimination_penalty = trial.suggest_float("elimination_penalty", 5.0, 25.0)
    win_bonus = trial.suggest_float("win_bonus", 10.0, 30.0)

    # 2. PPO Optimization dynamics
    lr = trial.suggest_float("learning_rate", 8.0e-5, 2.2e-4, log=True)
    min_lr_ratio = trial.suggest_float("min_lr_ratio", 0.20, 0.35)
    min_lr = lr * min_lr_ratio
    clip_range = trial.suggest_float("clip_range", 0.18, 0.25)
    initial_ent_coef = trial.suggest_float("initial_ent_coef", 0.015, 0.035)

    # Fixed proven architecture from previous 25-trial HPO
    n_restart_cycles = 6
    gamma = 0.98838
    gae_lambda = 0.91845
    embed_dim = 32
    num_heads = 4
    pooling = "mean"
    card_features_dim = 32
    game_features_dim = 64
    net_arch = [256, 256]

    if timesteps <= 1000:
        env = DummyVecEnv([
            make_holdem_b_env(
                i,
                seed=seed + trial.number * 100,
                fold_penalty=fold_penalty,
                elimination_penalty=elimination_penalty,
                win_bonus=win_bonus,
            ) for i in range(n_envs)
        ])
    else:
        env = SubprocVecEnv([
            make_holdem_b_env(
                i,
                seed=seed + trial.number * 100,
                fold_penalty=fold_penalty,
                elimination_penalty=elimination_penalty,
                win_bonus=win_bonus,
            ) for i in range(n_envs)
        ])

    lr_schedule = cosine_warm_restart_schedule(
        initial_lr=lr,
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

    # Tournament Benchmark Evaluation across 25 sessions
    num_eval_sessions = 25 if timesteps >= 10_000 else 4
    eval_env = HoldemPokerGymEnv(
        num_seats=5,
        starting_chips=200,
        hero_seat=0,
        max_hands_per_session=100,
        fold_penalty=0.0,
        elimination_penalty=0.0,
        win_bonus=0.0,
        dominated_masking=True,
        obs_version="b",
        rng_seed=seed + 999,
    )

    total_rewards = []
    wins = 0
    eliminations = 0

    for sess_i in range(num_eval_sessions):
        eval_seed = 300_000 + sess_i * 777
        obs, info = eval_env.reset(seed=eval_seed)
        sess_reward = 0.0
        done = False
        steps = 0
        while not done and steps < 200:
            mask = eval_env.action_masks()
            act, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, r, term, trunc, info = eval_env.step(int(act))
            sess_reward += r
            done = term or trunc
            steps += 1
        total_rewards.append(sess_reward)
        if info.get("is_winner", False):
            wins += 1
        if eval_env.env.seats[eval_env.hero_seat].chips <= 0:
            eliminations += 1

    env.close()
    eval_env.close()

    mean_reward = float(np.mean(total_rewards))
    win_rate = (wins / num_eval_sessions) * 100.0
    elim_rate = (eliminations / num_eval_sessions) * 100.0

    # Composite objective: heavily reward 1st place titles and penalize elimination
    score = (win_rate * 5.0) + mean_reward - (elim_rate * 2.0)

    trial.set_user_attr("mean_reward", round(mean_reward, 2))
    trial.set_user_attr("win_rate", round(win_rate, 2))
    trial.set_user_attr("elim_rate", round(elim_rate, 2))

    print(
        f"  [Trial {trial.number:02d}] Score: {score:.2f} | Win Rate: {win_rate:.1f}% | Elim Rate: {elim_rate:.1f}% | "
        f"Mean Return: {mean_reward:+.2f} BBs (fp={fold_penalty:+.3f}, elim_pen={elimination_penalty:.1f}, "
        f"win_bonus={win_bonus:.1f}, lr={lr:.1e}, ent={initial_ent_coef:.3f})"
    )
    return score


def run_focused_hpo(
    n_trials: int = 10,
    timesteps_per_trial: int = 400_000,
    n_envs: int = 16,
    smoke_test: bool = False,
    output_path: str = "configs/holdem/best_params_holdem_b_focused.json",
    history_path: str = "configs/holdem/hpo_trials_history_b_focused.json",
):
    if smoke_test:
        n_trials = 2
        timesteps_per_trial = 512
        n_envs = 2
        print("\n*** RUNNING FOCUSED HOLDEM MODEL B HPO SMOKE TEST (2 trials, 512 steps) ***\n")
    else:
        print(f"=== STARTING FOCUSED TEXAS HOLD'EM MODEL B HPO ({n_trials} trials, {timesteps_per_trial:,} steps/trial) ===")
        print(f"  Fixed Architecture: CardAttentionExtractor(7 cards, 32d, 4h, mean pool) + [256, 256] MLP")
        print(f"  Tuned Parameters: fold_penalty, elimination_penalty, win_bonus, learning_rate, ent_coef, clip_range")
        print(f"  Environments: {n_envs} (SubprocVecEnv) | Target: {output_path}")

    sampler = TPESampler(multivariate=True, seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(
        lambda trial: objective(trial, timesteps=timesteps_per_trial, n_envs=n_envs),
        n_trials=n_trials,
        n_jobs=1,
    )

    print("\n" + "=" * 55)
    print(f"[Focused HPO Complete]")
    best = study.best_trial
    print(f"Best Trial #{best.number} | Score: {best.value:.2f}")
    print(f"Best Hyperparameters: {json.dumps(best.params, indent=2)}")
    print("=" * 55 + "\n")

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
    params_to_save["net_arch"] = [256, 256]
    params_to_save["embed_dim"] = 32
    params_to_save["num_heads"] = 4
    params_to_save["pooling"] = "mean"
    params_to_save["card_features_dim"] = 32
    params_to_save["game_features_dim"] = 64
    params_to_save["gamma"] = 0.98838
    params_to_save["gae_lambda"] = 0.91845
    params_to_save["n_restart_cycles"] = 6

    if "min_lr_ratio" in best.params and "learning_rate" in best.params:
        params_to_save["min_lr"] = best.params["learning_rate"] * best.params["min_lr_ratio"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(params_to_save, f, indent=2)
    print(f"Saved optimal parameters to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Focused Texas Hold'em Model B HPO")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    run_focused_hpo(
        n_trials=args.trials,
        timesteps_per_trial=args.timesteps,
        n_envs=args.n_envs,
        smoke_test=args.smoke_test,
    )
