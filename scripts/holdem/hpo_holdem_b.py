"""
scripts/holdem/hpo_holdem_b.py — Dedicated Optuna Hyperparameter Optimization for Texas Hold'em Model B.

Targets:
1. P-Core CPU Affinity Pinning (12 parallel envs pinned to P-cores 0-11).
2. Precise Search Space:
   - fold_penalty: [-0.02, 0.01] (testing fold-reward vs small penalty)
   - elimination_penalty: [0.0, 15.0] (ICM survival incentive)
   - win_bonus: [0.0, 20.0] (championship 1st place incentive)
   - learning_rate: [1.0e-4, 3.5e-4]
   - initial_ent_coef: [0.02, 0.05]
   - min_ent_coef: [0.003, 0.01]
   - clip_range: [0.15, 0.25]
   - min_lr_ratio: [0.20, 0.35]
3. Proven Fixed Transformer Backbone:
   - CardAttentionExtractor(7 cards, embed_dim=32, num_heads=4, pooling="mean", card_features_dim=32, game_features_dim=64)
   - Policy Trunk: [256, 256] MLP
   - PPO dynamics: gamma=0.992, gae_lambda=0.92, n_restart_cycles=6, n_epochs=5
4. Comprehensive CRN Evaluation:
   - Evaluates each trial across 30 Common Random Numbers (CRN) tournament sessions against archetype bots.
   - Measures 1st place win rate, average BB return, elimination rate, and preflop VPIP.
"""

from __future__ import annotations
import os
import sys

# Prevent OpenMP / Intel MKL library collision under SubprocVecEnv
os.environ["MKL_THREADING_LAYER"] = "GNU"

import json
import argparse
import time
from typing import Dict, Any, List

import numpy as np
import torch
torch.set_num_threads(4)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import optuna
from optuna.samplers import TPESampler
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from core.neural.card_attention import CardAttentionExtractor
import core.neural.card_attention
sys.modules["card_attention"] = core.neural.card_attention

from games.holdem.gym_wrapper import HoldemPokerGymEnv
from games.holdem.action_abstraction import (
    A_FOLD, A_CHECK_CALL, A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN,
    STREET_PREFLOP
)
from scripts.holdem.train_holdem_b import (
    make_holdem_b_env,
    cosine_warm_restart_schedule,
    EntropyDecayCallback,
)


def pin_to_pcores(n_cores: int = 12):
    """Pins execution and worker processes to P-cores (cores 0..15 on Intel 13th Gen)."""
    try:
        if hasattr(os, "sched_setaffinity"):
            target_cores = set(range(min(n_cores, 16)))
            os.sched_setaffinity(0, target_cores)
            print(f"  [CPU Affinity] Pinned process & workers to P-cores: {sorted(list(target_cores))}")
    except Exception as e:
        print(f"  [CPU Affinity] Notice: Could not set CPU affinity ({e})")


def objective(
    trial: optuna.Trial,
    timesteps: int = 250_000,
    n_envs: int = 12,
    seed: int = 42,
    eval_sessions: int = 30,
) -> float:
    # 1. Strategic Tournament Incentives
    fold_penalty = trial.suggest_float("fold_penalty", -0.02, 0.01)
    elimination_penalty = trial.suggest_float("elimination_penalty", 0.0, 15.0)
    win_bonus = trial.suggest_float("win_bonus", 0.0, 20.0)

    # 2. PPO Optimization Dynamics
    lr = trial.suggest_float("learning_rate", 1.0e-4, 3.5e-4, log=True)
    initial_ent_coef = trial.suggest_float("initial_ent_coef", 0.02, 0.05)
    min_ent_coef = trial.suggest_float("min_ent_coef", 0.003, 0.01)
    clip_range = trial.suggest_float("clip_range", 0.15, 0.25)
    min_lr_ratio = trial.suggest_float("min_lr_ratio", 0.20, 0.35)
    min_lr = lr * min_lr_ratio

    # 3. Proven Transformer Backbone & Policy Trunk
    embed_dim = 32
    num_heads = 4
    pooling = "mean"
    card_features_dim = 32
    game_features_dim = 64
    net_arch = [256, 256]
    gamma = 0.992
    gae_lambda = 0.92
    n_restart_cycles = 6

    vec_cls = DummyVecEnv if timesteps <= 1000 else SubprocVecEnv
    env = vec_cls([
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

    n_steps = min(512, max(64, timesteps // n_envs))
    batch_size = min(256, max(32, (n_steps * n_envs) // 8))

    model = MaskablePPO(
        policy=MaskableActorCriticPolicy,
        env=env,
        learning_rate=lr_schedule,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=4,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=initial_ent_coef,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )


    callbacks = [
        EntropyDecayCallback(
            initial_ent=initial_ent_coef,
            min_ent=min_ent_coef,
            total_timesteps=timesteps,
        )
    ]

    model.learn(total_timesteps=timesteps, callback=callbacks)

    # 4. Standardized Benchmark CRN Tournament Evaluation
    num_eval_sessions = eval_sessions if timesteps >= 10_000 else 2
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
    total_hands = 0
    vpip_hands = 0

    for sess_i in range(num_eval_sessions):
        eval_seed = 300_000 + sess_i * 777
        obs, info = eval_env.reset(seed=eval_seed)
        sess_reward = 0.0
        done = False
        steps = 0
        last_hands_count = 0
        hero_vpip_hand = False

        while not done and steps < 250:
            street = eval_env.env.street
            mask = eval_env.action_masks()
            act, _ = model.predict(obs, action_masks=mask, deterministic=True)
            act = int(act)

            if street == STREET_PREFLOP:
                if act in (A_CHECK_CALL, A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN):
                    if eval_env.env.seats[0].bet_to_call > 0 or act != A_CHECK_CALL:
                        hero_vpip_hand = True

            obs, r, term, trunc, info = eval_env.step(act)
            sess_reward += r
            done = term or trunc
            steps += 1

            if eval_env.env.hands_played > last_hands_count or done:
                total_hands += 1
                if hero_vpip_hand:
                    vpip_hands += 1
                hero_vpip_hand = False
                last_hands_count = eval_env.env.hands_played

        total_rewards.append(sess_reward)
        if info.get("is_winner", False):
            wins += 1
        if eval_env.env.seats[eval_env.hero_seat].chips <= 0:
            eliminations += 1

    env.close()
    eval_env.close()

    mean_reward = float(np.mean(total_rewards))
    win_rate = (wins / max(1, num_eval_sessions)) * 100.0
    elim_rate = (eliminations / max(1, num_eval_sessions)) * 100.0
    vpip_pct = (vpip_hands / max(1, total_hands)) * 100.0

    # Composite tournament objective: prioritize championship titles and positive return while penalizing bustouts
    score = (win_rate * 5.0) + mean_reward - (elim_rate * 1.5)

    trial.set_user_attr("mean_reward", round(mean_reward, 2))
    trial.set_user_attr("win_rate", round(win_rate, 2))
    trial.set_user_attr("elim_rate", round(elim_rate, 2))
    trial.set_user_attr("vpip", round(vpip_pct, 1))

    print(
        f"  [Trial {trial.number:02d}] Score: {score:6.2f} | Win Rate: {win_rate:4.1f}% | Return: {mean_reward:+6.2f} BBs | "
        f"Elim: {elim_rate:4.1f}% | VPIP: {vpip_pct:4.1f}% | (fp={fold_penalty:+.3f}, elim_pen={elimination_penalty:4.1f}, "
        f"win_bonus={win_bonus:4.1f}, lr={lr:.1e}, ent={initial_ent_coef:.3f}->{min_ent_coef:.3f})"
    )
    return score


def run_hpo_holdem_b(
    n_trials: int = 15,
    timesteps_per_trial: int = 250_000,
    n_envs: int = 12,
    smoke_test: bool = False,
    pin_cores: bool = True,
    output_path: str = "configs/holdem/best_params_holdem_b.json",
    history_path: str = "configs/holdem/hpo_trials_history_b.json",
):
    if pin_cores:
        pin_to_pcores(n_envs)

    if smoke_test:
        n_trials = 2
        timesteps_per_trial = 512
        n_envs = 2
        eval_sessions = 2
        print("\n*** RUNNING TEXAS HOLD'EM MODEL B HPO SMOKE TEST (2 trials, 512 steps) ***\n")
    else:
        eval_sessions = 30
        print(f"=== STARTING TEXAS HOLD'EM MODEL B TARGETED HPO ({n_trials} trials, {timesteps_per_trial:,} steps/trial) ===")
        print(f"  Parallel Envs: {n_envs} (pinned to P-cores)")
        print(f"  Search Space: fold_penalty [-0.02, 0.01], elim_pen [0.0, 15.0], win_bonus [0.0, 20.0]")
        print(f"                learning_rate [1e-4, 3.5e-4], initial_ent [0.02, 0.05], min_ent [0.003, 0.01], clip [0.15, 0.25]")
        print(f"  Fixed: CardAttentionExtractor(7 cards, 32d, 4h, mean) + [256, 256] MLP, gamma=0.992, lambda=0.92")
        print(f"  Evaluation: 30 CRN tournament sessions against archetype bots | Target: {output_path}")

    sampler = TPESampler(multivariate=True, seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    t0 = time.time()
    study.optimize(
        lambda trial: objective(
            trial,
            timesteps=timesteps_per_trial,
            n_envs=n_envs,
            eval_sessions=eval_sessions,
        ),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    elapsed = time.time() - t0

    best = study.best_trial
    print(f"\n=======================================================")
    print(f"[Hold'em Model B HPO Complete] Elapsed: {elapsed/60:.1f} mins")
    print(f"Best Trial #{best.number} | Score: {best.value:.2f}")
    print(f"Best Metrics: Win Rate: {best.user_attrs.get('win_rate')}%, Mean Return: {best.user_attrs.get('mean_reward')} BBs, VPIP: {best.user_attrs.get('vpip')}%")
    print(f"Best Hyperparameters: {json.dumps(best.params, indent=2)}")
    print(f"=======================================================\n")

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

    # Build optimal parameter dictionary for production training
    params_to_save = dict(best.params)
    params_to_save["net_arch"] = [256, 256]
    params_to_save["embed_dim"] = 32
    params_to_save["num_heads"] = 4
    params_to_save["pooling"] = "mean"
    params_to_save["card_features_dim"] = 32
    params_to_save["game_features_dim"] = 64
    params_to_save["gamma"] = 0.992
    params_to_save["gae_lambda"] = 0.92
    params_to_save["n_restart_cycles"] = 6
    params_to_save["n_steps"] = 1024
    params_to_save["batch_size"] = 128
    params_to_save["n_epochs"] = 7

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
    print(f"Saved optimal production parameters to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Texas Hold'em Model B HPO")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--no-pin", action="store_true", help="Disable CPU pinning")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    run_hpo_holdem_b(
        n_trials=args.trials,
        timesteps_per_trial=args.timesteps,
        n_envs=args.n_envs,
        pin_cores=not args.no_pin,
        smoke_test=args.smoke_test,
    )
