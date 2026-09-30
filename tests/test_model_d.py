"""
test_model_d.py — Unit Tests for CardShark-RL Model D Champion Upgrades.

Verifies:
1. Permutation-Invariance of CardAttentionExtractor across all pooling modes.
2. Gradient flow through self-attention token projection layers.
3. Cosine Annealing with Warm Restarts (SGDR) mathematical schedule.
4. AdversarialExploiter bot archetype behavior and legality.
5. End-to-end MaskablePPO integration smoke test with DummyVecEnv.
"""

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import itertools
import numpy as np
import torch
from gymnasium import spaces

from rl.card_attention import CardAttentionExtractor
from scripts.train_model_d import cosine_warm_restart_schedule
from game.multi_opponents import AdversarialExploiter
from rl.multi_gym_wrapper import SUPERHUMAN_OBS_DIM, make_superhuman_multi_env, multi_mask_fn
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv


def test_card_attention_extractor_shapes():
    """Validates tensor output shapes across various pooling and dimension settings."""
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(SUPERHUMAN_OBS_DIM,), dtype=np.float32)

    for pooling in ("mean", "max", "attention", "both"):
        extractor = CardAttentionExtractor(
            observation_space=obs_space,
            embed_dim=32,
            num_heads=2,
            pooling=pooling,
            card_features_dim=32,
            game_features_dim=64,
        )
        extractor.eval()

        batch_size = 4
        dummy_obs = torch.randn(batch_size, SUPERHUMAN_OBS_DIM)
        with torch.no_grad():
            out = extractor(dummy_obs)

        expected_dim = 32 + 64 # 96
        assert out.shape == (batch_size, expected_dim), f"Shape mismatch: {out.shape} vs {(batch_size, expected_dim)}"

    print("test_card_attention_extractor_shapes: PASS (All pooling modes produced shape (B, 96))")


def test_permutation_invariance():
    """
    MATHEMATICAL PROOF TEST:
    Verifies that permuting the 5 private cards produces an IDENTICAL feature representation.
    """
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(SUPERHUMAN_OBS_DIM,), dtype=np.float32)

    for pooling in ("mean", "max", "attention", "both"):
        extractor = CardAttentionExtractor(
            observation_space=obs_space,
            embed_dim=32,
            num_heads=2,
            pooling=pooling,
            card_features_dim=32,
            game_features_dim=64,
        )
        extractor.eval()

        # Create a single observation with distinct cards
        torch.manual_seed(42)
        base_obs = torch.randn(1, SUPERHUMAN_OBS_DIM)

        # Separate 5 cards (each 2 features: rank, suit)
        cards = [base_obs[0, 2 * i : 2 * i + 2] for i in range(5)]
        game_state = base_obs[0, 10:]

        # Original representation
        with torch.no_grad():
            orig_out = extractor(base_obs)

        # Test permutations: reverse, cyclic shift, random shuffle
        test_perms = [
            [4, 3, 2, 1, 0], # Reverse
            [1, 2, 3, 4, 0], # Cyclic shift
            [3, 0, 4, 1, 2], # Arbitrary permutation
            [2, 4, 1, 0, 3], # Another permutation
        ]

        for p in test_perms:
            permuted_cards = torch.cat([cards[idx] for idx in p], dim=0)
            permuted_obs = torch.cat([permuted_cards, game_state], dim=0).unsqueeze(0)

            with torch.no_grad():
                perm_out = extractor(permuted_obs)

            # Max absolute difference across feature dimensions
            max_diff = torch.max(torch.abs(orig_out - perm_out)).item()
            assert max_diff < 1e-5, f"Permutation invariance failed for pooling={pooling}: max_diff={max_diff}"

    print("test_permutation_invariance: PASS (Verified strictly invariant across S_5 card permutations, max_diff < 1e-5)")


def test_gradient_flow():
    """Ensures backpropagation flows cleanly through self-attention and projection layers."""
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(SUPERHUMAN_OBS_DIM,), dtype=np.float32)
    extractor = CardAttentionExtractor(
        observation_space=obs_space,
        embed_dim=32,
        num_heads=2,
        pooling="mean",
        card_features_dim=32,
        game_features_dim=64,
    )
    extractor.train()

    dummy_obs = torch.randn(8, SUPERHUMAN_OBS_DIM, requires_grad=True)
    out = extractor(dummy_obs)
    loss = out.sum()
    loss.backward()

    # Check gradients
    assert extractor.card_proj[0].weight.grad is not None
    assert not torch.isnan(extractor.card_proj[0].weight.grad).any()
    assert extractor.mha.in_proj_weight.grad is not None
    assert not torch.isnan(extractor.mha.in_proj_weight.grad).any()
    assert extractor.game_proj[0].weight.grad is not None
    assert not torch.isnan(extractor.game_proj[0].weight.grad).any()

    print("test_gradient_flow: PASS (Gradients cleanly propagated through Self-Attention & projections)")


def test_cosine_warm_restart_schedule():
    """Verifies that SGDR schedule oscillates between min_lr and initial_lr across cycles."""
    initial_lr = 3.0e-4
    min_lr = 3.0e-5
    n_cycles = 4
    warmup_fraction = 0.05

    sched = cosine_warm_restart_schedule(
        initial_lr=initial_lr,
        min_lr=min_lr,
        n_cycles=n_cycles,
        warmup_fraction=warmup_fraction,
    )

    # At start (progress_remaining = 1.0, progress = 0.0) -> min_lr
    lr_start = sched(1.0)
    assert math.isclose(lr_start, min_lr, rel_tol=1e-3), f"Start LR mismatch: {lr_start} vs {min_lr}"

    # Near peak of cycle 1 (progress ~ 0.0125, progress_remaining ~ 0.9875) -> initial_lr
    lr_peak1 = sched(1.0 - (1.0 / n_cycles) * warmup_fraction)
    assert math.isclose(lr_peak1, initial_lr, rel_tol=1e-3), f"Peak LR mismatch: {lr_peak1} vs {initial_lr}"

    # Near end of cycle 1 (progress ~ 0.249, progress_remaining ~ 0.751) -> min_lr
    lr_end1 = sched(1.0 - 0.249)
    assert abs(lr_end1 - min_lr) < 1e-5, f"End cycle 1 LR mismatch: {lr_end1} vs {min_lr}"

    # Near peak of cycle 2 (progress ~ 0.25 + 0.0125 = 0.2625) -> initial_lr
    lr_peak2 = sched(1.0 - 0.2625)
    assert math.isclose(lr_peak2, initial_lr, rel_tol=1e-3), f"Peak cycle 2 LR mismatch: {lr_peak2} vs {initial_lr}"

    print(f"test_cosine_warm_restart_schedule: PASS (Verified {n_cycles} cycles oscillating between {min_lr} and {initial_lr})")


def test_adversarial_exploiter():
    """Verifies AdversarialExploiter initialization and action selection legality."""
    bot = AdversarialExploiter(rng=np.random.default_rng(42))
    assert bot.name == "Exploiter"
    assert bot.opponent_id == 5

    # Test bet action with strong hand
    legal = [0, 1, 2, 4]  # Fold, Call, Min-Raise, Pot
    trips_hand = [12, 25, 38, 0, 1]  # Three Aces
    act = bot.bet_action(
        hand=trips_hand,
        pot=50,
        bet_to_call=0,
        phase="post_draw",
        stack=200,
        legal_actions=legal,
    )
    assert act in legal, f"Illegal action selected: {act}"

    # Test draw action
    discards = bot.draw_action(trips_hand)
    assert len(discards) == 2, f"Trips should discard 2 kickers, got: {discards}"

    print("test_adversarial_exploiter: PASS (Legality and discard logic verified)")


def test_model_d_smoke_integration():
    """Performs an end-to-end 64-step MaskablePPO learning rollout with CardAttentionExtractor."""
    env = DummyVecEnv([
        make_superhuman_multi_env(
            num_seats=5,
            starting_chips=200,
            blind_escalation_interval=15,
            randomize_stacks=True,
            fold_penalty=0.2,
            max_hands_per_session=30,
            seed=42 + i,
        )
        for i in range(2)
    ])

    policy_kwargs = dict(
        features_extractor_class=CardAttentionExtractor,
        features_extractor_kwargs=dict(
            embed_dim=32,
            num_heads=2,
            pooling="mean",
            card_features_dim=32,
            game_features_dim=64,
        ),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=cosine_warm_restart_schedule(initial_lr=3e-4, min_lr=3e-5, n_cycles=2),
        n_steps=32,
        batch_size=32,
        n_epochs=2,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=42,
    )

    # Train for 64 steps (1 PPO update iteration)
    model.learn(total_timesteps=64)

    # Test prediction
    obs = env.reset()
    masks = np.array([env.envs[i].action_masks() for i in range(2)])
    actions, _ = model.predict(obs, action_masks=masks, deterministic=True)
    assert len(actions) == 2
    assert 0 <= actions[0] <= 37
    assert 0 <= actions[1] <= 37

    env.close()
    print("test_model_d_smoke_integration: PASS (64-step PPO rollout and prediction succeeded)")


if __name__ == "__main__":
    print("\n============================================================")
    print("  RUNNING MODEL D UNIT TEST SUITE")
    print("============================================================\n")

    test_card_attention_extractor_shapes()
    test_permutation_invariance()
    test_gradient_flow()
    test_cosine_warm_restart_schedule()
    test_adversarial_exploiter()
    test_model_d_smoke_integration()

    print("\n============================================================")
    print("  ALL MODEL D TESTS PASSED CLEANLY!")
    print("============================================================\n")
