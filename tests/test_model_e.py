"""
tests/test_model_e.py — Comprehensive Unit Tests for CardShark-RL Model E (The Superhuman Crusher).

Validates:
1. Dominated Discard Action Masker across all 5 poker hand categories.
2. Model E 91-Dimensional ICM & Blind-Aware Observation Space structure & normalization.
3. Dynamic Discard Action Masking in MultiDrawPokerGymEnv during Draw Phase.
4. CardAttentionExtractor compatibility with 91-dim input.
5. End-to-end MaskablePPO integration smoke test with DummyVecEnv.
"""

from __future__ import annotations
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from gymnasium import spaces

from rl.discard_masker import get_legal_discard_bitmasks, get_legal_discard_mask
from rl.multi_gym_wrapper import (
    MultiDrawPokerGymEnv,
    MODEL_E_OBS_DIM,
    make_model_e_multi_env,
    multi_mask_fn,
    A_DRAW_START,
    TOTAL_ACTIONS,
)
from rl.card_attention import CardAttentionExtractor
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv


def test_discard_masker_categories():
    """Validates that dominated discard action masking strictly forbids irrational discards."""
    # 1. Full House (A♠ A♥ A♦ K♠ K♥) -> only bitmask 0 (Stand Pat)
    full_house = [0, 13, 26, 12, 25]  # Rank 0 (A) and Rank 12 (K)
    masks_fh = get_legal_discard_bitmasks(full_house)
    assert masks_fh == {0}, f"Full house allowed illegal discards: {masks_fh}"

    # 2. Straight (10, J, Q, K, A) -> only bitmask 0
    straight = [9, 10, 11, 12, 0]
    assert get_legal_discard_bitmasks(straight) == {0}

    # 3. Two Pair (A♠ A♥ K♠ K♥ 5♦) -> 5♦ is at index 4 (bitmask 16)
    two_pair = [0, 13, 12, 25, 4]  # index 4 is 5
    masks_tp = get_legal_discard_bitmasks(two_pair)
    assert masks_tp == {0, 1 << 4}, f"Two pair should only allow stand pat or discard kicker 4, got: {masks_tp}"

    # 4. Three of a Kind (A♠ A♥ A♦ 8♠ 4♥) -> kickers at index 3 and 4
    trips = [0, 13, 26, 7, 3]  # kickers at index 3 and 4
    masks_trips = get_legal_discard_bitmasks(trips)
    # Expected: 0, (1<<3), (1<<4), (1<<3)|(1<<4) = 0, 8, 16, 24
    assert masks_trips == {0, 8, 16, 24}, f"Trips allowed invalid bitmasks: {masks_trips}"

    # 5. One Pair (A♠ A♥ 9♠ 7♦ 2♣) -> pair at 0 and 1, kickers at 2, 3, 4
    one_pair = [0, 13, 8, 6, 1]
    masks_op = get_legal_discard_bitmasks(one_pair)
    # Never allow discarding card 0 or 1!
    for b in masks_op:
        assert (b & 1) == 0, f"One pair discarded card 0: {b}"
        assert (b & 2) == 0, f"One pair discarded card 1: {b}"
    # Must allow standard draw 3: (1<<2)|(1<<3)|(1<<4) = 4 + 8 + 16 = 28
    assert 28 in masks_op, "One pair must include drawing 3 kickers (bitmask 28)"
    assert 0 in masks_op, "One pair must include stand pat"

    print("test_discard_masker_categories: PASS (All hand categories strictly enforce optimal discard sets)")


def test_model_e_obs_shape():
    """Validates that Model E environment produces 91-dim observations within [-1.0, 1.0]."""
    env = MultiDrawPokerGymEnv(
        num_seats=5,
        starting_chips=200,
        model_e_obs=True,
        discard_masking=True,
        rng_seed=42,
    )
    obs, info = env.reset(seed=42)

    assert obs.shape == (MODEL_E_OBS_DIM,), f"Obs shape mismatch: {obs.shape} vs {(MODEL_E_OBS_DIM,)}"
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0), f"Obs outside [-1.0, 1.0]: min={obs.min()}, max={obs.max()}"

    print(f"test_model_e_obs_shape: PASS (Exact shape: {obs.shape}, bounded [-1.0, 1.0])")


def test_model_e_action_masks():
    """Validates action masking during both betting and draw phases."""
    env = MultiDrawPokerGymEnv(
        num_seats=5,
        starting_chips=200,
        model_e_obs=True,
        discard_masking=True,
        rng_seed=42,
    )
    env.reset(seed=42)

    # In betting phase (Pre-draw)
    mask = env.action_masks()
    assert len(mask) == TOTAL_ACTIONS
    assert np.any(mask[:A_DRAW_START]), "Betting phase must have legal betting actions"
    assert not np.any(mask[A_DRAW_START:]), "Betting phase must NOT have legal draw actions"

    # Force draw phase
    env.env.phase = "draw"
    env.env.current_seat = env.hero_seat
    mask_draw = env.action_masks()

    assert not np.any(mask_draw[:A_DRAW_START]), "Draw phase must NOT have legal betting actions"
    assert np.any(mask_draw[A_DRAW_START:]), "Draw phase must have legal draw actions"
    # Verify dominated discard masking reduces total draw actions from 32 down to <= 6
    num_legal_draws = np.sum(mask_draw[A_DRAW_START:])
    assert 1 <= num_legal_draws <= 6, f"Draw actions should be pruned to 1-6, got: {num_legal_draws}"

    print(f"test_model_e_action_masks: PASS (Betting & Draw masks verified, pruned draw actions to {num_legal_draws})")


def test_card_attention_extractor_with_model_e():
    """Validates CardAttentionExtractor forward pass with Model E 91-dim observations."""
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(MODEL_E_OBS_DIM,), dtype=np.float32)

    extractor = CardAttentionExtractor(
        observation_space=obs_space,
        embed_dim=32,
        num_heads=4,
        pooling="attention",
        card_features_dim=32,
        game_features_dim=64,
        slot_features_dim=24,
    )
    extractor.eval()

    batch_size = 4
    dummy_obs = torch.randn(batch_size, MODEL_E_OBS_DIM)
    with torch.no_grad():
        out = extractor(dummy_obs)

    # 32 (global) + 5 * 24 (slots) + 64 (game) = 216
    expected_dim = 32 + (5 * 24) + 64
    assert out.shape == (batch_size, expected_dim), f"Shape mismatch: {out.shape} vs {(batch_size, expected_dim)}"

    print(f"test_card_attention_extractor_with_model_e: PASS (Output shape: {out.shape})")


def test_model_e_smoke_integration():
    """Runs end-to-end 64-step MaskablePPO integration rollout on Model E environment."""
    env = DummyVecEnv([make_model_e_multi_env(seed=123)])

    policy_kwargs = dict(
        features_extractor_class=CardAttentionExtractor,
        features_extractor_kwargs=dict(
            embed_dim=32,
            num_heads=4,
            pooling="attention",
            card_features_dim=32,
            game_features_dim=64,
            slot_features_dim=24,
        ),
        net_arch=[256, 256, 256],
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )

    model.learn(total_timesteps=64)
    obs = env.reset()
    masks = [env.envs[0].action_masks()]
    action, _ = model.predict(obs, action_masks=masks, deterministic=True)
    assert 0 <= int(action[0]) < TOTAL_ACTIONS

    print("test_model_e_smoke_integration: PASS (64-step rollout and prediction succeeded)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  RUNNING MODEL E UNIT TEST SUITE")
    print("=" * 60 + "\n")
    test_discard_masker_categories()
    test_model_e_obs_shape()
    test_model_e_action_masks()
    test_card_attention_extractor_with_model_e()
    test_model_e_smoke_integration()
    print("\n" + "=" * 60)
    print("  ALL MODEL E TESTS PASSED CLEANLY!")
    print("=" * 60 + "\n")
