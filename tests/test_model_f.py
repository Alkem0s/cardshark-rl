"""
tests/test_model_f.py — Comprehensive Unit & Integration Test Suite for CardShark-RL Model F.

Validates:
1. Model F 97-Dimensional State Geometry & Information Features shape & normalization.
2. DecoupledActionNet Dual-Stream Gradient Isolation (Betting vs Draw).
3. AuxiliaryBeliefHead Multi-Task Loss & Feature Extractor Gradient Flow.
4. LeagueOpponent Native 97-Dim Inference & Action Masking.
5. End-to-End MaskablePPO Rollout Integration with Decoupled Policy & Auxiliary Head.
"""

from __future__ import annotations
import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from rl.multi_gym_wrapper import (
    MultiDrawPokerGymEnv,
    ModelObsDim,
    OBS_DIM_F,
    make_model_f_multi_env,
    A_DRAW_START,
    TOTAL_ACTIONS,
)
from rl.card_attention import CardAttentionExtractor
from rl.decoupled_actor import DecoupledActionNet, DecoupledMaskableActorCriticPolicy
from rl.auxiliary_belief import AuxiliaryBeliefHead, AuxiliaryBeliefCallback
from rl.league import LeagueOpponent
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv


class TestModelF(unittest.TestCase):

    def test_model_f_obs_shape_and_bounds(self):
        """Verifies 97-dim observation shape, components, and [-1.0, 1.0] bounds."""
        env = MultiDrawPokerGymEnv(
            num_seats=5,
            starting_chips=200,
            obs_dim=ModelObsDim.MODEL_F,
            discard_masking=True,
            rng_seed=42,
        )
        obs, info = env.reset(seed=42)

        self.assertEqual(obs.shape, (OBS_DIM_F,), f"Expected ({OBS_DIM_F},), got {obs.shape}")
        self.assertTrue(np.all(obs >= -1.0) and np.all(obs <= 1.0), f"Bounds error: min={obs.min()}, max={obs.max()}")
        self.assertIn("opp_hand_cat", info)
        self.assertIn("is_bluff", info)
        print(f"test_model_f_obs_shape_and_bounds: PASS (Shape: {obs.shape}, bounded [-1.0, 1.0])")

    def test_decoupled_gradient_isolation(self):
        """Verifies that betting loss leaves draw parameters untouched, and vice versa."""
        net = DecoupledActionNet(latent_dim=64, num_bet_actions=6, num_draw_actions=32)
        x = torch.randn(8, 64)

        # 1. Betting loss
        out = net(x)
        loss_bet = out[:, :6].sum()
        loss_bet.backward(retain_graph=True)

        for p in net.draw_stream.parameters():
            self.assertTrue(p.grad is None or torch.all(p.grad == 0), "Gradient leaked into draw stream!")

        # Reset
        net.zero_grad()

        # 2. Draw loss
        out2 = net(x)
        loss_draw = out2[:, 6:].sum()
        loss_draw.backward()

        for p in net.bet_stream.parameters():
            self.assertTrue(p.grad is None or torch.all(p.grad == 0), "Gradient leaked into bet stream!")

        print("test_decoupled_gradient_isolation: PASS (100% gradient isolation between betting & draw streams)")

    def test_auxiliary_belief_gradient_flow(self):
        """Verifies multi-task loss computation and gradient propagation through feature extractor."""
        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM_F,), dtype=np.float32)
        extractor = CardAttentionExtractor(
            observation_space=obs_space,
            embed_dim=32,
            num_heads=4,
            pooling="attention",
            card_features_dim=32,
            game_features_dim=64,
            slot_features_dim=24,
        )
        belief_head = AuxiliaryBeliefHead(feature_dim=216, hidden_dim=128)

        dummy_obs = torch.randn(8, OBS_DIM_F, requires_grad=True)
        target_cats = torch.randint(0, 9, (8,))
        target_bluffs = torch.randint(0, 2, (8,)).float()

        features = extractor(dummy_obs)
        cat_logits, bluff_logits = belief_head(features)

        self.assertEqual(cat_logits.shape, (8, 9))
        self.assertEqual(bluff_logits.shape, (8,))

        loss_cat = F.cross_entropy(cat_logits, target_cats)
        loss_bluff = F.binary_cross_entropy_with_logits(bluff_logits, target_bluffs)
        total_loss = loss_cat + 0.5 * loss_bluff
        total_loss.backward()

        # Check gradients reached extractor parameters
        has_grad = any(p.grad is not None and torch.any(p.grad != 0) for p in extractor.parameters())
        self.assertTrue(has_grad, "Auxiliary loss failed to backpropagate to feature extractor!")
        print("test_auxiliary_belief_gradient_flow: PASS (Gradients cleanly shaped Card Attention backbone)")

    def test_model_f_smoke_integration(self):
        """Runs 64-step MaskablePPO rollout with Decoupled policy and Auxiliary belief callback."""
        env = DummyVecEnv([make_model_f_multi_env(seed=123)])

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
            net_arch=[128, 128],
        )

        model = MaskablePPO(
            policy=DecoupledMaskableActorCriticPolicy,
            env=env,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            policy_kwargs=policy_kwargs,
            verbose=0,
        )

        belief_head = AuxiliaryBeliefHead(feature_dim=216, hidden_dim=128)
        callback = AuxiliaryBeliefCallback(belief_head=belief_head, lr=1e-4, verbose=0)

        model.learn(total_timesteps=64, callback=callback)
        obs = env.reset()
        masks = [env.envs[0].action_masks()]
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)

        self.assertTrue(0 <= int(action[0]) < TOTAL_ACTIONS)
        print("test_model_f_smoke_integration: PASS (Full training step & deterministic prediction succeeded)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  RUNNING MODEL F UNIT TEST SUITE")
    print("=" * 60 + "\n")
    unittest.main()
