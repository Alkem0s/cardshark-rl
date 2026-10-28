"""
tests/test_holdem_f.py — Unit and Integration Tests for Texas Hold'em Model F.

Validates:
1. 168-dimensional observation vector generation and strict [-1.0, 1.0] bounds.
2. 8-action discrete betting space (Fold, Call, Min, 33%, 50%, 100%, 150%, Shove).
3. Air-bluff unshackling on turn/river.
4. Uncapped blind escalation up to Level 7 (128x multiplier).
5. Decoupled hierarchical policy architecture with 8-action forward pass.
6. Multi-step tournament rollout stability.
"""

import os
import sys
import unittest
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from games.holdem.action_abstraction import (
    NUM_ACTIONS_HOLDEM_F,
    A_F_FOLD,
    A_F_CHECK_CALL,
    A_F_MIN_RAISE,
    A_F_THIRD_POT,
    A_F_HALF_POT,
    A_F_POT,
    A_F_OVERBET,
    A_F_ALL_IN,
    STREET_TURN,
    STREET_RIVER,
    STREET_FLOP,
)
from games.holdem.env import HoldemPokerEnv
from games.holdem.gym_wrapper import (
    HoldemPokerGymEnv,
    OBS_DIM_HOLDEM_F,
    build_observation_vector_holdem,
)
from rl.holdem_decoupled_actor import HoldemDecoupledActionNet


class TestHoldemModelF(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_observation_dimensions_and_bounds(self):
        """Verify Model F generates exactly 168 bounded features."""
        env = HoldemPokerEnv(
            num_seats=5,
            starting_chips=200,
            num_actions=NUM_ACTIONS_HOLDEM_F,
            uncap_blinds=True,
            unshackle_bluffs=True,
            rng=self.rng,
        )
        env.reset_session()
        obs = build_observation_vector_holdem(env, hero_seat=0, obs_version="f")

        self.assertEqual(obs.shape, (OBS_DIM_HOLDEM_F,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(np.all(obs >= -1.0 - 1e-5), f"Min value violated: {np.min(obs)}")
        self.assertTrue(np.all(obs <= 1.0 + 1e-5), f"Max value violated: {np.max(obs)}")
        self.assertFalse(np.isnan(obs).any(), "NaN found in observation vector")
        self.assertFalse(np.isinf(obs).any(), "Inf found in observation vector")

    def test_eight_action_space_and_masking(self):
        """Verify 8-action space initialization and mask bounds."""
        gym_env = HoldemPokerGymEnv(obs_version="f", rng_seed=42)
        self.assertEqual(gym_env.action_space.n, 8)
        self.assertEqual(gym_env.obs_dim, 168)
        self.assertEqual(gym_env.observation_space.shape, (168,))
        self.assertAlmostEqual(gym_env.uncontested_pot_bonus, 0.015)

        obs, info = gym_env.reset()
        self.assertEqual(obs.shape, (168,))

        mask = gym_env.env.get_action_mask(gym_env.hero_seat)
        self.assertEqual(mask.shape, (8,))
        self.assertTrue(np.all((mask == 0) | (mask == 1)))
        self.assertGreaterEqual(np.sum(mask), 1, "Must have at least one legal action")

    def test_air_bluff_unshackling(self):
        """Verify that when unshackle_bluffs=True, air hands on turn are not forced to check."""
        env = HoldemPokerEnv(
            num_seats=2,
            starting_chips=200,
            num_actions=NUM_ACTIONS_HOLDEM_F,
            uncap_blinds=True,
            unshackle_bluffs=True,
            rng=self.rng,
        )
        env.reset_session()
        p = env.seats[0]
        # Force air hand (7-2 offsuit) on a disconnected dry board
        p.hand = [20, 3]  # 7c, 2d
        env.board = [40, 28, 16, 4]  # Kh, Jc, 9d, 3s
        env.street = STREET_TURN
        p.bet_to_call = 0
        p.chips = 150
        env.pot = 20
        env.current_bet = 0

        legal = env.get_legal_actions(0)
        # With bluffs unshackled, hero is NOT locked into [A_CHECK_CALL] only
        self.assertIn(A_F_CHECK_CALL, legal)
        # Should have raise/bet options available
        raise_actions = [a for a in legal if a >= 2]
        self.assertGreater(len(raise_actions), 0, "Air bluffing should be permitted postflop")

    def test_uncapped_blind_escalation(self):
        """Verify blinds escalate past level 4 up to level 7 (128x multiplier)."""
        env_f = HoldemPokerEnv(
            num_seats=2,
            starting_chips=200,
            num_actions=NUM_ACTIONS_HOLDEM_F,
            uncap_blinds=True,
            blind_escalation_interval=10,
            rng=self.rng,
        )
        env_f.hands_played = 70  # level 7
        env_f.start_hand()
        # Level 7 multiplier is 2^7 = 128 -> big blind = 2 * 128 = 256
        self.assertEqual(env_f.big_blind, 2 * (2 ** 7))

        # Legacy comparison
        env_legacy = HoldemPokerEnv(
            num_seats=2,
            starting_chips=200,
            num_actions=6,
            uncap_blinds=False,
            blind_escalation_interval=10,
            rng=self.rng,
        )
        env_legacy.hands_played = 70
        env_legacy.start_hand()
        # Legacy caps at min(7, 4) = 4 -> 2^4 = 16 -> big blind = 32
        self.assertEqual(env_legacy.big_blind, 2 * 16)

    def test_decoupled_actor_8_action_forward(self):
        """Verify decoupled actor works seamlessly with both 8 and 6 actions."""
        # 8 actions (Model F)
        net_f = HoldemDecoupledActionNet(latent_dim=64, hidden_dim=128, action_dim=8)
        dummy_in = torch.randn(5, 64)
        out_f = net_f(dummy_in)
        self.assertEqual(out_f.shape, (5, 8))

        # 6 actions (Model D/E backward compatibility)
        net_d = HoldemDecoupledActionNet(latent_dim=64, hidden_dim=128, action_dim=6)
        out_d = net_d(dummy_in)
        self.assertEqual(out_d.shape, (5, 6))

    def test_gym_rollout_stability(self):
        """Verify multi-step simulation in gym environment executes without errors."""
        gym_env = HoldemPokerGymEnv(obs_version="f", rng_seed=123)
        obs, _ = gym_env.reset()

        steps = 0
        for _ in range(50):
            mask = gym_env.env.get_action_mask(gym_env.hero_seat)
            legal = np.where(mask == 1)[0]
            if len(legal) == 0:
                action = 1
            else:
                action = int(np.random.choice(legal))

            obs, reward, term, trunc, info = gym_env.step(action)
            self.assertEqual(obs.shape, (168,))
            self.assertTrue(np.all(obs >= -1.0 - 1e-4) and np.all(obs <= 1.0 + 1e-4))
            self.assertFalse(np.isnan(reward))
            steps += 1
            if term or trunc:
                obs, _ = gym_env.reset()

        self.assertEqual(steps, 50)


if __name__ == "__main__":
    unittest.main()
