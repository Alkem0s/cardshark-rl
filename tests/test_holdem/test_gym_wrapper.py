"""
tests/test_holdem/test_gym_wrapper.py — Unit tests for Hold'em Gym wrapper.
"""

import unittest
import numpy as np
from games.holdem.gym_wrapper import HoldemPokerGymEnv, OBS_DIM_HOLDEM


class TestHoldemGymWrapper(unittest.TestCase):
    def setUp(self):
        self.env = HoldemPokerGymEnv(
            num_seats=5,
            starting_chips=200,
            rng_seed=42,
        )

    def test_obs_shape_and_bounds(self):
        obs, info = self.env.reset(seed=42)
        self.assertEqual(obs.shape, (OBS_DIM_HOLDEM,))
        self.assertEqual(obs.shape, (73,))
        self.assertTrue(np.all(obs >= -1.0 - 1e-5), f"Min obs: {np.min(obs)}")
        self.assertTrue(np.all(obs <= 1.0 + 1e-5), f"Max obs: {np.max(obs)}")

    def test_action_mask_validity(self):
        obs, info = self.env.reset(seed=42)
        mask = self.env.action_masks()
        self.assertEqual(len(mask), 6)
        self.assertGreater(np.sum(mask), 0)
        # Check/Call (action 1) is always valid when it is Hero's turn to act
        self.assertEqual(mask[1], 1)

    def test_step_progression(self):
        obs, info = self.env.reset(seed=42)
        steps = 0
        done = False
        while not done and steps < 50:
            mask = self.env.action_masks()
            valid_actions = np.where(mask == 1)[0]
            action = int(np.random.choice(valid_actions))
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.assertTrue(np.all(obs >= -1.0 - 1e-5))
            self.assertTrue(np.all(obs <= 1.0 + 1e-5))
            done = terminated or truncated
            steps += 1
        self.assertGreater(steps, 0)


if __name__ == "__main__":
    unittest.main()
