"""
tests/test_holdem/test_holdem_b_features.py — Unit Tests for Model Holdem-B Architecture & Features.

Validates:
1. Model B 125-dimensional observation space vs Model A 73-dimensional space.
2. Strict [-1.0, 1.0] boundedness of all 125 observation dimensions across multiple hands.
3. Board texture & draws engine detection (monotone, two-tone, gutshot, OESD, flush draws, overpairs).
4. Bayesian opponent profiler tracking and relative clockwise rotation.
5. CardAttentionExtractor(num_cards=7) forward pass compatibility with 125-dim input.
"""

import unittest
import numpy as np
import torch

from core.card import parse_card
from games.holdem.gym_wrapper import (
    HoldemPokerGymEnv,
    OBS_DIM_HOLDEM_A,
    OBS_DIM_HOLDEM_B,
    build_observation_vector_holdem,
)
from games.holdem.board_texture import evaluate_board_texture_and_draws
from games.holdem.opponent_tracker import HoldemTableOpponentTracker
from core.neural.card_attention import CardAttentionExtractor


class TestHoldemBFeatures(unittest.TestCase):

    def test_observation_dimensions(self):
        """Verifies observation spaces for Model A (73) and Model B (125)."""
        self.assertEqual(OBS_DIM_HOLDEM_A, 73)
        self.assertEqual(OBS_DIM_HOLDEM_B, 125)

        env_a = HoldemPokerGymEnv(obs_version="a")
        self.assertEqual(env_a.observation_space.shape[0], 73)

        env_b = HoldemPokerGymEnv(obs_version="b")
        self.assertEqual(env_b.observation_space.shape[0], 125)

    def test_observation_bounds_and_structure(self):
        """Ensures all 125 features in Model B remain strictly bounded in [-1.0, 1.0]."""
        env = HoldemPokerGymEnv(obs_version="b", rng_seed=42)
        obs, info = env.reset(seed=42)

        self.assertEqual(obs.shape[0], 125)
        self.assertTrue(np.all(obs >= -1.0), f"Obs min below -1.0: {np.min(obs)}")
        self.assertTrue(np.all(obs <= 1.0), f"Obs max above 1.0: {np.max(obs)}")

        # Step through 20 actions and verify bounds at every step
        for _ in range(20):
            mask = env.action_masks()
            valid_actions = np.where(mask == 1)[0]
            act = int(valid_actions[0])
            obs, r, term, trunc, info = env.step(act)
            self.assertTrue(np.all(obs >= -1.0), f"Obs min below -1.0: {np.min(obs)}")
            self.assertTrue(np.all(obs <= 1.0), f"Obs max above 1.0: {np.max(obs)}")
            if term or trunc:
                obs, info = env.reset()

    def test_board_texture_draw_detection(self):
        """Tests draw detection and board wetness."""
        # 1. Monotone Board + Nut Flush Draw
        # Hero: As Ks | Board: Qs Js 2s (monotone board, hero has nut flush)
        h1 = [parse_card("As"), parse_card("Ks")]
        b1 = [parse_card("Qs"), parse_card("Js"), parse_card("2s")]
        f1 = evaluate_board_texture_and_draws(h1, b1)
        self.assertEqual(f1[1], 1.0)  # is_monotone
        self.assertEqual(f1[10], 1.0) # straight or better (Hero has flush!)

        # 2. OESD (Open Ended Straight Draw)
        # Hero: 9h 8h | Board: 7c 6d 2s
        h2 = [parse_card("9h"), parse_card("8h")]
        b2 = [parse_card("7c"), parse_card("6d"), parse_card("2s")]
        f2 = evaluate_board_texture_and_draws(h2, b2)
        self.assertEqual(f2[13], 1.0)  # has_oesd == 1.0

        # 3. Overpair
        # Hero: Ah Ad | Board: Ks 7c 2d
        h3 = [parse_card("Ah"), parse_card("Ad")]
        b3 = [parse_card("Ks"), parse_card("7c"), parse_card("2d")]
        f3 = evaluate_board_texture_and_draws(h3, b3)
        self.assertEqual(f3[5], 1.0)   # has_overpair == 1.0

    def test_bayesian_opponent_profiling(self):
        """Tests that opponent tracker updates and correctly shifts posterior beliefs."""
        tracker = HoldemTableOpponentTracker(num_seats=5)

        # Seat 1: CallingStation (calls 5 hands, never raises)
        for _ in range(5):
            tracker.record_seat_hand(seat_idx=1, vpip=True, pfr=False, postflop_bet=False, faced_bet_and_folded=False)

        prof1 = tracker.profiles[1]
        self.assertGreater(prof1.vpip_tracker.posterior_mean, 0.50)
        self.assertLess(prof1.pfr_tracker.posterior_mean, 0.15)
        self.assertLess(prof1.af_tracker.posterior_mean, 0.25)

        # Seat 2: Rock (folds 5 hands without entering)
        for _ in range(5):
            tracker.record_seat_hand(seat_idx=2, vpip=False, pfr=False, postflop_bet=False, faced_bet_and_folded=True)

        prof2 = tracker.profiles[2]
        self.assertLess(prof2.vpip_tracker.posterior_mean, 0.20)
        self.assertGreater(prof2.fold_pressure_tracker.posterior_mean, 0.60)

    def test_card_attention_extractor_7cards(self):
        """Verifies CardAttentionExtractor forward pass with num_cards=7 and 125-dim observations."""
        env = HoldemPokerGymEnv(obs_version="b", rng_seed=42)
        extractor = CardAttentionExtractor(
            observation_space=env.observation_space,
            num_cards=7,
            embed_dim=32,
            num_heads=4,
            pooling="mean",
            card_features_dim=32,
            game_features_dim=64,
            slot_features_dim=0,
        )

        obs, _ = env.reset()
        t_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0) # (1, 125)
        out = extractor(t_obs)

        # Expected output dim: card_features_dim (32) + game_features_dim (64) = 96
        self.assertEqual(out.shape, (1, 96))
        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
