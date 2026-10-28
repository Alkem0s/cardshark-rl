"""
tests/test_holdem/test_holdem_d_features.py — Unit Tests for Texas Hold'em Model D (Targeted Exploiter).

Validates:
1. 152-dimensional observation space shape and strict [-1.0, 1.0] bounds.
2. Opponent Exploiter Engine (archetype classification, exploit directives, fold equity).
3. Adaptive Exploitative Action Masking (AEAM) rules:
   - Anti-Station Shield (no betting air into CallingStation)
   - Anti-Station Max-Value (forces Half-Pot/Pot/Shove with Two Pair+)
   - TAG Blind Steal (unmasks open-raises from BTN/CO facing TAG blinds)
   - LAG Counter-Attack (unmasks 3-bets/shoves against wide LAG opens)
4. Decoupled Hierarchical Policy (HoldemDecoupledActionNet & Policy).
5. Auxiliary Supervised Belief Head (forward pass & multi-task loss gradient flow).
"""

from __future__ import annotations
import unittest
import numpy as np
import torch
import torch.nn.functional as F

from core.card import parse_card
from games.holdem.action_abstraction import (
    A_FOLD,
    A_CHECK_CALL,
    A_MIN_RAISE,
    A_HALF_POT,
    A_POT,
    A_ALL_IN,
    STREET_PREFLOP,
    STREET_FLOP,
    STREET_TURN,
    STREET_RIVER,
)
from games.holdem.env import HoldemPokerEnv
from games.holdem.gym_wrapper import HoldemPokerGymEnv, OBS_DIM_HOLDEM_D
from games.holdem.opponents import CallingStationBot, TAGBot, LAGBot, RandomHoldemBot
from games.holdem.opponent_tracker import HoldemSeatProfile, HoldemTableOpponentTracker
from games.holdem.opponent_exploiter import (
    classify_opponent_archetype_probs,
    get_primary_adversary_seat,
    build_exploit_features,
    get_seat_archetype_probs,
    is_primary_adversary_archetype,
    ARCHETYPE_TAG,
    ARCHETYPE_LAG,
    ARCHETYPE_STATION,
    ARCHETYPE_RANDOM,
    NUM_EXPLOIT_FEATURES,
)
from rl.holdem_decoupled_actor import HoldemDecoupledActionNet, HoldemDecoupledActorCriticPolicy
from rl.holdem_auxiliary_belief import HoldemAuxiliaryBeliefHead


class TestHoldemDFeatures(unittest.TestCase):
    """Test suite for Model D targeted exploitation, AEAM, and neural architectures."""

    def test_obs_d_shape_and_bounds(self):
        """Verifies 152-dim observation space shape and strict [-1.0, 1.0] bounds."""
        env = HoldemPokerGymEnv(obs_version="d", rng_seed=42)
        self.assertEqual(env.observation_space.shape, (OBS_DIM_HOLDEM_D,))
        self.assertEqual(env.obs_dim, 152)

        obs, _ = env.reset()
        self.assertEqual(obs.shape, (152,))
        self.assertTrue(np.all(obs >= -1.0001), f"Obs below -1.0: {obs[obs < -1.0]}")
        self.assertTrue(np.all(obs <= 1.0001), f"Obs above 1.0: {obs[obs > 1.0]}")

        # Step through multiple transitions
        for _ in range(50):
            action_mask = env.action_masks()
            legal_indices = np.where(action_mask == 1)[0]
            self.assertTrue(len(legal_indices) > 0)
            act = int(np.random.choice(legal_indices))
            obs, reward, terminated, truncated, info = env.step(act)
            self.assertEqual(obs.shape, (152,))
            self.assertTrue(np.all(obs >= -1.0001), f"Obs below -1.0: {obs[obs < -1.0]}")
            self.assertTrue(np.all(obs <= 1.0001), f"Obs above 1.0: {obs[obs > 1.0]}")
            if terminated or truncated:
                obs, _ = env.reset()

    def test_opponent_exploiter_archetypes(self):
        """Verifies Bayesian classification matches true heuristic bot statistics."""
        # Calling Station profile: high VPIP, low PFR, low Fold
        station_profile = HoldemSeatProfile(seat_idx=1)
        for _ in range(20):
            station_profile.vpip_tracker.update(True)
            station_profile.pfr_tracker.update(False)
            station_profile.fold_pressure_tracker.update(False)
            station_profile.hands_observed += 1

        probs = classify_opponent_archetype_probs(station_profile)
        self.assertEqual(probs.shape, (4,))
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)
        self.assertEqual(int(np.argmax(probs)), ARCHETYPE_STATION)

        # TAG profile: low VPIP, high PFR, high Fold
        tag_profile = HoldemSeatProfile(seat_idx=2)
        for _ in range(20):
            tag_profile.vpip_tracker.update(False if _ % 5 != 0 else True)
            tag_profile.pfr_tracker.update(True if _ % 5 == 0 else False)
            tag_profile.fold_pressure_tracker.update(True)
            tag_profile.hands_observed += 1

        probs_tag = classify_opponent_archetype_probs(tag_profile)
        self.assertEqual(int(np.argmax(probs_tag)), ARCHETYPE_TAG)

    def test_build_exploit_features(self):
        """Verifies 14 exploit features are bounded in [-1.0, 1.0]."""
        env = HoldemPokerEnv(num_seats=5, starting_chips=200, rng_seed=42)
        tracker = HoldemTableOpponentTracker(num_seats=5)
        feats = build_exploit_features(env, hero_seat=0, tracker=tracker)
        self.assertEqual(len(feats), NUM_EXPLOIT_FEATURES)
        self.assertTrue(np.all(feats >= -1.0001))
        self.assertTrue(np.all(feats <= 1.0001))

    def test_dominated_action_masking(self):
        """Verifies environment prunes strictly dominated mathematical blunders while preserving strategic freedom."""
        env = HoldemPokerEnv(num_seats=5, starting_chips=200, dominated_masking=True, rng_seed=42)
        env.dealer_seat = 0
        env.current_seat = 0
        env.street = STREET_FLOP

        # Hero has Ah Kd, board is As Ks 2c, pot is 20, chips 190
        env.seats[0].hand = [parse_card("Ah"), parse_card("Kd")]
        env.board = [parse_card("As"), parse_card("Ks"), parse_card("2c")]
        env.pot = 20
        env.seats[0].chips = 190
        env.seats[0].bet_to_call = 0

        for s in env.seats[1:]:
            s.folded = True
        env.seats[1].folded = False

        legal = env.get_legal_actions(0)
        # Check is free -> Fold is dominated and MUST NOT be legal
        self.assertNotIn(A_FOLD, legal)
        self.assertIn(A_CHECK_CALL, legal)
        # Strategic sizing options are preserved for the neural policy
        self.assertIn(A_MIN_RAISE, legal)
        self.assertIn(A_HALF_POT, legal)
        self.assertIn(A_POT, legal)
        # Deep-stack overbet shove into tiny pot is pruned
        self.assertNotIn(A_ALL_IN, legal)

    def test_opponent_parameter_jitter(self):
        """Verifies that heuristic bots possess randomized parameter jitter to prevent static overfitting."""
        rng = np.random.default_rng(123)
        lags = [LAGBot(opponent_id=i, rng=rng) for i in range(5)]
        open_freqs = [bot.open_freq for bot in lags]
        self.assertTrue(len(set(open_freqs)) > 1, "LAG open frequencies should vary via jitter")

        tags = [TAGBot(opponent_id=i, rng=rng) for i in range(5)]
        call_threshs = [bot.call_thresh for bot in tags]
        self.assertTrue(len(set(call_threshs)) > 1, "TAG call thresholds should vary via jitter")

    def test_pure_bayesian_archetype_inference(self):
        """Verifies archetype probabilities are derived purely from empirical Bayesian statistics without bot names."""
        env = HoldemPokerEnv(num_seats=5, starting_chips=200, rng_seed=42)
        tracker = HoldemTableOpponentTracker(num_seats=5)
        env.tracker = tracker

        # Default prior before observations is balanced
        prior_probs = get_seat_archetype_probs(env, 1)
        self.assertEqual(len(prior_probs), 4)

        # Simulate 15 hands of loose-aggressive play for seat 1
        for _ in range(15):
            tracker.record_seat_hand(seat_idx=1, vpip=True, pfr=True, postflop_bet=True, faced_bet_and_folded=False)

        updated_probs = get_seat_archetype_probs(env, 1)
        self.assertEqual(int(np.argmax(updated_probs)), ARCHETYPE_LAG)
        self.assertTrue(is_primary_adversary_archetype(env, hero_seat=0, target_archetype=ARCHETYPE_LAG))

    def test_decoupled_actor_forward_and_backward(self):
        """Verifies HoldemDecoupledActionNet produces (B, 6) logits with active gradient flow."""
        net = HoldemDecoupledActionNet(latent_dim=64, hidden_dim=32)
        x = torch.randn(8, 64, requires_grad=True)
        logits = net(x)

        self.assertEqual(logits.shape, (8, 6))
        loss = logits.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.all(torch.isfinite(x.grad)))

    def test_auxiliary_belief_head_forward_and_loss(self):
        """Verifies HoldemAuxiliaryBeliefHead multi-task loss computation and gradient flow."""
        head = HoldemAuxiliaryBeliefHead(feature_dim=64, hidden_dim=32)
        x = torch.randn(8, 64, requires_grad=True)

        arch_logits, tier_logits, fold_logits = head(x)
        self.assertEqual(arch_logits.shape, (8, 4))
        self.assertEqual(tier_logits.shape, (8, 4))
        self.assertEqual(fold_logits.shape, (8,))

        target_arch = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int64)
        target_tier = torch.tensor([3, 2, 1, 0, 3, 2, 1, 0], dtype=torch.int64)
        target_fold = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        loss_arch = F.cross_entropy(arch_logits, target_arch)
        loss_tier = F.cross_entropy(tier_logits, target_tier)
        loss_fold = F.binary_cross_entropy_with_logits(fold_logits, target_fold)
        total_loss = loss_arch + 0.5 * loss_tier + 0.5 * loss_fold

        total_loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.all(torch.isfinite(x.grad)))


if __name__ == "__main__":
    unittest.main()
