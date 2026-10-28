"""
tests/test_holdem/test_holdem_c_features.py — Unit Tests for Texas Hold'em Model C Features.

Validates:
1. 138-dimensional observation space shape and strict [-1.0, 1.0] bounds.
2. Canonical 169-hand GTO preflop matrix, hand tiering, and positional gating.
3. Malmuth-Harville ICM tournament equity calculations and bubble factor extraction.
4. CardAttentionExtractor(num_cards=7) forward pass compatibility with 138-dim input.
5. HoldemModelOpponent integration and league-mode sparring.
6. Positional GTO masking within the poker engine.
"""

from __future__ import annotations
import unittest
import numpy as np
import torch

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
)
from games.holdem.env import HoldemPokerEnv
from games.holdem.gym_wrapper import HoldemPokerGymEnv, OBS_DIM_HOLDEM_C
from games.holdem.gto_ranges import (
    get_canonical_hand,
    get_hand_gto_tier,
    is_gto_preflop_permitted,
    extract_preflop_gto_features,
    TIER_MONSTER,
    TIER_STRONG,
    TIER_PLAYABLE,
    TIER_SPECULATIVE,
    TIER_MARGINAL,
    TIER_TRASH,
)
from games.holdem.icm_engine import calculate_icm_equities, compute_icm_features
from games.holdem.opponents import HoldemModelOpponent, make_random_holdem_opponent
from core.neural.card_attention import CardAttentionExtractor


class TestHoldemCFeatures(unittest.TestCase):
    """Test suite for Model C GTO, ICM, and Neural Attention features."""

    def test_obs_c_shape_and_bounds(self):
        """Verifies 138-dim observation space shape and strict [-1.0, 1.0] bounds across 50 steps."""
        env = HoldemPokerGymEnv(obs_version="c", rng_seed=42)
        self.assertEqual(env.observation_space.shape, (OBS_DIM_HOLDEM_C,))
        self.assertEqual(env.obs_dim, 138)

        obs, _ = env.reset()
        self.assertEqual(obs.shape, (138,))
        self.assertTrue(np.all(obs >= -1.0001), f"Obs below -1.0: {obs[obs < -1.0]}")
        self.assertTrue(np.all(obs <= 1.0001), f"Obs above 1.0: {obs[obs > 1.0]}")

        # Step 50 actions to ensure all postflop, pot-control, and ICM features stay bounded
        rng = np.random.default_rng(123)
        for _ in range(50):
            mask = env.action_masks()
            legal = np.where(mask == 1)[0]
            action = int(rng.choice(legal))
            obs, reward, term, trunc, _ = env.step(action)
            self.assertEqual(obs.shape, (138,))
            self.assertTrue(np.all(obs >= -1.0001), f"Obs below -1.0: {obs[obs < -1.0]}")
            self.assertTrue(np.all(obs <= 1.0001), f"Obs above 1.0: {obs[obs > 1.0]}")
            if term or trunc:
                obs, _ = env.reset()

    def test_gto_ranges_canonical_and_tiers(self):
        """Validates canonical hand mapping, GTO tiering, and positional thresholds."""
        # AA pocket pair
        c_as = parse_card("As")
        c_ah = parse_card("Ah")
        self.assertEqual(get_canonical_hand(c_as, c_ah), "AA")
        self.assertEqual(get_hand_gto_tier(c_as, c_ah), TIER_MONSTER)

        # AKs suited
        c_ks = parse_card("Ks")
        self.assertEqual(get_canonical_hand(c_as, c_ks), "AKs")
        self.assertEqual(get_hand_gto_tier(c_as, c_ks), TIER_MONSTER)

        # AKo offsuit
        c_kh = parse_card("Kh")
        self.assertEqual(get_canonical_hand(c_as, c_kh), "AKo")
        self.assertEqual(get_hand_gto_tier(c_as, c_kh), TIER_STRONG)

        # 72o trash
        c_7s = parse_card("7s")
        c_2h = parse_card("2h")
        self.assertEqual(get_canonical_hand(c_7s, c_2h), "72o")
        self.assertEqual(get_hand_gto_tier(c_7s, c_2h), TIER_TRASH)

        # Positional GTO permission:
        # UTG (pos 3 in 5-player) facing 1 BB to call:
        # 72o MUST NOT be permitted to call or raise
        can_call_utg_trash, can_raise_utg_trash = is_gto_preflop_permitted(
            c_7s, c_2h, pos_offset=3, bet_to_call=2, big_blind=2
        )
        self.assertFalse(can_call_utg_trash)
        self.assertFalse(can_raise_utg_trash)

        # UTG holding AA MUST be permitted to call and raise
        can_call_utg_aa, can_raise_utg_aa = is_gto_preflop_permitted(
            c_as, c_ah, pos_offset=3, bet_to_call=2, big_blind=2
        )
        self.assertTrue(can_call_utg_aa)
        self.assertTrue(can_raise_utg_aa)

        # Button (pos 0) holding marginal hand (e.g. K6s, Tier 5) can open-raise unopened pot
        c_6s = parse_card("6s")
        can_call_btn, can_raise_btn = is_gto_preflop_permitted(
            c_ks, c_6s, pos_offset=0, bet_to_call=2, big_blind=2
        )
        self.assertTrue(can_raise_btn)

    def test_gto_features_extraction(self):
        """Validates 6-feature GTO preflop vector extraction."""
        c_as = parse_card("As")
        c_ks = parse_card("Ks")
        feats = extract_preflop_gto_features(
            c_as, c_ks, pos_offset=0, num_callers=0, bet_to_call=2, big_blind=2
        )
        self.assertEqual(feats.shape, (6,))
        self.assertTrue(np.all(feats >= -1.0))
        self.assertTrue(np.all(feats <= 1.0))
        # AKs is suited -> feats[2] == 1.0
        self.assertEqual(feats[2], 1.0)
        # Button unopened -> steal flag feats[3] == 1.0
        self.assertEqual(feats[3], 1.0)

    def test_icm_engine_calculations(self):
        """Validates Malmuth-Harville ICM tournament equities and bubble factor calculations."""
        # 1. Equal 5-player stacks
        chips = [200, 200, 200, 200, 200]
        eqs = calculate_icm_equities(chips, payouts=[0.65, 0.35])
        self.assertEqual(len(eqs), 5)
        self.assertAlmostEqual(sum(eqs), 1.0, places=5)
        for eq in eqs:
            self.assertAlmostEqual(eq, 0.20, places=4)

        # 2. Dominant chip leader
        chips_dom = [800, 50, 50, 50, 50]
        eqs_dom = calculate_icm_equities(chips_dom, payouts=[0.65, 0.35])
        self.assertGreater(eqs_dom[0], 0.55)
        for i in range(1, 5):
            self.assertAlmostEqual(eqs_dom[i], eqs_dom[1], places=4)

        # 3. Eliminated players receive 0 equity
        chips_elim = [600, 400, 0, 0, 0]
        eqs_elim = calculate_icm_equities(chips_elim, payouts=[0.65, 0.35])
        self.assertAlmostEqual(eqs_elim[2], 0.0)
        self.assertAlmostEqual(eqs_elim[3], 0.0)
        self.assertAlmostEqual(eqs_elim[4], 0.0)
        self.assertAlmostEqual(sum(eqs_elim), 1.0, places=5)

        # 4. ICM observation features
        feats = compute_icm_features(hero_seat=0, seats_chips=[200, 200, 200, 200, 200], bet_to_call=20, pot=40)
        self.assertEqual(feats.shape, (3,))
        self.assertTrue(np.all(feats >= -1.0))
        self.assertTrue(np.all(feats <= 1.0))

    def test_card_attention_extractor_138dims(self):
        """Verifies CardAttentionExtractor forward pass with num_cards=7 and 138-dim observations."""
        env = HoldemPokerGymEnv(obs_version="c", rng_seed=42)
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
        t_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1, 138)
        out = extractor(t_obs)

        # Expected output dim: card_features_dim (32) + game_features_dim (64) = 96
        self.assertEqual(out.shape, (1, 96))
        self.assertFalse(torch.isnan(out).any())

        # Test batch forward pass and backward pass
        t_batch = torch.as_tensor(np.stack([obs, obs]), dtype=torch.float32)  # (2, 138)
        out_batch = extractor(t_batch)
        self.assertEqual(out_batch.shape, (2, 96))
        loss = out_batch.sum()
        loss.backward()
        # Verify gradients exist and are finite
        for p in extractor.parameters():
            if p.requires_grad and p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any())

    def test_holdem_model_opponent_integration(self):
        """Verifies HoldemModelOpponent acting and league-mode sparring."""
        bot = HoldemModelOpponent(opponent_id=1, model_path="models/holdem/model_holdem_b.zip")
        legal = [A_FOLD, A_CHECK_CALL, A_MIN_RAISE]
        act = bot.act(
            hole_cards=[parse_card("As"), parse_card("Ks")],
            board=[],
            pot=10,
            bet_to_call=2,
            street=STREET_PREFLOP,
            stack=200,
            legal_actions=legal,
        )
        self.assertIn(act, legal)

        # League mode gym environment
        env_league = HoldemPokerGymEnv(obs_version="c", league_mode=True, rng_seed=99)
        obs, _ = env_league.reset()
        self.assertEqual(obs.shape, (138,))
        mask = env_league.action_masks()
        legal = np.where(mask == 1)[0]
        obs2, rew, term, trunc, info = env_league.step(int(legal[0]))
        self.assertEqual(obs2.shape, (138,))

    def test_pos_gto_masking_in_env(self):
        """Verifies get_seat_position_from_dealer and GTO masking in HoldemPokerEnv."""
        env = HoldemPokerEnv(num_seats=5, pos_gto_masking=True, rng_seed=42)
        env.reset_session()
        env.start_hand()

        # Check seat position relative to dealer
        d_seat = env.dealer_seat
        self.assertEqual(env.get_seat_position_from_dealer(d_seat), 0)

        # Force a preflop hand with trash cards (72o)
        cur = env.current_seat
        env.seats[cur].hand = [parse_card("7s"), parse_card("2h")]
        pos = env.get_seat_position_from_dealer(cur)

        legal = env.get_legal_actions(cur)
        if pos == 3:  # UTG
            # 72o from UTG must not be able to raise
            self.assertNotIn(A_MIN_RAISE, legal)
            self.assertNotIn(A_HALF_POT, legal)
            self.assertNotIn(A_POT, legal)


if __name__ == "__main__":
    unittest.main()
