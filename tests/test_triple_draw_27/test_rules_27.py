"""
test_rules_27.py — Unit Tests for 2-7 Triple Draw (Deuce-to-Seven Lowball) Evaluation.
"""

import unittest
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.card import parse_card
from games.triple_draw_27.rules import (
    evaluate_27_hand,
    hand_category_27,
    hand_category_name_27,
    is_wheel_27,
    get_lowball_draw_candidates,
)


class TestRules27(unittest.TestCase):
    """Test suite covering 2-7 Lowball rules, hand rankings, and edge cases."""

    def cards(self, card_strs: list[str]) -> list[int]:
        return [parse_card(s) for s in card_strs]

    def test_wheel_is_absolute_nuts(self):
        wheel = self.cards(["7s", "5d", "4c", "3h", "2s"])
        self.assertTrue(is_wheel_27(wheel))
        self.assertEqual(hand_category_27(wheel), 8)
        self.assertEqual(hand_category_name_27(wheel), "7-High Low")

        # Must beat second-best 7-low: 7-6-4-3-2
        seven_six = self.cards(["7s", "6d", "4c", "3h", "2s"])
        self.assertGreater(evaluate_27_hand(wheel), evaluate_27_hand(seven_six))

        # Must beat smooth eight: 8-5-4-3-2
        smooth_eight = self.cards(["8s", "5d", "4c", "3h", "2s"])
        self.assertGreater(evaluate_27_hand(wheel), evaluate_27_hand(smooth_eight))

    def test_flush_penalty(self):
        # 7-5-4-3-2 in all spades is a FLUSH, so it is severely penalized
        flush_wheel = self.cards(["7s", "5s", "4s", "3s", "2s"])
        self.assertFalse(is_wheel_27(flush_wheel))
        self.assertEqual(hand_category_27(flush_wheel), 3)  # Tier 3 (Flush)
        self.assertEqual(hand_category_name_27(flush_wheel), "Flush")

        # Any unsuited no-pair hand (even King-high) must beat this flush
        king_low = self.cards(["Kh", "Qc", "Jd", "9s", "7c"])
        self.assertEqual(hand_category_27(king_low), 8)
        self.assertGreater(evaluate_27_hand(king_low), evaluate_27_hand(flush_wheel))

        # Even One Pair must beat a flush in 2-7
        pair_of_twos = self.cards(["2s", "2d", "5c", "4h", "3s"])
        self.assertEqual(hand_category_27(pair_of_twos), 7)
        self.assertGreater(evaluate_27_hand(pair_of_twos), evaluate_27_hand(flush_wheel))

    def test_straight_penalty(self):
        # 6-5-4-3-2 is a straight! Straights hurt the hand
        six_straight = self.cards(["6s", "5d", "4c", "3h", "2s"])
        self.assertEqual(hand_category_27(six_straight), 4)  # Tier 4 (Straight)
        self.assertEqual(hand_category_name_27(six_straight), "Straight")

        # 7-6-5-4-2 (rough seven) is NOT a straight, so it must beat 6-5-4-3-2
        rough_seven = self.cards(["7s", "6d", "5c", "4h", "2s"])
        self.assertEqual(hand_category_27(rough_seven), 8)
        self.assertGreater(evaluate_27_hand(rough_seven), evaluate_27_hand(six_straight))

    def test_aces_are_strictly_high(self):
        # A-5-4-3-2 is Ace-high, NOT a straight (Ace does not wrap)
        ace_low = self.cards(["As", "5d", "4c", "3h", "2s"])
        self.assertEqual(hand_category_27(ace_low), 8)  # Tier 8 (No pair low)
        self.assertEqual(hand_category_name_27(ace_low), "A-High Low")

        # King-high must beat Ace-high
        king_high = self.cards(["Ks", "Qd", "Jc", "8h", "7s"])
        self.assertGreater(evaluate_27_hand(king_high), evaluate_27_hand(ace_low))

    def test_tiebreaker_hierarchy(self):
        # 8-6-5-4-2 beats 8-7-4-3-2
        eight_six = self.cards(["8c", "6d", "5h", "4s", "2c"])
        eight_seven = self.cards(["8d", "7c", "4h", "3s", "2d"])
        self.assertGreater(evaluate_27_hand(eight_six), evaluate_27_hand(eight_seven))

        # 9-8-7-6-2 beats 9-8-7-6-3
        nine_two = self.cards(["9c", "8d", "7h", "6s", "2c"])
        nine_three = self.cards(["9d", "8h", "7c", "6d", "3s"])
        self.assertGreater(evaluate_27_hand(nine_two), evaluate_27_hand(nine_three))

        # Pair of 2s beats Pair of 3s
        pair_twos = self.cards(["2c", "2d", "7h", "5s", "4c"])
        pair_threes = self.cards(["3c", "3d", "7h", "5s", "4c"])
        self.assertGreater(evaluate_27_hand(pair_twos), evaluate_27_hand(pair_threes))

    def test_discard_recommendations(self):
        # In hand with pair of Kings and low cards (2, 4, 7), should discard Kings
        hand = self.cards(["Ks", "Kd", "7c", "4h", "2s"])
        discards = get_lowball_draw_candidates(hand)
        # Should discard indices 0 and 1 (both Kings)
        self.assertIn(0, discards)
        self.assertIn(1, discards)
        self.assertNotIn(2, discards)  # Keep 7
        self.assertNotIn(3, discards)  # Keep 4
        self.assertNotIn(4, discards)  # Keep 2


if __name__ == "__main__":
    unittest.main()
