"""
tests/test_holdem/test_evaluator.py — Unit tests for Texas Hold'em 7-card evaluator.
"""

import unittest
from core.card import parse_card
from games.holdem.evaluator import (
    evaluate_holdem_hand,
    holdem_category,
    holdem_category_name,
)


class TestHoldemEvaluator(unittest.TestCase):
    def test_best_5_of_7_cards(self):
        # Board: 2h 3h 4h 5h 9c (has flush draw / straight draw)
        # Hero: Ah Kh (Nut flush)
        cards = [parse_card(c) for c in ["Ah", "Kh", "2h", "3h", "4h", "5h", "9c"]]
        score, best_5 = evaluate_holdem_hand(cards)
        self.assertEqual(holdem_category(cards), 8)  # Straight flush A-2-3-4-5 of hearts!

    def test_full_house_over_flush(self):
        # Hero: Kc Kd
        # Board: Ks 7s 7d 2s 4s (Flush is possible on board, but Hero has kings full of sevens)
        cards = [parse_card(c) for c in ["Kc", "Kd", "Ks", "7s", "7d", "2s", "4s"]]
        self.assertEqual(holdem_category_name(cards), "Full House")

    def test_kicker_resolution(self):
        # Both players have pair of Aces
        # Hand A: As Kd on Ah 8c 7d 4s 2c -> Pair of Aces, King kicker
        # Hand B: Ac Qd on Ah 8c 7d 4s 2c -> Pair of Aces, Queen kicker
        hand_a = [parse_card(c) for c in ["As", "Kd", "Ah", "8c", "7d", "4s", "2c"]]
        hand_b = [parse_card(c) for c in ["Ac", "Qd", "Ah", "8c", "7d", "4s", "2c"]]
        score_a, _ = evaluate_holdem_hand(hand_a)
        score_b, _ = evaluate_holdem_hand(hand_b)
        self.assertGreater(score_a, score_b)

    def test_wheel_straight(self):
        # A-2-3-4-5 straight
        cards = [parse_card(c) for c in ["Ac", "2d", "3h", "4s", "5c", "9d", "Jc"]]
        self.assertEqual(holdem_category_name(cards), "Straight")


if __name__ == "__main__":
    unittest.main()
