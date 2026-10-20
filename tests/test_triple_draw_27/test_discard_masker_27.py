"""
test_discard_masker_27.py — Unit Tests for 2-7 Triple Draw Dominated Discard Masking.
"""

import unittest
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.card import parse_card
from games.triple_draw_27.discard_masker import get_legal_discard_mask_27


class TestDiscardMasker27(unittest.TestCase):
    def cards(self, card_strs: list[str]) -> list[int]:
        return [parse_card(s) for s in card_strs]

    def test_wheel_forces_stand_pat(self):
        wheel = self.cards(["7s", "5d", "4c", "3h", "2s"])
        mask = get_legal_discard_mask_27(wheel)
        # Only bitmask 0 should be legal
        self.assertEqual(mask[0], 1)
        self.assertEqual(mask.sum(), 1)

    def test_never_keeps_pairs(self):
        hand = self.cards(["Ks", "Kd", "7c", "4h", "2s"])
        mask = get_legal_discard_mask_27(hand)
        # Any bitmask that keeps both Kings (card 0 and card 1) must be masked out (0)
        for bitmask in range(32):
            if not (bitmask & (1 << 0)) and not (bitmask & (1 << 1)):
                # Keeping both cards 0 and 1
                self.assertEqual(mask[bitmask], 0, f"Bitmask {bitmask} kept pair of Kings!")

    def test_suited_wheel_breaks_flush(self):
        flush_wheel = self.cards(["7s", "5s", "4s", "3s", "2s"])
        mask = get_legal_discard_mask_27(flush_wheel)
        # Standing pat (bitmask 0) must be masked out because a flush is bad
        self.assertEqual(mask[0], 0)
        # Discarding one card (e.g. card 0: 7s, bitmask 1) must be legal
        self.assertEqual(mask[1], 1)
        # At least one discard bitmask must be legal
        self.assertGreater(mask.sum(), 0)


if __name__ == "__main__":
    unittest.main()
