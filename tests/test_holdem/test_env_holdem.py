"""
tests/test_holdem/test_env_holdem.py — Unit tests for Texas Hold'em table engine.
"""

import unittest
import numpy as np
from games.holdem.env import HoldemPokerEnv
from games.holdem.action_abstraction import (
    A_FOLD,
    A_CHECK_CALL,
    A_MIN_RAISE,
    A_ALL_IN,
    STREET_PREFLOP,
    STREET_FLOP,
    STREET_TURN,
    STREET_RIVER,
    STREET_SHOWDOWN,
)


class TestHoldemEnv(unittest.TestCase):
    def setUp(self):
        self.env = HoldemPokerEnv(
            num_seats=5,
            starting_chips=200,
            small_blind=1,
            big_blind=2,
            rng=np.random.default_rng(12345),
        )
        self.env.reset_session()

    def test_initial_deal_and_blinds(self):
        # 5 players, alive
        self.assertEqual(len(self.env.alive_seats), 5)
        # Each player has 2 hole cards
        for s in self.env.seats:
            self.assertEqual(len(s.hand), 2)
        # Board is empty preflop
        self.assertEqual(len(self.env.board), 0)
        self.assertEqual(self.env.street, STREET_PREFLOP)
        # Pot has at least small blind + big blind
        self.assertEqual(self.env.pot, 3)

    def test_uncontested_winner(self):
        # Everyone folds to the big blind
        while self.env.street == STREET_PREFLOP and not self.env.session_done:
            cur = self.env.current_seat
            legal = self.env.get_legal_actions(cur)
            if A_FOLD in legal and len(self.env.active_seats) > 2:
                self.env.step_action(cur, A_FOLD)
            elif A_FOLD in legal:
                self.env.step_action(cur, A_FOLD)
                break
            else:
                self.env.step_action(cur, A_CHECK_CALL)

        # Hand finished and advanced
        self.assertEqual(self.env.hands_played, 1)

    def test_full_hand_to_showdown(self):
        # Everyone checks / calls all the way to showdown
        initial_hands_played = self.env.hands_played
        max_steps = 100
        step_count = 0

        while self.env.hands_played == initial_hands_played and step_count < max_steps:
            cur = self.env.current_seat
            legal = self.env.get_legal_actions(cur)
            self.assertIn(A_CHECK_CALL, legal)
            self.env.step_action(cur, A_CHECK_CALL)
            step_count += 1

        self.assertEqual(self.env.hands_played, initial_hands_played + 1)
        self.assertEqual(sum(s.chips for s in self.env.seats) + self.env.pot, 1000)

    def test_all_in_runout(self):
        # Player pushes all-in preflop, another calls
        cur = self.env.current_seat
        self.env.step_action(cur, A_ALL_IN)
        # Next player calls all-in
        cur = self.env.current_seat
        self.env.step_action(cur, A_ALL_IN)
        # Remaining fold
        while self.env.street == STREET_PREFLOP and len(self.env.active_seats) > 2:
            cur = self.env.current_seat
            self.env.step_action(cur, A_FOLD)

        # Total chips on table strictly conserved (player chips + active pot)
        self.assertEqual(sum(s.chips for s in self.env.seats) + self.env.pot, 1000)


if __name__ == "__main__":
    unittest.main()
