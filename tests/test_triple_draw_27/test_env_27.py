"""
test_env_27.py — Unit Tests for 2-7 Triple Draw Table Environment.
"""

import unittest
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from games.triple_draw_27.env import (
    TripleDrawPokerEnv,
    PHASE_BET_1,
    PHASE_DRAW_1,
    PHASE_BET_2,
    PHASE_DRAW_2,
    PHASE_BET_3,
    PHASE_DRAW_3,
    PHASE_BET_4,
    PHASE_SHOWDOWN,
    A_FOLD,
    A_CALL,
    A_DRAW_START,
)


class TestTripleDrawEnv(unittest.TestCase):
    def setUp(self):
        self.env = TripleDrawPokerEnv(
            num_seats=5,
            starting_chips=200,
            small_blind=1,
            big_blind=2,
            rng_seed=42,
        )
        self.env.reset_tournament()

    def test_initial_state(self):
        self.assertEqual(len(self.env.alive_seats), 5)
        self.assertEqual(self.env.phase, PHASE_BET_1)
        for s in self.env.seats:
            self.assertEqual(len(s.hand), 5)

    def test_full_round_phase_transitions(self):
        # 1. Bet 1: Everyone calls/checks
        while self.env.phase == PHASE_BET_1:
            seat = self.env.current_seat
            self.env.step_action(seat, A_CALL)

        self.assertEqual(self.env.phase, PHASE_DRAW_1)

        # 2. Draw 1: Everyone stands pat (bitmask 0)
        while self.env.phase == PHASE_DRAW_1:
            seat = self.env.current_seat
            self.env.step_action(seat, A_DRAW_START + 0)

        self.assertEqual(self.env.phase, PHASE_BET_2)

        # 3. Bet 2: Everyone calls
        while self.env.phase == PHASE_BET_2:
            seat = self.env.current_seat
            self.env.step_action(seat, A_CALL)

        self.assertEqual(self.env.phase, PHASE_DRAW_2)

        # 4. Draw 2: Everyone stands pat
        while self.env.phase == PHASE_DRAW_2:
            seat = self.env.current_seat
            self.env.step_action(seat, A_DRAW_START + 0)

        self.assertEqual(self.env.phase, PHASE_BET_3)

        # 5. Bet 3: Everyone calls
        while self.env.phase == PHASE_BET_3:
            seat = self.env.current_seat
            self.env.step_action(seat, A_CALL)

        self.assertEqual(self.env.phase, PHASE_DRAW_3)

        # 6. Draw 3: Everyone stands pat
        while self.env.phase == PHASE_DRAW_3:
            seat = self.env.current_seat
            self.env.step_action(seat, A_DRAW_START + 0)

        self.assertEqual(self.env.phase, PHASE_BET_4)

        # 7. Bet 4 (River): Everyone calls
        while self.env.phase == PHASE_BET_4:
            seat = self.env.current_seat
            self.env.step_action(seat, A_CALL)

        # Hand should conclude and advance hands_played
        self.assertEqual(self.env.hands_played, 1)

    def test_uncontested_fold_awards_pot(self):
        # In Bet 1, fold 4 players to finish exactly 1 hand
        for _ in range(4):
            s = self.env.current_seat
            self.env.step_action(s, A_FOLD)

        self.assertEqual(self.env.hands_played, 1)


if __name__ == "__main__":
    unittest.main()
