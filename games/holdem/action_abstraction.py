"""
games/holdem/action_abstraction.py — Action Encodings and Constants for No-Limit Texas Hold'em.

Standard 6-Action Betting Abstraction:
    0: Fold
    1: Check / Call
    2: Min Raise
    3: Half-Pot Raise (50% Pot)
    4: Full-Pot Raise (100% Pot)
    5: All-In
"""

from __future__ import annotations
from typing import List

# Action constants (Legacy 6-Action: Models A..E)
A_FOLD = 0
A_CHECK_CALL = 1
A_MIN_RAISE = 2
A_HALF_POT = 3
A_POT = 4
A_ALL_IN = 5

NUM_ACTIONS = 6

ACTION_NAMES = {
    A_FOLD: "Fold",
    A_CHECK_CALL: "Check/Call",
    A_MIN_RAISE: "Min-Raise",
    A_HALF_POT: "Half-Pot",
    A_POT: "Pot",
    A_ALL_IN: "All-In",
}

# Model F 8-Action Fine-Grained Betting Abstraction
A_F_FOLD = 0
A_F_CHECK_CALL = 1
A_F_MIN_RAISE = 2
A_F_THIRD_POT = 3
A_F_HALF_POT = 4
A_F_POT = 5
A_F_OVERBET = 6
A_F_ALL_IN = 7

NUM_ACTIONS_HOLDEM_F = 8

ACTION_NAMES_F = {
    A_F_FOLD: "Fold",
    A_F_CHECK_CALL: "Check/Call",
    A_F_MIN_RAISE: "Min-Raise",
    A_F_THIRD_POT: "1/3-Pot",
    A_F_HALF_POT: "Half-Pot",
    A_F_POT: "Pot",
    A_F_OVERBET: "Overbet-1.5x",
    A_F_ALL_IN: "All-In",
}

# Street constants
STREET_PREFLOP = "preflop"
STREET_FLOP = "flop"
STREET_TURN = "turn"
STREET_RIVER = "river"
STREET_SHOWDOWN = "showdown"

STREETS = [STREET_PREFLOP, STREET_FLOP, STREET_TURN, STREET_RIVER, STREET_SHOWDOWN]
