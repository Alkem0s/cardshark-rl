"""
core package — Mode-agnostic card primitives, math formulas, and neural layers.
"""

from core.card import (
    Deck,
    rank_of,
    suit_of,
    card_str,
    hand_str,
    parse_card,
    normalize_rank,
    normalize_suit,
    RANK_NAMES,
    SUIT_NAMES,
    SUIT_SYMBOLS,
)
from core.math_utils import (
    pot_odds,
    minimum_defense_frequency,
    calculate_proportions_z_test,
    wilson_score_interval,
    calculate_bb_per_100,
)

__all__ = [
    "Deck",
    "rank_of",
    "suit_of",
    "card_str",
    "hand_str",
    "parse_card",
    "normalize_rank",
    "normalize_suit",
    "RANK_NAMES",
    "SUIT_NAMES",
    "SUIT_SYMBOLS",
    "pot_odds",
    "minimum_defense_frequency",
    "calculate_proportions_z_test",
    "wilson_score_interval",
    "calculate_bb_per_100",
]
