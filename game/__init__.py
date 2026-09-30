"""
game package — Pure 5-Card Draw Poker Game Simulation Engine.

Contains zero PyTorch / RL / Stable-Baselines3 code.
Pure game rules, deck/card evaluation, all-in side-pot math, and heuristic opponent bots.
"""

from game.card_utils import (
    Deck,
    evaluate_hand,
    hand_category,
    hand_category_name,
    rank_of,
    suit_of,
    normalize_rank,
    normalize_suit,
    normalize_hand_score,
)
from game.side_pot import (
    calculate_side_pots,
    resolve_showdown_payouts,
    PotTier,
)
from game.draw_poker_env import DrawPokerEnv
from game.multi_draw_poker_env import MultiDrawPokerEnv
from game.multi_opponents import (
    MultiPlayerOpponent,
    CallingStation,
    Maniac,
    Rock,
    TAG,
    LAG,
    AdversarialExploiter,
    make_random_archetype,
    make_opponent_by_id,
    ARCHETYPE_CLASSES,
    NUM_ARCHETYPES,
)

__all__ = [
    "Deck",
    "evaluate_hand",
    "hand_category",
    "hand_category_name",
    "rank_of",
    "suit_of",
    "normalize_rank",
    "normalize_suit",
    "normalize_hand_score",
    "calculate_side_pots",
    "resolve_showdown_payouts",
    "PotTier",
    "DrawPokerEnv",
    "MultiDrawPokerEnv",
    "MultiPlayerOpponent",
    "CallingStation",
    "Maniac",
    "Rock",
    "TAG",
    "LAG",
    "AdversarialExploiter",
    "make_random_archetype",
    "make_opponent_by_id",
    "ARCHETYPE_CLASSES",
    "NUM_ARCHETYPES",
    "CardSharkNPC",
]

def __getattr__(name: str):
    if name == "CardSharkNPC":
        from game.npc import CardSharkNPC
        return CardSharkNPC
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
