"""
games/five_card_draw package — 5-Card Draw (High Poker) Game Engine & Environment.

Provides:
- Hand rules & card evaluators (rules.py)
- 5-seat multi-player table engine (env.py)
- Limit heads-up 1v1 table engine (heads_up_env.py)
- Heuristic opponent archetypes (opponents.py)
- Side-pot calculator and showdown payouts (side_pot.py)
- Neural and heuristic NPC decision engine (npc.py)
"""

from games.five_card_draw.rules import (
    evaluate_hand,
    hand_category,
    hand_category_name,
    get_pairs_info,
    normalize_hand_score,
    normalize_rank,
    normalize_suit,
    has_flush_draw,
    has_straight_draw,
    rank_of,
    suit_of,
    card_str,
    hand_str,
    format_hand,
    Deck,
)
from games.five_card_draw.side_pot import (
    calculate_side_pots,
    resolve_showdown_payouts,
    PotTier,
)
from games.five_card_draw.env import MultiDrawPokerEnv, PlayerState
from games.five_card_draw.heads_up_env import DrawPokerEnv
from games.five_card_draw.opponents import (
    MultiPlayerOpponent,
    CallingStation,
    Maniac,
    Rock,
    TAG,
    LAG,
    AdversarialExploiter,
    RandomCyclingBot,
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
    "card_str",
    "hand_str",
    "format_hand",
    "normalize_rank",
    "normalize_suit",
    "normalize_hand_score",
    "has_flush_draw",
    "has_straight_draw",
    "calculate_side_pots",
    "resolve_showdown_payouts",
    "PotTier",
    "DrawPokerEnv",
    "MultiDrawPokerEnv",
    "PlayerState",
    "MultiPlayerOpponent",
    "CallingStation",
    "Maniac",
    "Rock",
    "TAG",
    "LAG",
    "AdversarialExploiter",
    "RandomCyclingBot",
    "make_random_archetype",
    "make_opponent_by_id",
    "ARCHETYPE_CLASSES",
    "NUM_ARCHETYPES",
    "CardSharkNPC",
]

def __getattr__(name: str):
    if name == "CardSharkNPC":
        from games.five_card_draw.npc import CardSharkNPC
        return CardSharkNPC
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
