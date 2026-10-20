"""
games/holdem/evaluator.py — Standard 7-Card Hand Evaluator for Texas Hold'em.

Finds the highest-ranking 5-card poker hand out of 5, 6, or 7 available cards.
Uses standard poker hand categories:
    8 = Straight Flush (Royal Flush is highest straight flush)
    7 = Four of a Kind
    6 = Full House
    5 = Flush
    4 = Straight
    3 = Three of a Kind
    2 = Two Pair
    1 = One Pair
    0 = High Card
"""

from __future__ import annotations
import itertools
from typing import List, Tuple
from collections import Counter
from core.card import rank_of, suit_of, RANK_NAMES, SUIT_NAMES, card_str


CATEGORY_NAMES = [
    "High Card",
    "One Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
]


def _is_flush(suits: List[int]) -> bool:
    return len(set(suits)) == 1


def _is_straight(sorted_ranks: List[int]) -> bool:
    if sorted_ranks[-1] - sorted_ranks[0] == 4 and len(set(sorted_ranks)) == 5:
        return True
    # Ace-low wheel: A-2-3-4-5 -> ranks {0, 1, 2, 3, 12}
    if set(sorted_ranks) == {0, 1, 2, 3, 12}:
        return True
    return False


def _straight_high(sorted_ranks: List[int]) -> int:
    if set(sorted_ranks) == {0, 1, 2, 3, 12}:
        return 3  # 5-high straight
    return sorted_ranks[-1]


def evaluate_5card_hand(cards: List[int]) -> int:
    """Return single integer score for a 5-card hand. Higher is better."""
    assert len(cards) == 5, f"Expected 5 cards, got {len(cards)}"
    ranks = [rank_of(c) for c in cards]
    suits = [suit_of(c) for c in cards]
    sorted_ranks = sorted(ranks)
    counts = Counter(ranks)

    flush = _is_flush(suits)
    straight = _is_straight(sorted_ranks)
    groups = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

    def _kicker_value(group_list):
        val = 0
        base = 14
        for i, (rank, _cnt) in enumerate(group_list):
            val += rank * (base ** (len(group_list) - 1 - i))
        return val

    kicker = _kicker_value(groups)

    if straight and flush:
        high = _straight_high(sorted_ranks)
        return 8_000_000 + high
    if groups[0][1] == 4:
        return 7_000_000 + kicker
    if groups[0][1] == 3 and groups[1][1] == 2:
        return 6_000_000 + kicker
    if flush:
        return 5_000_000 + kicker
    if straight:
        high = _straight_high(sorted_ranks)
        return 4_000_000 + high
    if groups[0][1] == 3:
        return 3_000_000 + kicker
    if groups[0][1] == 2 and groups[1][1] == 2:
        return 2_000_000 + kicker
    if groups[0][1] == 2:
        return 1_000_000 + kicker
    return kicker


def evaluate_holdem_hand(cards: List[int]) -> Tuple[int, List[int]]:
    """
    Evaluate 5, 6, or 7 cards (hole cards + board).
    Returns (best_score, best_5_cards).
    """
    n = len(cards)
    if n < 5:
        raise ValueError(f"Need at least 5 cards to evaluate, got {n}")
    if n == 5:
        return evaluate_5card_hand(cards), cards

    best_score = -1
    best_combo: List[int] = []
    for combo in itertools.combinations(cards, 5):
        combo_list = list(combo)
        score = evaluate_5card_hand(combo_list)
        if score > best_score:
            best_score = score
            best_combo = combo_list

    return best_score, best_combo


def holdem_category(cards: List[int]) -> int:
    score, _ = evaluate_holdem_hand(cards)
    return score // 1_000_000


def holdem_category_name(cards: List[int]) -> str:
    cat = holdem_category(cards)
    return CATEGORY_NAMES[cat]
