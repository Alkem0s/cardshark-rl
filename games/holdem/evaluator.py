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


def evaluate_preflop_hand_chen(c1: int, c2: int) -> float:
    """
    Computes Bill Chen's preflop hand formula for Texas Hold'em hole cards.
    Returns normalized equity in [0.0, 1.0].
    
    Chen Formula:
    1. Score highest card:
       A: 10, K: 8, Q: 7, J: 6, T: 5, 9: 4.5, 8: 4, 7: 3.5, 6: 3, 5: 2.5, 4: 2, 3: 1.5, 2: 1
    2. If pair, double score with a minimum of 5 (e.g. 22=5, 55=5, AA=20).
    3. If suited, add 2 points.
    4. Gaps between ranks:
       0 gap (connected): add 1 point (if high card < A)
       1 gap (e.g. Q-T, J-9): -1 point
       2 gap (e.g. Q-9, J-8): -2 points
       3 gap (e.g. Q-8, J-7): -4 points
       >= 4 gap (e.g. K-2, Q-2): -5 points
    5. Result clamped to [0, 20], normalized to [0.0, 1.0].
    """
    CHEN_CARD_SCORES = {
        12: 10.0,  # Ace
        11: 8.0,   # King
        10: 7.0,   # Queen
        9: 6.0,    # Jack
        8: 5.0,    # Ten
        7: 4.5,    # 9
        6: 4.0,    # 8
        5: 3.5,    # 7
        4: 3.0,    # 6
        3: 2.5,    # 5
        2: 2.0,    # 4
        1: 1.5,    # 3
        0: 1.0,    # 2
    }
    r1, r2 = rank_of(c1), rank_of(c2)
    s1, s2 = suit_of(c1), suit_of(c2)
    high_r = max(r1, r2)
    low_r = min(r1, r2)
    gap = high_r - low_r - 1

    if r1 == r2:
        score = max(5.0, CHEN_CARD_SCORES[high_r] * 2.0)
    else:
        score = CHEN_CARD_SCORES[high_r]
        if s1 == s2:
            score += 2.0  # Suited bonus
        
        if gap == -1 or gap == 0:  # Connected
            if high_r < 12:  # bonus for connectedness below Ace
                score += 1.0
        elif gap == 1:
            score -= 1.0
        elif gap == 2:
            score -= 2.0
        elif gap == 3:
            score -= 4.0
        else:
            score -= 5.0

    score = max(0.0, min(20.0, score))
    return score / 20.0

