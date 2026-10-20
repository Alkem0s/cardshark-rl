"""
games/five_card_draw/rules.py — Standard 5-Card Draw (High Poker) Evaluation Engine.

Hand categories (highest -> lowest):
    8 = Straight Flush
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
import numpy as np
from collections import Counter
from typing import List, Tuple, Optional
from core.card import rank_of, suit_of, card_str, hand_str, Deck, RANK_NAMES, SUIT_NAMES


def _rank_counts(ranks: List[int]) -> Counter:
    return Counter(ranks)

def _is_flush(suits: List[int]) -> bool:
    return len(set(suits)) == 1

def _is_straight(sorted_ranks: List[int]) -> bool:
    if sorted_ranks[-1] - sorted_ranks[0] == 4 and len(set(sorted_ranks)) == 5:
        return True
    # Wheel: A-2-3-4-5 -> ranks {0, 1, 2, 3, 12}
    if set(sorted_ranks) == {0, 1, 2, 3, 12}:
        return True
    return False

def _straight_high(sorted_ranks: List[int]) -> int:
    if set(sorted_ranks) == {0, 1, 2, 3, 12}:
        return 3  # 5-high straight
    return sorted_ranks[-1]

def evaluate_hand(cards: List[int]) -> int:
    """Return a single integer score for a 5-card hand. Higher is better."""
    assert len(cards) == 5, f"Need exactly 5 cards, got {len(cards)}"

    ranks = [rank_of(c) for c in cards]
    suits = [suit_of(c) for c in cards]
    sorted_ranks = sorted(ranks)
    counts = _rank_counts(ranks)

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


def hand_category(cards: List[int]) -> int:
    return evaluate_hand(cards) // 1_000_000


def hand_category_name(cards: List[int]) -> str:
    names = [
        "High Card", "One Pair", "Two Pair", "Three of a Kind",
        "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush",
    ]
    return names[hand_category(cards)]


def get_pairs_info(cards: List[int]) -> dict:
    ranks = [rank_of(c) for c in cards]
    counts = Counter(ranks)

    pair_ranks = [r for r, cnt in counts.items() if cnt >= 2]
    trip_ranks = [r for r, cnt in counts.items() if cnt >= 3]
    quad_ranks = [r for r, cnt in counts.items() if cnt == 4]

    best_rank = max(counts, key=lambda r: (counts[r], r))
    best_count = counts[best_rank]

    kept_count = 0
    kicker_indices = []
    for i, c in enumerate(cards):
        if rank_of(c) == best_rank and kept_count < best_count:
            kept_count += 1
        else:
            kicker_indices.append(i)

    if len(pair_ranks) >= 2:
        second_pair = sorted(pair_ranks)[-2]
        if second_pair != best_rank:
            kicker_indices = [
                i for i in range(5)
                if rank_of(cards[i]) != best_rank and rank_of(cards[i]) != second_pair
            ]

    return {
        "category": hand_category(cards),
        "pair_ranks": sorted(pair_ranks, reverse=True),
        "trip_ranks": sorted(trip_ranks, reverse=True),
        "quad_ranks": sorted(quad_ranks, reverse=True),
        "kicker_indices": kicker_indices,
    }


def has_flush_draw(cards: List[int]) -> Tuple[bool, Optional[int]]:
    suits = [suit_of(c) for c in cards]
    sc = Counter(suits)
    for s, cnt in sc.items():
        if cnt == 4:
            return True, s
    return False, None


def has_straight_draw(cards: List[int]) -> Tuple[bool, List[int]]:
    ranks = [rank_of(c) for c in cards]
    from itertools import combinations
    for combo_idx in combinations(range(5), 4):
        sub_ranks = sorted([ranks[i] for i in combo_idx])
        if len(set(sub_ranks)) < 4:
            continue
        span = sub_ranks[-1] - sub_ranks[0]
        if span <= 4 and len(set(sub_ranks)) == 4:
            return True, list(combo_idx)
    if 12 in ranks:
        low_ranks = [r for r in ranks if r <= 3]
        if len(low_ranks) >= 3:
            keep = [i for i in range(5) if rank_of(cards[i]) == 12 or rank_of(cards[i]) <= 3]
            if len(keep) >= 4:
                return True, keep[:4]
    return False, []


def normalize_rank(card: int) -> float:
    return rank_of(card) / 12.0


def normalize_suit(card: int) -> float:
    return suit_of(card) / 3.0


def normalize_hand_score(score: int) -> float:
    return min(score / 9_000_000, 1.0)


format_hand = hand_str
