"""
card_utils.py — Integer-based deck and lightweight 5-card hand evaluator.

Card encoding:
    card  = 0..51
    rank  = card % 13   (0=2, 1=3, ..., 8=T, 9=J, 10=Q, 11=K, 12=A)
    suit  = card // 13  (0=Clubs, 1=Diamonds, 2=Hearts, 3=Spades)

Hand categories (highest → lowest):
    8 = Straight Flush
    7 = Four of a Kind
    6 = Full House
    5 = Flush
    4 = Straight
    3 = Three of a Kind
    2 = Two Pair
    1 = One Pair
    0 = High Card

Score = category * 1_000_000 + tiebreaker
Higher score wins.
"""

from __future__ import annotations
import numpy as np
from collections import Counter
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Card helpers
# ---------------------------------------------------------------------------

RANK_NAMES = "23456789TJQKA"
SUIT_NAMES = "cdhs"

def rank_of(card: int) -> int:
    """Return rank 0‑12 (2..A)."""
    return card % 13

def suit_of(card: int) -> int:
    """Return suit 0‑3."""
    return card // 13

def card_str(card: int) -> str:
    """Human-readable card string, e.g. 'As', 'Td'."""
    return RANK_NAMES[rank_of(card)] + SUIT_NAMES[suit_of(card)]

def hand_str(cards: List[int]) -> str:
    """Pretty-print a hand."""
    return " ".join(card_str(c) for c in sorted(cards, key=rank_of, reverse=True))


# ---------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------

class Deck:
    """Fast integer deck with NumPy shuffle."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self):
        self._cards = np.arange(52, dtype=np.int32)
        self.rng.shuffle(self._cards)
        self._idx = 0

    def deal(self, n: int) -> List[int]:
        """Deal *n* cards from the top."""
        cards = self._cards[self._idx : self._idx + n].tolist()
        self._idx += n
        return cards

    def draw(self, n: int) -> List[int]:
        """Draw *n* replacement cards (alias for deal)."""
        return self.deal(n)

    @property
    def remaining(self) -> int:
        return 52 - self._idx


# ---------------------------------------------------------------------------
# Hand evaluation
# ---------------------------------------------------------------------------

def _rank_counts(ranks: List[int]) -> Counter:
    return Counter(ranks)

def _is_flush(suits: List[int]) -> bool:
    return len(set(suits)) == 1

def _is_straight(sorted_ranks: List[int]) -> bool:
    """Check for a 5-card straight.  Handles A-2-3-4-5 (wheel)."""
    # Normal straight check
    if sorted_ranks[-1] - sorted_ranks[0] == 4 and len(set(sorted_ranks)) == 5:
        return True
    # Wheel: A-2-3-4-5  →  ranks {0,1,2,3,12}
    if set(sorted_ranks) == {0, 1, 2, 3, 12}:
        return True
    return False

def _straight_high(sorted_ranks: List[int]) -> int:
    """Return high card of the straight (3 for wheel)."""
    if set(sorted_ranks) == {0, 1, 2, 3, 12}:
        return 3  # 5-high straight (wheel)
    return sorted_ranks[-1]

def evaluate_hand(cards: List[int]) -> int:
    """Return a single integer score for a 5-card hand.  Higher is better.

    Score = category * 1_000_000 + tiebreaker
    Tiebreaker uses base-14 encoding with at most 5 positions,
    guaranteeing max tiebreaker = 13*14^4 + 13*14^3 + ... < 760_000 < 1M.
    """
    assert len(cards) == 5, f"Need exactly 5 cards, got {len(cards)}"

    ranks = [rank_of(c) for c in cards]
    suits = [suit_of(c) for c in cards]
    sorted_ranks = sorted(ranks)
    counts = _rank_counts(ranks)

    flush = _is_flush(suits)
    straight = _is_straight(sorted_ranks)

    # Group ranks by count descending, then by rank descending within groups
    # e.g. Full House KKK44 → groups = [(K, 3), (4, 2)]
    groups = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

    def _kicker_value(group_list):
        """Encode groups into a tiebreaker value using base-14."""
        val = 0
        base = 14  # > 13 ranks so each position is unambiguous
        for i, (rank, _cnt) in enumerate(group_list):
            val += rank * (base ** (len(group_list) - 1 - i))
        return val

    kicker = _kicker_value(groups)

    if straight and flush:
        high = _straight_high(sorted_ranks)
        return 8_000_000 + high
    if groups[0][1] == 4:  # Four of a kind
        return 7_000_000 + kicker
    if groups[0][1] == 3 and groups[1][1] == 2:  # Full house
        return 6_000_000 + kicker
    if flush:
        return 5_000_000 + kicker
    if straight:
        high = _straight_high(sorted_ranks)
        return 4_000_000 + high
    if groups[0][1] == 3:  # Three of a kind
        return 3_000_000 + kicker
    if groups[0][1] == 2 and groups[1][1] == 2:  # Two pair
        return 2_000_000 + kicker
    if groups[0][1] == 2:  # One pair
        return 1_000_000 + kicker
    return kicker  # High card


def hand_category(cards: List[int]) -> int:
    """Return the category index 0‑8 for a hand."""
    return evaluate_hand(cards) // 1_000_000


def hand_category_name(cards: List[int]) -> str:
    names = [
        "High Card", "One Pair", "Two Pair", "Three of a Kind",
        "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush",
    ]
    return names[hand_category(cards)]


# ---------------------------------------------------------------------------
# Draw-phase helpers (used by opponent archetypes)
# ---------------------------------------------------------------------------

def get_pairs_info(cards: List[int]) -> dict:
    """Analyse a hand for the draw phase.

    Returns dict with keys:
        'category': int 0-8
        'pair_ranks': list of ranks that appear >= 2 times
        'trip_ranks': list of ranks that appear >= 3 times
        'quad_ranks': list of ranks that appear == 4 times
        'kicker_indices': indices of cards that are NOT part of the best group
    """
    ranks = [rank_of(c) for c in cards]
    counts = Counter(ranks)

    pair_ranks = [r for r, cnt in counts.items() if cnt >= 2]
    trip_ranks = [r for r, cnt in counts.items() if cnt >= 3]
    quad_ranks = [r for r, cnt in counts.items() if cnt == 4]

    # Determine which card indices are "kickers" (not part of the best group)
    best_rank = max(counts, key=lambda r: (counts[r], r))
    best_count = counts[best_rank]

    # Keep cards matching the best group; rest are kickers
    kept_count = 0
    kicker_indices = []
    for i, c in enumerate(cards):
        if rank_of(c) == best_rank and kept_count < best_count:
            kept_count += 1
        else:
            kicker_indices.append(i)

    # For two pair, also keep the second pair
    if len(pair_ranks) >= 2:
        second_pair = sorted(pair_ranks)[-2] if len(pair_ranks) >= 2 else None
        if second_pair is not None and second_pair != best_rank:
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


def has_flush_draw(cards: List[int]) -> Tuple[bool, int | None]:
    """Check if 4 of 5 cards share a suit.  Returns (True, suit) or (False, None)."""
    suits = [suit_of(c) for c in cards]
    sc = Counter(suits)
    for s, cnt in sc.items():
        if cnt == 4:
            return True, s
    return False, None


def has_straight_draw(cards: List[int]) -> Tuple[bool, List[int]]:
    """Check if 4 of 5 cards could form a straight (open-ended or gutshot).

    Returns (True, list_of_4_card_indices_to_keep) or (False, []).
    """
    ranks = [rank_of(c) for c in cards]
    unique_ranks = sorted(set(ranks))

    # Try every combination of 4 cards
    from itertools import combinations
    for combo_idx in combinations(range(5), 4):
        sub_ranks = sorted([ranks[i] for i in combo_idx])
        if len(set(sub_ranks)) < 4:
            continue
        span = sub_ranks[-1] - sub_ranks[0]
        # Open-ended: span == 3, Gutshot: span == 4 with a gap
        if span <= 4 and len(set(sub_ranks)) == 4:
            return True, list(combo_idx)
    # Wheel draw: check for A + subset of {2,3,4,5}
    if 12 in ranks:
        low_ranks = [r for r in ranks if r <= 3]
        if len(low_ranks) >= 3:
            keep = [i for i in range(5) if rank_of(cards[i]) == 12 or rank_of(cards[i]) <= 3]
            if len(keep) >= 4:
                return True, keep[:4]
    return False, []


# ---------------------------------------------------------------------------
# Normalisation for observation space
# ---------------------------------------------------------------------------

MAX_HAND_SCORE = evaluate_hand([8, 21, 34, 47, 12])  # Royal flush approx upper bound

def normalize_card(card: int) -> float:
    """Normalise card integer to [0, 1]."""
    return card / 51.0

def normalize_hand_score(score: int) -> float:
    """Normalise hand score to approx [0, 1]."""
    return min(score / 9_000_000, 1.0)
