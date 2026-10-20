"""
games/triple_draw_27/rules.py — 2-7 Lowball (Deuce-to-Seven) Hand Evaluation Engine.

Rules of Deuce-to-Seven Lowball:
1. Lowest hand wins at showdown.
2. Aces are strictly high (rank 12 / value 14). They do not wrap to low (A-2-3-4-5 is NOT a straight).
3. Straights and flushes penalize the hand (they are made poker hands and lose to non-straights/non-flushes).
4. The absolute best hand ("The Wheel" / "Number 1") is:
       7 - 5 - 4 - 3 - 2   (unsuited)
5. Hand Tier Hierarchy (Higher Tier = Better Lowball Hand):
       Tier 8: No Pair Low (unpaired, unsuited, non-straight) — [7-high down to A-high]
       Tier 7: One Pair — [Pair of 2s down to Pair of Aces]
       Tier 6: Two Pair — [2s & 3s down to Aces & Kings]
       Tier 5: Three of a Kind — [Trip 2s down to Trip Aces]
       Tier 4: Straight — [6-high straight down to A-high straight]
       Tier 3: Flush — [7-high flush down to A-high flush]
       Tier 2: Full House — [22233 down to AAAKK]
       Tier 1: Four of a Kind — [Quad 2s down to Quad Aces]
       Tier 0: Straight Flush — [6-high SF down to Royal/Ace-high SF]

Scoring:
    Score = Tier * 1_000_000 + Tiebreaker
    Higher Score = Winning Lowball Hand.
"""

from __future__ import annotations
from collections import Counter
from typing import List, Tuple
from core.card import rank_of, suit_of, card_str


def _is_flush(suits: List[int]) -> bool:
    return len(set(suits)) == 1


def _is_straight(sorted_ranks: List[int]) -> bool:
    """
    Checks for a 5-card straight in 2-7 Lowball.
    Aces are strictly high (12), so A-2-3-4-5 is NOT a straight.
    Only consecutive ranks where ranks[4] - ranks[0] == 4 and all 5 unique are straights.
    """
    if len(set(sorted_ranks)) != 5:
        return False
    return (sorted_ranks[4] - sorted_ranks[0]) == 4


def evaluate_27_hand(cards: List[int]) -> int:
    """
    Computes a canonical integer score for a 5-card hand under 2-7 Lowball rules.
    HIGHER integer score = BETTER lowball hand (Winner).
    
    Score = Tier * 1_000_000 + tiebreaker
    """
    assert len(cards) == 5, f"Need exactly 5 cards, got {len(cards)}"

    ranks = [rank_of(c) for c in cards]
    suits = [suit_of(c) for c in cards]
    sorted_ranks = sorted(ranks)  # ascending: r0 <= r1 <= r2 <= r3 <= r4
    counts = Counter(ranks)

    flush = _is_flush(suits)
    straight = _is_straight(sorted_ranks)

    # Invert ranks so smaller card ranks produce larger values (0 -> 12, 12 -> 0)
    # Ranks: 0=2, 1=3, 2=4, 3=5, 4=6, 5=7, ..., 12=Ace
    # Inverted: 2 has value 12, 7 has value 7, Ace has value 0.
    def inv(r: int) -> int:
        return 12 - r

    # Group ranks by frequency descending, then by rank ascending (lower card = better group)
    # e.g., for Two Pair (2,2) and (K,K), 2s are better than Ks.
    # Sorting by frequency desc, then inv(rank) desc (which means rank asc)
    groups = sorted(counts.items(), key=lambda x: (x[1], inv(x[0])), reverse=True)

    def _kicker_value(group_list):
        val = 0
        base = 14
        for i, (r, _cnt) in enumerate(group_list):
            val += inv(r) * (base ** (len(group_list) - 1 - i))
        return val

    # 1. Straight Flush (Tier 0 — Absolute worst)
    if straight and flush:
        # Lower straight is better in lowball: 6-high SF beats 7-high SF
        return 0_000_000 + inv(sorted_ranks[4])

    # 2. Four of a Kind (Tier 1)
    if groups[0][1] == 4:
        return 1_000_000 + _kicker_value(groups)

    # 3. Full House (Tier 2)
    if groups[0][1] == 3 and groups[1][1] == 2:
        return 2_000_000 + _kicker_value(groups)

    # 4. Flush (Tier 3)
    if flush:
        # Sorted descending ranks inverted: highest card is most significant kicker
        sorted_desc = sorted(ranks, reverse=True)
        return 3_000_000 + _kicker_value([(r, 1) for r in sorted_desc])

    # 5. Straight (Tier 4)
    if straight:
        # Lower straight is better: 6-high straight (inv 12-4 = 8) beats 7-high (inv 12-5 = 7)
        return 4_000_000 + inv(sorted_ranks[4])

    # 6. Three of a Kind (Tier 5)
    if groups[0][1] == 3:
        return 5_000_000 + _kicker_value(groups)

    # 7. Two Pair (Tier 6)
    if groups[0][1] == 2 and groups[1][1] == 2:
        return 6_000_000 + _kicker_value(groups)

    # 8. One Pair (Tier 7)
    if groups[0][1] == 2:
        return 7_000_000 + _kicker_value(groups)

    # 9. No Pair Low (Tier 8 — The Best Tier in 2-7 Lowball)
    # All 5 cards unique, no straight, no flush.
    # Highest card is most important tiebreaker, evaluated descending
    sorted_desc = sorted(ranks, reverse=True)
    return 8_000_000 + _kicker_value([(r, 1) for r in sorted_desc])


def hand_category_27(cards: List[int]) -> int:
    """Returns Tier index 0..8 for a 2-7 hand (8=No Pair Low, 0=Straight Flush)."""
    return evaluate_27_hand(cards) // 1_000_000


def hand_category_name_27(cards: List[int]) -> str:
    names = [
        "Straight Flush",
        "Four of a Kind",
        "Full House",
        "Flush",
        "Straight",
        "Three of a Kind",
        "Two Pair",
        "One Pair",
        "No Pair Low",
    ]
    tier = hand_category_27(cards)
    if tier == 8:
        # For No Pair Low, describe by highest card, e.g. "7-Low", "8-Low"
        highest_rank = max(rank_of(c) for c in cards)
        from core.card import RANK_NAMES
        return f"{RANK_NAMES[highest_rank]}-High Low"
    return names[tier]


def is_wheel_27(cards: List[int]) -> bool:
    """Returns True if the hand is the absolute nuts: 7-5-4-3-2 unsuited."""
    ranks = sorted(rank_of(c) for c in cards)
    if ranks == [0, 1, 2, 3, 5]:  # 2, 3, 4, 5, 7
        suits = [suit_of(c) for c in cards]
        return not _is_flush(suits)
    return False


def get_lowball_draw_candidates(cards: List[int]) -> List[int]:
    """
    Analyzes cards and returns a recommended list of card indices (0..4) to discard.
    Discards:
      - Any cards involved in pairs, trips, or quads (keeping only 1)
      - Aces, Kings, Queens, Jacks, Tens
      - Cards that would complete an unwanted straight or flush
    """
    ranks = [rank_of(c) for c in cards]
    counts = Counter(ranks)

    discard_indices = []
    kept_ranks = set()

    for idx, card in enumerate(cards):
        r = rank_of(card)
        if r in kept_ranks:
            # Duplicate rank (pair) -> discard
            discard_indices.append(idx)
        elif r >= 7:  # 9, T, J, Q, K, A -> discard high cards
            discard_indices.append(idx)
        else:
            kept_ranks.add(r)

    # Check for accidental flush
    if len(cards) - len(discard_indices) >= 4:
        kept_suits = [suit_of(cards[i]) for i in range(5) if i not in discard_indices]
        if len(set(kept_suits)) == 1:
            # 4 or 5 suited low cards -> discard one of them to avoid flush penalty
            # Discard highest rank suited card
            highest_kept_idx = max(
                (i for i in range(5) if i not in discard_indices),
                key=lambda i: rank_of(cards[i])
            )
            discard_indices.append(highest_kept_idx)

    return sorted(discard_indices)
