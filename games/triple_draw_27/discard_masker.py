"""
games/triple_draw_27/discard_masker.py — Dominated Discard Action Masking for 2-7 Lowball.

In 2-7 Triple Draw, there are 32 discrete discard bitmasks (0 to 31):
    bitmask = sum(1 << i for i in discards)
    bitmask 0 = Stand Pat (discard 0 cards)
    bitmask 31 = Discard all 5 cards

Dominated Discard Logic for 2-7 Lowball:
1. Low Cards (2, 3, 4, 5, 7): Hero should almost never discard unique cards in the golden range [2..7]
   to keep higher cards.
2. High Cards (T, J, Q, K, A): Keeping high cards while discarding low cards is strictly dominated.
3. Duplicates (Pairs / Trips): Keeping duplicate ranks is dominated (pairs hurt the hand).
   At most one card of each rank should ever be retained.
4. Flushes: If hero holds 5 suited low cards (e.g. 7s 5s 4s 3s 2s), standing pat is a flush penalty.
   Hero should break the flush by discarding one card.
"""

from __future__ import annotations
import numpy as np
from collections import Counter
from typing import List, Set
from core.card import rank_of, suit_of
from games.triple_draw_27.rules import hand_category_27, is_wheel_27


def get_legal_discard_mask_27(cards: List[int]) -> np.ndarray:
    """
    Returns a boolean mask of length 32 for the 32 discard bitmasks.
    1 = Legal (plausible lowball draw), 0 = Masked (dominated).
    Guarantees at least one discard action (or stand pat) is always legal.
    """
    mask = np.zeros(32, dtype=np.int8)
    ranks = [rank_of(c) for c in cards]
    suits = [suit_of(c) for c in cards]
    counts = Counter(ranks)

    tier = hand_category_27(cards)
    highest_rank = max(ranks)

    # 1. Standing Pat (bitmask 0):
    # Standing pat is legal if hero has an 8-or-better low (Tier 8 with high card <= 6 (which is an 8)),
    # or even a 9-low on the 3rd draw.
    if tier == 8 and highest_rank <= 7:  # 9-high or better made low
        mask[0] = 1

    # If hero has the Wheel, standing pat is the ONLY optimal action
    if is_wheel_27(cards):
        mask[0] = 1
        return mask

    # 2. Evaluate each of the 32 discard combinations
    # Bitmask i: if (i & (1 << j)), card j is discarded; else kept.
    # A candidate draw is viable if:
    #   a) It does not keep duplicate ranks (e.g. keeping a pair is dominated).
    #   b) It does not keep a card >= rank 8 (T, J, Q, K, A) while discarding a card <= 5 (2, 3, 4, 5, 7).
    #   c) It does not keep all cards of the same suit if all 5 cards are suited (avoiding made flushes).

    for bitmask in range(32):
        discards = [j for j in range(5) if (bitmask & (1 << j))]
        kept = [j for j in range(5) if not (bitmask & (1 << j))]

        # If discarding all 5 cards, it's always legal as a panic reset
        if bitmask == 31:
            mask[bitmask] = 1
            continue

        if bitmask == 0 and mask[0] == 1:
            continue

        kept_ranks = [ranks[j] for j in kept]

        # Condition A: Never keep duplicate ranks
        if len(set(kept_ranks)) != len(kept_ranks):
            continue

        # Condition B: Never discard a prime low card [2, 3, 4, 5, 7] while keeping a high card [T, J, Q, K, A]
        discarded_ranks = [ranks[j] for j in discards]
        kept_high_ranks = [r for r in kept_ranks if r >= 8]  # T, J, Q, K, A
        discarded_low_ranks = [r for r in discarded_ranks if r <= 5 and counts[r] == 1]  # unique 2, 3, 4, 5, 7

        if len(kept_high_ranks) > 0 and len(discarded_low_ranks) > 0:
            continue

        # Condition C: Avoid keeping 5 suited cards if they would force an unwanted flush
        if len(kept) == 5:
            kept_suits = [suits[j] for j in kept]
            if len(set(kept_suits)) == 1:
                # 5 suited cards = made flush
                continue

        mask[bitmask] = 1

    # Safety: If all bitmasks were somehow pruned, enable the recommended discard candidates
    if not np.any(mask):
        from games.triple_draw_27.rules import get_lowball_draw_candidates
        cands = get_lowball_draw_candidates(cards)
        fallback_mask = sum((1 << j) for j in cands)
        mask[fallback_mask] = 1
        mask[0] = 1  # Also allow pat

    return mask
