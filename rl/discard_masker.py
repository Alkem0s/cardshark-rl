"""
rl/discard_masker.py — Dominated Discard Action Masking Engine for Model E.

Eliminates mathematically dominated and irrational discard actions during the draw phase:
1. Category >= 4 (Straight, Flush, Full House, Quads, Straight Flush):
   - Strict Stand Pat: only bitmask 0 (0 discards) is legal. Breaking a made hand is strictly illegal.
2. Category == 3 (Three of a Kind):
   - Discarding any of the 3 trip cards is strictly illegal.
   - Legal actions: discard 2 kickers (standard), discard 1 kicker (deception), stand pat (bluff).
3. Category == 2 (Two Pair):
   - Discarding any of the 4 paired cards is strictly illegal.
   - Legal actions: discard 1 kicker (standard), stand pat (deception).
4. Category == 1 (One Pair):
   - Discarding either of the 2 paired cards is strictly illegal.
   - Legal actions: discard 3 kickers (standard), discard 2 kickers (kicker hold), stand pat.
5. Category == 0 (High Card):
   - 4-Flush / 4-Straight draws: discard the 1 non-connecting card.
   - General High Card: keep 1-2 highest cards, discard 5, or stand pat.
   - Discarding high cards while holding lower rank kickers is strictly illegal.
"""

from __future__ import annotations
from typing import List, Set, Tuple
import numpy as np

from game.card_utils import (
    hand_category,
    get_pairs_info,
    has_flush_draw,
    has_straight_draw,
    rank_of,
    suit_of,
)

NUM_DRAW_ACTIONS = 32  # 2^5 bitmask combinations


def get_legal_discard_bitmasks(hand: List[int]) -> Set[int]:
    """Returns the set of mathematically sound discard bitmasks (0 to 31) for a 5-card hand."""
    if not hand or len(hand) != 5:
        # Fallback: all bitmasks legal
        return set(range(NUM_DRAW_ACTIONS))

    cat = hand_category(hand)
    legal_masks: Set[int] = set()

    # 1. Made Monster Hands (Straight, Flush, Full House, Quads, Straight Flush)
    if cat >= 4:
        # Stand pat is strictly dominant. Never break a made hand.
        legal_masks.add(0)
        return legal_masks

    # 2. Three of a Kind (Trips)
    if cat == 3:
        info = get_pairs_info(hand)
        kickers = info["kicker_indices"]
        # Always allow Stand Pat (0) for deception
        legal_masks.add(0)
        if len(kickers) == 2:
            k1, k2 = kickers[0], kickers[1]
            # Standard draw 2: discard both kickers
            legal_masks.add((1 << k1) | (1 << k2))
            # Deception draw 1: discard one kicker
            legal_masks.add(1 << k1)
            legal_masks.add(1 << k2)
        elif len(kickers) == 1:
            legal_masks.add(1 << kickers[0])
        return legal_masks

    # 3. Two Pair
    if cat == 2:
        info = get_pairs_info(hand)
        kickers = info["kicker_indices"]
        # Stand pat (0) for deception
        legal_masks.add(0)
        if len(kickers) == 1:
            # Standard draw 1: discard the single kicker
            legal_masks.add(1 << kickers[0])
        return legal_masks

    # 4. One Pair
    if cat == 1:
        info = get_pairs_info(hand)
        kickers = info["kicker_indices"]
        # Stand pat (0) for bluff
        legal_masks.add(0)
        if len(kickers) == 3:
            k1, k2, k3 = kickers[0], kickers[1], kickers[2]
            # Standard GTO draw 3: discard all 3 kickers
            legal_masks.add((1 << k1) | (1 << k2) | (1 << k3))
            # Draw 2: keep the pair + 1 kicker
            legal_masks.add((1 << k1) | (1 << k2))
            legal_masks.add((1 << k1) | (1 << k3))
            legal_masks.add((1 << k2) | (1 << k3))
        return legal_masks

    # 5. High Card
    # Check 4-Flush draw
    has_flush, flush_suit = has_flush_draw(hand)
    if has_flush and flush_suit is not None:
        off_cards = [i for i in range(5) if suit_of(hand[i]) != flush_suit]
        if len(off_cards) == 1:
            legal_masks.add(1 << off_cards[0])
        legal_masks.add(0)  # Stand pat bluff
        return legal_masks

    # Check 4-Straight draw
    has_straight, keep_indices = has_straight_draw(hand)
    if has_straight and len(keep_indices) == 4:
        off_cards = [i for i in range(5) if i not in keep_indices]
        if len(off_cards) == 1:
            legal_masks.add(1 << off_cards[0])
        legal_masks.add(0)  # Stand pat bluff
        return legal_masks

    # General High Card:
    # Stand pat bluff
    legal_masks.add(0)
    # Discard all 5 (draw 5)
    legal_masks.add((1 << 5) - 1)  # 31

    # Keep 1 highest card (draw 4) or keep 2 highest cards (draw 3)
    ranks_with_idx = [(rank_of(hand[i]), i) for i in range(5)]
    ranks_with_idx.sort(reverse=True)

    # Keep highest card, discard other 4
    top1_keep = {ranks_with_idx[0][1]}
    mask_draw4 = sum((1 << i) for i in range(5) if i not in top1_keep)
    legal_masks.add(mask_draw4)

    # Keep top 2 highest cards, discard other 3
    top2_keep = {ranks_with_idx[0][1], ranks_with_idx[1][1]}
    mask_draw3 = sum((1 << i) for i in range(5) if i not in top2_keep)
    legal_masks.add(mask_draw3)

    return legal_masks


def get_legal_discard_mask(hand: List[int]) -> np.ndarray:
    """
    Returns a 32-element boolean array where index `bitmask` is True if
    the discard action is rational, and False if strictly dominated.
    """
    mask = np.zeros(NUM_DRAW_ACTIONS, dtype=bool)
    legal_bitmasks = get_legal_discard_bitmasks(hand)
    for b in legal_bitmasks:
        if 0 <= b < NUM_DRAW_ACTIONS:
            mask[b] = True
    # Safety fallback: ensure at least Stand Pat (0) is legal
    if not np.any(mask):
        mask[0] = True
    return mask
