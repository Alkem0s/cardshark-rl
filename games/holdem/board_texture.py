"""
games/holdem/board_texture.py — Board Texture & Dynamic Hand-Board Interaction Engine.

Computes 17 bounded [-1.0, 1.0] features capturing:
1. Board Wetness & Texture:
   - Board high card rank
   - Monotone (flush danger)
   - Two-tone
   - Paired board
   - Connectedness (straight danger)
2. Hero Made Hand Hierarchy:
   - Overpair
   - Top pair (and kicker strength)
   - Mid/bottom pair
   - Two pair
   - Set (hidden 3-of-a-kind) or trips
   - Straight or better
3. Hero Draws & Outs (Equity Engine):
   - 4-card flush draw
   - Nut flush draw (holding the Ace)
   - Open-Ended Straight Draw (OESD - 8 outs)
   - Inside gutshot straight draw (4 outs)
   - Overcards to board (3 outs each)
   - Rule-of-2/4 drawing equity estimate
"""

from __future__ import annotations
from typing import List, Tuple, Set, Optional
from collections import Counter
import numpy as np

from core.card import rank_of, suit_of, normalize_rank

NUM_TEXTURE_FEATURES = 17


def evaluate_board_texture_and_draws(
    hole_cards: List[int],
    board_cards: List[int],
) -> np.ndarray:
    """
    Computes a 17-dimensional vector of board texture and hero interaction features.
    All values are bounded in [-1.0, 1.0].
    """
    feats = np.full(NUM_TEXTURE_FEATURES, -1.0, dtype=np.float32)

    valid_board = [c for c in board_cards if c != -1]
    if len(hole_cards) != 2 or len(valid_board) < 3:
        # Preflop or invalid: return default baseline (-1.0)
        return feats

    h_ranks = [rank_of(c) for c in hole_cards]
    h_suits = [suit_of(c) for c in hole_cards]
    b_ranks = [rank_of(c) for c in valid_board]
    b_suits = [suit_of(c) for c in valid_board]

    all_ranks = h_ranks + b_ranks
    all_suits = h_suits + b_suits

    b_ranks_sorted = sorted(b_ranks)
    b_high_rank = b_ranks_sorted[-1]
    b_suit_counts = Counter(b_suits)
    b_rank_counts = Counter(b_ranks)

    # 1. Board High Rank [-1.0, 1.0]
    feats[0] = normalize_rank(b_high_rank)

    # 2. Is Monotone (3+ cards of same suit on board)
    max_b_suit = max(b_suit_counts.values()) if b_suit_counts else 0
    feats[1] = 1.0 if max_b_suit >= 3 else -1.0

    # 3. Is Two-Tone (exactly 2 of dominant suit on 3 or 4 card board)
    feats[2] = 1.0 if max_b_suit == 2 else -1.0

    # 4. Is Board Paired (or trips)
    max_b_rank_cnt = max(b_rank_counts.values()) if b_rank_counts else 0
    feats[3] = 1.0 if max_b_rank_cnt >= 2 else -1.0

    # 5. Board Connectedness: count rank gaps <= 2 between adjacent board cards
    connected_pairs = 0
    unique_b_ranks = sorted(set(b_ranks))
    for i in range(len(unique_b_ranks) - 1):
        gap = unique_b_ranks[i + 1] - unique_b_ranks[i]
        if gap == 1:
            connected_pairs += 2
        elif gap == 2:
            connected_pairs += 1
    # Normalized to [-1.0, 1.0] (max ~6)
    feats[4] = float(np.clip((connected_pairs / 3.0) - 1.0, -1.0, 1.0))

    # --- Hero Made Hand Relative Strength ---
    is_pocket_pair = (h_ranks[0] == h_ranks[1])
    h_high_rank = max(h_ranks)
    h_low_rank = min(h_ranks)

    # 6. Overpair (pocket pair higher than board high rank)
    has_overpair = is_pocket_pair and (h_high_rank > b_high_rank)
    feats[5] = 1.0 if has_overpair else -1.0

    # 7. Top Pair (Hero matches the highest board card)
    matches_high = (h_high_rank == b_high_rank) or (h_low_rank == b_high_rank)
    has_top_pair = (not is_pocket_pair) and matches_high
    feats[6] = 1.0 if has_top_pair else -1.0

    # 8. Mid or Bottom Pair (Hero matches 2nd, 3rd, or lower board card)
    matches_other = False
    for r in h_ranks:
        if r in b_ranks and r != b_high_rank:
            matches_other = True
            break
    has_mid_bottom_pair = (not is_pocket_pair) and matches_other and (not has_top_pair)
    feats[7] = 1.0 if has_mid_bottom_pair else -1.0

    # 9. Two Pair (Hero pairs both cards with board, or pocket pair + board pair, or board pairs + match)
    hero_paired_cards = sum(1 for r in set(h_ranks) if r in b_ranks)
    has_two_pair = (hero_paired_cards == 2) or (is_pocket_pair and max_b_rank_cnt >= 2 and not has_overpair)
    feats[8] = 1.0 if has_two_pair else -1.0

    # 10. Set or Trips (Set = pocket pair matches 1 board card; Trips = 1 hole card matches board pair)
    has_set = is_pocket_pair and (h_high_rank in b_ranks)
    has_trips = (not is_pocket_pair) and any(b_rank_counts[r] >= 2 for r in h_ranks)
    feats[9] = 1.0 if (has_set or has_trips) else -1.0

    # 11. Straight or Better (Check 5-card evaluator)
    from games.holdem.evaluator import evaluate_holdem_hand
    score, _ = evaluate_holdem_hand(hole_cards + valid_board)
    has_straight_plus = (score >= 4_000_000)
    feats[10] = 1.0 if has_straight_plus else -1.0

    # --- Hero Draws & Outs Engine ---
    outs = 0

    # 12. Flush Draw: 4 cards of the same suit among hero + board (and not already a flush)
    all_suit_counts = Counter(all_suits)
    flush_suit = None
    has_flush_draw = False
    for s, cnt in all_suit_counts.items():
        if cnt == 4 and score < 5_000_000:
            # Must use at least 1 hero card
            if s in h_suits:
                has_flush_draw = True
                flush_suit = s
                outs += 9
                break
    feats[11] = 1.0 if has_flush_draw else -1.0

    # 13. Nut Flush Draw: Hero holds Ace of the flush suit
    has_nut_flush_draw = False
    if has_flush_draw and flush_suit is not None:
        for c in hole_cards:
            if rank_of(c) == 12 and suit_of(c) == flush_suit:
                has_nut_flush_draw = True
                break
    feats[12] = 1.0 if has_nut_flush_draw else -1.0

    # 14. Straight Draws: Open-Ended Straight Draw (OESD) vs Gutshot
    # Check rank subsets for 4-to-a-straight
    unique_all_ranks = set(all_ranks)
    if 12 in unique_all_ranks:
        unique_all_ranks.add(-1) # Ace can be low (-1 for 5-high wheel)

    has_oesd = False
    has_gutshot = False

    if score < 4_000_000:
        straight_completing_ranks = 0
        for test_r in range(13):
            if test_r not in unique_all_ranks:
                test_set = unique_all_ranks | {test_r}
                # Check if any consecutive 5 exist
                sorted_test = sorted(test_set)
                for start_idx in range(len(sorted_test) - 4):
                    if sorted_test[start_idx + 4] - sorted_test[start_idx] == 4 and len(set(sorted_test[start_idx:start_idx+5])) == 5:
                        straight_completing_ranks += 1
                        break

        if straight_completing_ranks >= 2:
            has_oesd = True
            outs += 8
        elif straight_completing_ranks == 1:
            has_gutshot = True
            outs += 4

    feats[13] = 1.0 if has_oesd else -1.0
    feats[14] = 1.0 if has_gutshot else -1.0

    # 15. Overcards: Hole cards higher than any board card
    num_overcards = 0
    if not is_pocket_pair and score < 1_000_000: # high card only
        for r in h_ranks:
            if r > b_high_rank:
                num_overcards += 1
                outs += 3 # standard 3 outs per overcard to hit top pair
    feats[15] = float((num_overcards / 1.0) - 1.0) # 0 -> -1.0, 1 -> 0.0, 2 -> 1.0

    # 16. Rule of 2 and 4 Equity Estimate:
    # Flop: ~outs * 4%, Turn: ~outs * 2%
    is_flop = (len(valid_board) == 3)
    mult = 4.0 if is_flop else 2.0
    approx_equity = min(1.0, (outs * mult) / 100.0)
    feats[16] = float((approx_equity * 2.0) - 1.0)

    return feats
