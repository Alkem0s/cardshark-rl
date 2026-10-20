"""
games/five_card_draw/side_pot.py — Multi-Way Side Pot and All-In Resolution for Poker.

Handles arbitrary player counts, differing stack depths, all-in thresholds,
folds, and split pots (ties) for main and side pots.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple


class PotTier:
    """Represents a single pot (main pot or side pot) and who is eligible to win it."""
    def __init__(self, amount: int, eligible: Set[int]):
        self.amount = amount
        self.eligible = set(eligible)

    def __repr__(self):
        return f"PotTier(amount={self.amount}, eligible={sorted(list(self.eligible))})"


def calculate_side_pots(
    investments: Dict[int, int],
    folded_players: Set[int],
) -> List[PotTier]:
    """Calculate main pot and side pots from player investments."""
    all_contributors = {p: amt for p, amt in investments.items() if amt > 0}
    if not all_contributors:
        return []

    sorted_levels = sorted(list(set(all_contributors.values())))
    raw_tiers: List[PotTier] = []
    prev_level = 0

    for level in sorted_levels:
        diff = level - prev_level
        if diff <= 0:
            continue

        contributors_at_level = [p for p, amt in all_contributors.items() if amt >= level]
        tier_amount = diff * len(contributors_at_level)
        eligible_at_level = {p for p in contributors_at_level if p not in folded_players}

        if tier_amount > 0 and len(eligible_at_level) > 0:
            raw_tiers.append(PotTier(tier_amount, eligible_at_level))

        prev_level = level

    if not raw_tiers:
        return []

    merged_tiers: List[PotTier] = []
    for tier in raw_tiers:
        if merged_tiers and merged_tiers[-1].eligible == tier.eligible:
            merged_tiers[-1].amount += tier.amount
        else:
            merged_tiers.append(tier)

    return merged_tiers


def resolve_showdown_payouts(
    pots: List[PotTier],
    hand_scores: Dict[int, int],
    seat_order_from_button: List[int] | None = None,
) -> Tuple[Dict[int, int], List[dict]]:
    """Distribute pot chips to winners based on hand evaluation scores."""
    payouts: Dict[int, int] = {p: 0 for p in hand_scores.keys()}
    details = []

    for idx, pot in enumerate(pots):
        eligible = pot.eligible
        if not eligible:
            continue

        eligible_scores = {p: hand_scores[p] for p in eligible if p in hand_scores}
        if not eligible_scores:
            continue

        best_score = max(eligible_scores.values())
        winners = [p for p, score in eligible_scores.items() if score == best_score]

        split_amount = pot.amount // len(winners)
        remainder = pot.amount % len(winners)

        pot_winners = {}
        for w in winners:
            payouts[w] = payouts.get(w, 0) + split_amount
            pot_winners[w] = split_amount

        if remainder > 0:
            if seat_order_from_button:
                ordered_winners = [s for s in seat_order_from_button if s in winners]
            else:
                ordered_winners = sorted(winners)

            for i in range(remainder):
                recipient = ordered_winners[i % len(ordered_winners)]
                payouts[recipient] += 1
                pot_winners[recipient] += 1

        details.append({
            "pot_tier": idx,
            "pot_amount": pot.amount,
            "winners": pot_winners,
            "best_score": best_score,
        })

    return payouts, details
