"""
games/holdem/icm_engine.py — Malmuth-Harville Independent Chip Model (ICM) Tournament Equity Engine.

Provides:
1. Exact recursive Malmuth-Harville ICM probability calculations for up to 5-player tournaments.
2. Conversion of tournament chip distribution into non-linear dollar equities.
3. Bubble factor calculation (risk/reward ratio of risking chips).
4. Observation features extraction for Model C.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np


# Standard tournament payouts for 5-player Sit-and-Go: 65% for 1st, 35% for 2nd
DEFAULT_PAYOUTS_5P = [0.65, 0.35]


def calculate_icm_equities(
    chips: List[int],
    payouts: Optional[List[float]] = None,
) -> List[float]:
    """
    Computes Malmuth-Harville ICM tournament equities for each seat given chip counts.

    Returns:
        equities: List of equities summing to 1.0 (or total payouts sum).
    """
    if payouts is None:
        payouts = DEFAULT_PAYOUTS_5P

    n = len(chips)
    equities = [0.0] * n
    total_chips = sum(chips)

    if total_chips <= 0:
        return [1.0 / n] * n

    # Filter out busted seats (0 chips)
    active_indices = [i for i in range(n) if chips[i] > 0]
    if len(active_indices) == 0:
        return [1.0 / n] * n
    if len(active_indices) == 1:
        winner = active_indices[0]
        res = [0.0] * n
        res[winner] = sum(payouts)
        return res

    # 1st Place Probabilities: P(1st = i) = chips[i] / total_chips
    p_1st = [chips[i] / float(total_chips) for i in range(n)]

    # If only 1 payout tier (winner take all):
    if len(payouts) == 1:
        for i in range(n):
            equities[i] = p_1st[i] * payouts[0]
        return equities

    # Add 1st place payout equity
    for i in range(n):
        equities[i] += p_1st[i] * payouts[0]

    # 2nd Place Probabilities: P(2nd = j) = sum_{i != j} [ P(1st = i) * chips[j] / (total - chips[i]) ]
    if len(payouts) >= 2 and len(active_indices) >= 2:
        p2_amt = payouts[1]
        for j in active_indices:
            p_2nd_j = 0.0
            c_j = chips[j]
            for i in active_indices:
                if i != j:
                    rem_chips = total_chips - chips[i]
                    if rem_chips > 0:
                        p_2nd_j += p_1st[i] * (c_j / float(rem_chips))
            equities[j] += p_2nd_j * p2_amt

    # 3rd Place Probabilities (if 3 payouts specified)
    if len(payouts) >= 3 and len(active_indices) >= 3:
        p3_amt = payouts[2]
        for k in active_indices:
            p_3rd_k = 0.0
            c_k = chips[k]
            for i in active_indices:
                if i == k:
                    continue
                rem_i = total_chips - chips[i]
                if rem_i <= 0:
                    continue
                for j in active_indices:
                    if j == i or j == k:
                        continue
                    rem_ij = rem_i - chips[j]
                    if rem_ij > 0:
                        p_3rd_k += p_1st[i] * (chips[j] / float(rem_i)) * (c_k / float(rem_ij))
            equities[k] += p_3rd_k * p3_amt

    return equities


def compute_icm_features(
    hero_seat: int,
    seats_chips: List[int],
    bet_to_call: int = 0,
    pot: int = 0,
    payouts: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Extracts 3 normalized ICM metrics for Model C:
    1. Hero ICM Equity [-1.0, 1.0] (0.0 = average equity, 1.0 = tournament lock)
    2. Bubble Factor / Risk Asymmetry [-1.0, 1.0]
    3. Elimination Risk on Current Bet [-1.0, 1.0]
    """
    feats = np.zeros(3, dtype=np.float32)
    equities = calculate_icm_equities(seats_chips, payouts=payouts)
    hero_eq = equities[hero_seat] if hero_seat < len(equities) else 0.0

    # 1. Equity normalized [-1.0, 1.0] (baseline 0.20 for 5-player Sit-and-Go)
    feats[0] = np.clip((hero_eq - 0.20) * 2.5, -1.0, 1.0)

    # 2. Bubble Factor: Ratio of equity lost on fold/loss vs equity gained on win
    hero_chips = seats_chips[hero_seat] if hero_seat < len(seats_chips) else 0
    if hero_chips <= 0:
        feats[1] = 1.0   # Maximum risk/bustout
        feats[2] = 1.0
        return feats

    # Estimate equity if Hero wins pot vs if Hero loses bet_to_call
    if bet_to_call > 0 and pot > 0:
        win_chips = list(seats_chips)
        win_chips[hero_seat] += pot
        win_eq = calculate_icm_equities(win_chips, payouts=payouts)[hero_seat]

        lose_chips = list(seats_chips)
        lose_chips[hero_seat] = max(0, lose_chips[hero_seat] - bet_to_call)
        lose_eq = calculate_icm_equities(lose_chips, payouts=payouts)[hero_seat]

        gain = max(1e-5, win_eq - hero_eq)
        loss = max(1e-5, hero_eq - lose_eq)
        bubble_factor = loss / gain  # Standard bubble factor >= 1.0
        # Normalize: BF = 1.0 (neutral) -> 0.0, BF >= 2.0 -> 1.0
        feats[1] = np.clip((bubble_factor - 1.0), -1.0, 1.0)

        # 3. Elimination risk: bet_to_call relative to Hero chips
        elim_risk = min(1.0, bet_to_call / float(hero_chips))
        feats[2] = (elim_risk * 2.0) - 1.0
    else:
        feats[1] = -1.0
        feats[2] = -1.0

    return feats
