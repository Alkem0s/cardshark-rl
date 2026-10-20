"""
core/math_utils.py — Game-Theoretic & Statistical Evaluation Utilities.
"""

from __future__ import annotations
import math
from typing import Tuple


def pot_odds(bet_to_call: float, pot: float) -> float:
    """Computes exact pot odds: bet_to_call / (pot + bet_to_call).

    Returns 0.0 if there is no bet to call.
    """
    if bet_to_call <= 0:
        return 0.0
    denom = pot + bet_to_call
    if denom <= 0:
        return 0.0
    return float(bet_to_call / denom)


def minimum_defense_frequency(bet: float, pot: float) -> float:
    """Computes Minimum Defense Frequency (MDF): 1 - (bet / (pot + bet)).

    Represents the fraction of range hero must defend to make villain's bluff break-even.
    Returns 1.0 if facing no bet.
    """
    if bet <= 0:
        return 1.0
    denom = pot + bet
    if denom <= 0:
        return 1.0
    alpha = bet / denom
    return float(max(0.0, min(1.0, 1.0 - alpha)))


def calculate_proportions_z_test(k1: int, n1: int, k2: int, n2: int) -> Tuple[float, float]:
    """Two-proportion z-test for difference in win rates. Returns (z_score, p_value)."""
    if n1 <= 0 or n2 <= 0:
        return 0.0, 1.0

    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)

    if p_pool == 0 or p_pool == 1:
        return 0.0, 1.0

    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    if se <= 1e-12:
        return 0.0, 1.0

    z = (p1 - p2) / se
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return float(z), float(p_value)


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes Wilson score 95% confidence interval for a proportion."""
    if n <= 0:
        return 0.0, 0.0
    z = 1.96
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) + (z**2) / (4.0 * n)) / n)) / denom
    lower = max(0.0, center - margin) * 100.0
    upper = min(1.0, center + margin) * 100.0
    return round(lower, 1), round(upper, 1)


def calculate_bb_per_100(net_chips: float, big_blind: float, total_hands: int) -> float:
    """Computes standard BB/100 win rate."""
    if total_hands <= 0 or big_blind <= 0:
        return 0.0
    return float((net_chips / big_blind) / (total_hands / 100.0))


# Re-export side-pot calculation utilities
try:
    from games.five_card_draw.side_pot import calculate_side_pots, resolve_showdown_payouts, PotTier
except ImportError:
    try:
        from game.side_pot import calculate_side_pots, resolve_showdown_payouts, PotTier
    except ImportError:
        pass
