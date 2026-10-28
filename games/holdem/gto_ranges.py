"""
games/holdem/gto_ranges.py — Canonical 169-Hand Preflop GTO Matrix & Positional Profiler.

Provides:
1. Canonical starting hand notation (AA, AKs, AKo, 72o) mapping 1,326 raw combos to 169 canonical hands.
2. GTO Starting Hand Tier classification (Monster, Strong, Playable, Speculative, Marginal Steal, Trash).
3. Position-conditioned opening and 3-bet defense thresholds (UTG, MP, CO, BTN, SB, BB).
4. Feature extraction for Model C observation space (6 normalized features).
"""

from __future__ import annotations
from typing import Tuple, List, Dict
import numpy as np

from core.card import rank_of, suit_of

# Canonical Hand Tiers:
TIER_MONSTER = 1      # AA, KK, QQ, AKs (top ~2.5%)
TIER_STRONG = 2       # JJ-TT, AQs-AJs, KQs, AKo, AQo (top ~6%)
TIER_PLAYABLE = 3     # 99-77, ATs-A2s, KJs-KTs, QJs-QTs, JTs, AJo-ATo, KJo (top ~16%)
TIER_SPECULATIVE = 4  # 66-22, K9s-K6s, Q9s-Q8s, J9s-J8s, T9s-T8s, 98s, 87s, 76s, KTo, QJo, JTo (top ~30%)
TIER_MARGINAL = 5     # Late position steals: suited gappers, offsuit connectors (top ~42%)
TIER_TRASH = 6        # Unplayable negative-EV holdings (bottom ~58%)

RANK_CHARS = "23456789TJQKA"


def get_canonical_hand(c1: int, c2: int) -> str:
    """Returns canonical string representation of starting hand (e.g. 'AKs', 'AKo', 'TT')."""
    if c1 < 0 or c2 < 0:
        return "Unknown"
    r1, r2 = rank_of(c1), rank_of(c2)
    s1, s2 = suit_of(c1), suit_of(c2)

    high_r = max(r1, r2)
    low_r = min(r1, r2)
    c_high = RANK_CHARS[high_r]
    c_low = RANK_CHARS[low_r]

    if high_r == low_r:
        return f"{c_high}{c_low}"
    elif s1 == s2:
        return f"{c_high}{c_low}s"
    else:
        return f"{c_high}{c_low}o"


# Pre-computed GTO Matrix of 169 Canonical Hands
_GTO_TIERS: Dict[str, int] = {
    # Tier 1: Monsters
    "AA": TIER_MONSTER, "KK": TIER_MONSTER, "QQ": TIER_MONSTER, "AKs": TIER_MONSTER,

    # Tier 2: Strong
    "JJ": TIER_STRONG, "TT": TIER_STRONG, "AQs": TIER_STRONG, "AJs": TIER_STRONG,
    "KQs": TIER_STRONG, "AKo": TIER_STRONG, "AQo": TIER_STRONG,

    # Tier 3: Playable
    "99": TIER_PLAYABLE, "88": TIER_PLAYABLE, "77": TIER_PLAYABLE,
    "ATs": TIER_PLAYABLE, "A9s": TIER_PLAYABLE, "A8s": TIER_PLAYABLE, "A7s": TIER_PLAYABLE,
    "A6s": TIER_PLAYABLE, "A5s": TIER_PLAYABLE, "A4s": TIER_PLAYABLE, "A3s": TIER_PLAYABLE,
    "A2s": TIER_PLAYABLE, "KJs": TIER_PLAYABLE, "KTs": TIER_PLAYABLE, "QJs": TIER_PLAYABLE,
    "QTs": TIER_PLAYABLE, "JTs": TIER_PLAYABLE, "AJo": TIER_PLAYABLE, "ATo": TIER_PLAYABLE,
    "KQo": TIER_PLAYABLE, "KJo": TIER_PLAYABLE,

    # Tier 4: Speculative
    "66": TIER_SPECULATIVE, "55": TIER_SPECULATIVE, "44": TIER_SPECULATIVE,
    "33": TIER_SPECULATIVE, "22": TIER_SPECULATIVE, "K9s": TIER_SPECULATIVE,
    "K8s": TIER_SPECULATIVE, "K7s": TIER_SPECULATIVE, "Q9s": TIER_SPECULATIVE,
    "Q8s": TIER_SPECULATIVE, "J9s": TIER_SPECULATIVE, "J8s": TIER_SPECULATIVE,
    "T9s": TIER_SPECULATIVE, "T8s": TIER_SPECULATIVE, "98s": TIER_SPECULATIVE,
    "97s": TIER_SPECULATIVE, "87s": TIER_SPECULATIVE, "76s": TIER_SPECULATIVE,
    "65s": TIER_SPECULATIVE, "54s": TIER_SPECULATIVE, "KTo": TIER_SPECULATIVE,
    "QJo": TIER_SPECULATIVE, "QTo": TIER_SPECULATIVE, "JTo": TIER_SPECULATIVE,

    # Tier 5: Marginal Steal (Button / Small Blind unraised)
    "K6s": TIER_MARGINAL, "K5s": TIER_MARGINAL, "K4s": TIER_MARGINAL, "Q7s": TIER_MARGINAL,
    "Q6s": TIER_MARGINAL, "J7s": TIER_MARGINAL, "T7s": TIER_MARGINAL, "86s": TIER_MARGINAL,
    "75s": TIER_MARGINAL, "64s": TIER_MARGINAL, "53s": TIER_MARGINAL, "A9o": TIER_MARGINAL,
    "A8o": TIER_MARGINAL, "K9o": TIER_MARGINAL, "Q9o": TIER_MARGINAL, "J9o": TIER_MARGINAL,
    "T9o": TIER_MARGINAL, "98o": TIER_MARGINAL,
}


def get_hand_gto_tier(c1: int, c2: int) -> int:
    """Returns the GTO tier (1..6) for the hole cards."""
    canon = get_canonical_hand(c1, c2)
    return _GTO_TIERS.get(canon, TIER_TRASH)


def is_gto_preflop_permitted(
    c1: int,
    c2: int,
    pos_offset: int,
    bet_to_call: int,
    big_blind: int,
    num_callers: int = 0,
    is_short_stack: bool = False,
) -> Tuple[bool, bool]:
    """
    Evaluates whether Hero holding (c1, c2) is permitted to open-raise or call preflop.

    pos_offset: (hero_seat - dealer_seat) % num_seats:
        0 = Button (BTN)
        1 = Small Blind (SB)
        2 = Big Blind (BB)
        3 = Under the Gun (UTG) in 5-handed
        4 = Cutoff (CO) in 5-handed

    Returns:
        (can_call, can_raise)
    """
    if is_short_stack:
        # Push-or-fold tournament dynamics: short stacks (<20 BB) have wider shove ranges
        tier = get_hand_gto_tier(c1, c2)
        if tier <= TIER_SPECULATIVE:
            return True, True
        return (bet_to_call == 0), False

    tier = get_hand_gto_tier(c1, c2)

    # 1. Facing Heavy Action (3-Bet or higher, >= 3x BB)
    if bet_to_call >= big_blind * 3:
        if tier <= TIER_MONSTER:
            return True, True   # 4-bet or call with monsters
        elif tier <= TIER_STRONG:
            return True, False  # Defend strong hands with call, no 4-bet
        else:
            return False, False # Must fold speculative / trash facing 3-bet

    # 2. Facing a Standard Open Raise (bet_to_call > big_blind and <= 3x BB)
    if bet_to_call > big_blind and bet_to_call <= big_blind * 3:
        if pos_offset == 2:  # Big Blind defending with discount
            if tier <= TIER_SPECULATIVE:
                return True, (tier <= TIER_STRONG)
            return False, False
        else:
            # Cold-calling in position
            if tier <= TIER_STRONG:
                return True, True
            elif tier <= TIER_PLAYABLE and num_callers >= 1:
                # Multiway pot: speculative suited hands have implied odds to call
                return True, False
            elif tier <= TIER_PLAYABLE and pos_offset in (0, 4):  # BTN, CO
                return True, False
            else:
                return False, False

    # 3. Unopened Pot (bet_to_call == 0 or Hero in BB checked to)
    if bet_to_call == 0:
        return True, True

    # 4. Unraised Pot facing 1 BB to call (e.g. SB limp or limper before Hero)
    if bet_to_call <= big_blind:
        # Positional Open-Raise Thresholds:
        if pos_offset == 3:  # UTG: tightest range (Tiers 1..3 only, top 16%)
            can_raise = (tier <= TIER_PLAYABLE)
            can_call = (tier <= TIER_PLAYABLE)
        elif pos_offset == 4:  # Cutoff: Tiers 1..4 (top ~30%)
            can_raise = (tier <= TIER_SPECULATIVE)
            can_call = (tier <= TIER_SPECULATIVE)
        elif pos_offset == 0:  # Button: steal position (Tiers 1..5, top ~42%)
            can_raise = (tier <= TIER_MARGINAL)
            can_call = (tier <= TIER_MARGINAL)
        elif pos_offset == 1:  # Small Blind: Tiers 1..4
            can_raise = (tier <= TIER_SPECULATIVE)
            can_call = (tier <= TIER_MARGINAL)
        else:  # Big Blind facing limper
            can_raise = (tier <= TIER_STRONG)
            can_call = True  # Can always check for free
        return can_call, can_raise

    return False, False


def extract_preflop_gto_features(
    c1: int,
    c2: int,
    pos_offset: int,
    num_callers: int,
    bet_to_call: int,
    big_blind: int,
) -> np.ndarray:
    """
    Extracts 6 normalized features for Model C observation vector:
    1. GTO Tier normalized [-1.0, 1.0]
    2. Positional threshold delta [-1.0, 1.0] (positive = value open, negative = fold)
    3. Suitedness flag [-1.0 or 1.0]
    4. Steal opportunity flag [-1.0 or 1.0] (Button / Cutoff unopened)
    5. Multiway caller discount [-1.0, 1.0]
    6. 3-Bet defense equity [-1.0, 1.0]
    """
    feats = np.zeros(6, dtype=np.float32)
    if c1 < 0 or c2 < 0:
        return np.full(6, -1.0, dtype=np.float32)

    tier = get_hand_gto_tier(c1, c2)
    r1, r2 = rank_of(c1), rank_of(c2)
    s1, s2 = suit_of(c1), suit_of(c2)

    # 1. Tier normalized: Tier 1 -> 1.0, Tier 6 -> -1.0
    feats[0] = ((6 - tier) / 2.5) - 1.0

    # 2. Positional threshold delta:
    # UTG threshold = 3, CO = 4, BTN = 5, SB = 4, BB = 5
    pos_thresholds = {3: 3, 4: 4, 0: 5, 1: 4, 2: 5}
    thresh = pos_thresholds.get(pos_offset, 4)
    delta = thresh - tier  # >0 = inside opening range, <0 = outside
    feats[1] = np.clip(delta / 3.0, -1.0, 1.0)

    # 3. Suitedness
    feats[2] = 1.0 if s1 == s2 else -1.0

    # 4. Steal opportunity: in BTN (0) or CO (4) with 0 callers
    is_steal = (pos_offset in (0, 4) and num_callers == 0 and bet_to_call <= big_blind)
    feats[3] = 1.0 if is_steal else -1.0

    # 5. Multiway discount: more callers reduces raw high-card value
    caller_ratio = min(1.0, num_callers / 4.0)
    feats[4] = (caller_ratio * 2.0) - 1.0

    # 6. 3-bet defense equity: High tier hands defend strongly
    defense = 1.0 if tier <= TIER_STRONG else (0.5 if tier <= TIER_PLAYABLE else -1.0)
    feats[5] = defense

    return feats
