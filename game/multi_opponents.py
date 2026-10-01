"""
multi_opponents.py — Parameterized Bot Archetypes for Multi-Player 5-Card Draw Poker.

Supports variable bet sizing (Fold, Call, Min-Raise, Half-Pot, Pot, All-In) and
provides 5 distinct behavioral archetypes:
1. CallingStation (Loose-Passive)
2. Maniac (Loose-Aggressive / Hyper)
3. Rock (Tight-Passive)
4. TAG (Tight-Aggressive)
5. LAG (Loose-Aggressive)
"""

from __future__ import annotations
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

try:
    from game.card_utils import (
        rank_of, suit_of, hand_category, get_pairs_info,
        has_flush_draw, has_straight_draw, evaluate_hand
    )
except ImportError:
    from card_utils import (
        rank_of, suit_of, hand_category, get_pairs_info,
        has_flush_draw, has_straight_draw, evaluate_hand
    )

# Action indices
A_FOLD = 0
A_CALL = 1
A_MIN_RAISE = 2
A_HALF_POT = 3
A_POT = 4
A_ALL_IN = 5
A_DRAW_START = 6


class MultiPlayerOpponent(ABC):
    """Base class for multi-player bot policies with variable bet sizing."""

    def __init__(self, opponent_id: int, name: str, rng: np.random.Generator | None = None):
        self.opponent_id = opponent_id
        self.name = name
        self.rng = rng or np.random.default_rng()

    @abstractmethod
    def bet_action(
        self,
        hand: List[int],
        pot: int,
        bet_to_call: int,
        phase: str,
        stack: int,
        legal_actions: List[int],
    ) -> int:
        """Choose betting action among legal_actions (0 to 5)."""
        ...

    @abstractmethod
    def draw_action(self, hand: List[int]) -> List[int]:
        """Return list of card indices (0-4) to discard."""
        ...


# ---------------------------------------------------------------------------
# Mathematical Discard Helper
# ---------------------------------------------------------------------------

def standard_math_discard(hand: List[int]) -> List[int]:
    """Mathematically sound draw strategy: keep best group, discard kickers."""
    info = get_pairs_info(hand)
    cat = info["category"]

    if cat >= 4:  # Straight, Flush, Full House, Quads, Straight Flush -> Stand Pat
        return []
    if cat == 3:  # Trips -> discard 2 kickers
        return info["kicker_indices"]
    if cat == 2:  # Two pair -> discard 1 kicker
        return info["kicker_indices"]
    if cat == 1:  # One pair -> discard 3 kickers
        return info["kicker_indices"]

    # High card: check 4-card flush or 4-card straight draw
    has_flush, flush_suit = has_flush_draw(hand)
    if has_flush and flush_suit is not None:
        return [i for i in range(5) if suit_of(hand[i]) != flush_suit]

    has_straight, keep_indices = has_straight_draw(hand)
    if has_straight and keep_indices:
        return [i for i in range(5) if i not in keep_indices]

    # Keep two highest cards, discard 3
    ranks_with_idx = [(rank_of(hand[i]), i) for i in range(5)]
    ranks_with_idx.sort(reverse=True)
    keep = {ranks_with_idx[0][1], ranks_with_idx[1][1]}
    return [i for i in range(5) if i not in keep]


# ---------------------------------------------------------------------------
# 1. Calling Station (Loose-Passive)
# ---------------------------------------------------------------------------

class CallingStation(MultiPlayerOpponent):
    """Enters almost all pots, calls any bet, never folds unless broke, rarely raises."""

    def __init__(self, rng=None):
        super().__init__(opponent_id=0, name="CallingStation", rng=rng)

    def bet_action(self, hand, pot, bet_to_call, phase, stack, legal_actions) -> int:
        # 95% Call/Check, 5% Min-Raise if legal
        if A_CALL in legal_actions:
            if A_MIN_RAISE in legal_actions and self.rng.random() < 0.05:
                return A_MIN_RAISE
            return A_CALL
        return legal_actions[0] if legal_actions else A_CALL

    def draw_action(self, hand: List[int]) -> List[int]:
        return standard_math_discard(hand)


# ---------------------------------------------------------------------------
# 2. Maniac (Loose-Aggressive / Wild)
# ---------------------------------------------------------------------------

class Maniac(MultiPlayerOpponent):
    """High VPIP, bets and raises with aggressive pot/all-in sizes, folds rarely."""

    def __init__(self, rng=None):
        super().__init__(opponent_id=1, name="Maniac", rng=rng)

    def bet_action(self, hand, pot, bet_to_call, phase, stack, legal_actions) -> int:
        # Folds only 5% of the time to extreme pressure
        if bet_to_call > stack * 0.5 and self.rng.random() < 0.08 and A_FOLD in legal_actions:
            return A_FOLD

        # 60% of the time, attempts an aggressive raise (Pot or Half-Pot or All-in)
        r = self.rng.random()
        if r < 0.25 and A_POT in legal_actions:
            return A_POT
        elif r < 0.45 and A_HALF_POT in legal_actions:
            return A_HALF_POT
        elif r < 0.60 and A_MIN_RAISE in legal_actions:
            return A_MIN_RAISE
        elif r < 0.65 and A_ALL_IN in legal_actions:
            return A_ALL_IN

        if A_CALL in legal_actions:
            return A_CALL
        return legal_actions[0]

    def draw_action(self, hand: List[int]) -> List[int]:
        # Chases draws aggressively: draws 1 card 60% of the time on High Card
        cat = hand_category(hand)
        if cat == 0 and self.rng.random() < 0.60:
            return [self.rng.integers(0, 5)]
        return standard_math_discard(hand)


# ---------------------------------------------------------------------------
# 3. Rock (Tight-Passive)
# ---------------------------------------------------------------------------

class Rock(MultiPlayerOpponent):
    """Extremely tight range. Folds unpaired hands pre-draw to bets, raises only nuts."""

    def __init__(self, rng=None):
        super().__init__(opponent_id=2, name="Rock", rng=rng)

    def bet_action(self, hand, pot, bet_to_call, phase, stack, legal_actions) -> int:
        cat = hand_category(hand)
        info = get_pairs_info(hand)

        # Checking is free
        if bet_to_call == 0:
            if cat >= 2 and A_MIN_RAISE in legal_actions and self.rng.random() < 0.40:
                return A_MIN_RAISE
            return A_CALL

        # Facing a bet
        if phase == "pre_draw":
            # Fold unpaired hands
            if cat == 0 and A_FOLD in legal_actions:
                return A_FOLD
            # Call with weak pair
            if cat == 1 and info["pair_ranks"] and info["pair_ranks"][0] < 9: # Under Jacks
                if bet_to_call > stack * 0.25 and A_FOLD in legal_actions:
                    return A_FOLD
                return A_CALL
            # Raise with premium pair or better
            if cat >= 2 or (cat == 1 and info["pair_ranks"] and info["pair_ranks"][0] >= 9):
                if A_MIN_RAISE in legal_actions and self.rng.random() < 0.50:
                    return A_MIN_RAISE
                return A_CALL
        else: # post-draw
            if cat <= 1 and bet_to_call > stack * 0.15 and A_FOLD in legal_actions:
                return A_FOLD
            if cat >= 3 and A_HALF_POT in legal_actions:
                return A_HALF_POT

        if A_CALL in legal_actions:
            return A_CALL
        return legal_actions[0]

    def draw_action(self, hand: List[int]) -> List[int]:
        return standard_math_discard(hand)


# ---------------------------------------------------------------------------
# 4. TAG (Tight-Aggressive)
# ---------------------------------------------------------------------------

class TAG(MultiPlayerOpponent):
    """Selective starting hands, aggressive bet sizing (Half-Pot / Pot) when entering."""

    def __init__(self, rng=None):
        super().__init__(opponent_id=3, name="TAG", rng=rng)

    def bet_action(self, hand, pot, bet_to_call, phase, stack, legal_actions) -> int:
        cat = hand_category(hand)
        info = get_pairs_info(hand)

        # Pre-draw
        if phase == "pre_draw":
            if bet_to_call == 0:
                # Open raise with pair of 8s or better or high cards
                if cat >= 2 or (cat == 1 and info["pair_ranks"] and info["pair_ranks"][0] >= 6):
                    if A_HALF_POT in legal_actions:
                        return A_HALF_POT
                    elif A_MIN_RAISE in legal_actions:
                        return A_MIN_RAISE
                return A_CALL
            else:
                # Facing a raise
                if cat == 0 and A_FOLD in legal_actions:
                    return A_FOLD
                if cat == 1 and info["pair_ranks"] and info["pair_ranks"][0] < 8 and A_FOLD in legal_actions:
                    return A_FOLD
                # Re-raise with strong hand
                if cat >= 2 and A_POT in legal_actions and self.rng.random() < 0.40:
                    return A_POT
                return A_CALL if A_CALL in legal_actions else legal_actions[0]

        # Post-draw
        if cat >= 2: # Two pair or better -> value bet
            if A_HALF_POT in legal_actions and self.rng.random() < 0.70:
                return A_HALF_POT
            if A_MIN_RAISE in legal_actions:
                return A_MIN_RAISE
        elif cat == 1 and bet_to_call == 0:
            return A_CALL
        elif cat <= 1 and bet_to_call > stack * 0.20 and A_FOLD in legal_actions:
            return A_FOLD

        return A_CALL if A_CALL in legal_actions else legal_actions[0]

    def draw_action(self, hand: List[int]) -> List[int]:
        return standard_math_discard(hand)


# ---------------------------------------------------------------------------
# 5. LAG (Loose-Aggressive)
# ---------------------------------------------------------------------------

class LAG(MultiPlayerOpponent):
    """Wide range, attacks passive checks with Pot-sized bets, bluffs post-draw."""

    def __init__(self, rng=None):
        super().__init__(opponent_id=4, name="LAG", rng=rng)

    def bet_action(self, hand, pot, bet_to_call, phase, stack, legal_actions) -> int:
        cat = hand_category(hand)

        if bet_to_call == 0:
            # 55% chance to bet pot or half pot
            if self.rng.random() < 0.35 and A_HALF_POT in legal_actions:
                return A_HALF_POT
            if self.rng.random() < 0.20 and A_POT in legal_actions:
                return A_POT
            return A_CALL

        # Facing bet
        if cat == 0:
            # 20% bluff raise, 80% fold/call
            if self.rng.random() < 0.20 and A_POT in legal_actions:
                return A_POT
            if bet_to_call > stack * 0.15 and A_FOLD in legal_actions:
                return A_FOLD
            return A_CALL if A_CALL in legal_actions else legal_actions[0]

        # With pairs or better
        if cat >= 1:
            if self.rng.random() < 0.40 and A_HALF_POT in legal_actions:
                return A_HALF_POT
            return A_CALL

        return A_CALL if A_CALL in legal_actions else legal_actions[0]

    def draw_action(self, hand: List[int]) -> List[int]:
        return standard_math_discard(hand)


# ---------------------------------------------------------------------------
# 6. Adversarial Exploiter (Probes and attacks table leaks)
# ---------------------------------------------------------------------------

class AdversarialExploiter(MultiPlayerOpponent):
    """
    Dynamic exploitative sparring agent designed for Model D co-evolution.
    Probes for timid checks and passive calling tendencies:
    - Pre-draw: Puts maximum pressure on limpers and probes position.
    - Post-draw: Fires pot/all-in bets when opponents check or show weak draws.
    - Defends against over-bluffing by inducing bluffs with strong made hands (slow-playing trips+).
    """

    def __init__(self, rng=None):
        super().__init__(opponent_id=5, name="Exploiter", rng=rng)

    def bet_action(self, hand, pot, bet_to_call, phase, stack, legal_actions) -> int:
        cat = hand_category(hand)

        if bet_to_call == 0:
            if cat >= 3:
                # Slow play monster hands 40% to induce bluffs, bet 60%
                if self.rng.random() < 0.40:
                    return A_CALL
                if A_POT in legal_actions and self.rng.random() < 0.70:
                    return A_POT
                return A_HALF_POT if A_HALF_POT in legal_actions else A_CALL
            elif cat >= 1:
                if self.rng.random() < 0.65 and A_HALF_POT in legal_actions:
                    return A_HALF_POT
                return A_CALL
            else:
                # Probe bet bluff 35% of the time to punish passive checks
                if self.rng.random() < 0.35:
                    if A_POT in legal_actions:
                        return A_POT
                    if A_HALF_POT in legal_actions:
                        return A_HALF_POT
                return A_CALL

        # Facing a bet
        if cat >= 3:
            if A_ALL_IN in legal_actions and self.rng.random() < 0.40:
                return A_ALL_IN
            if A_POT in legal_actions:
                return A_POT
            if A_MIN_RAISE in legal_actions:
                return A_MIN_RAISE
            return A_CALL

        if cat in (1, 2):
            if bet_to_call > stack * 0.50 and cat == 1 and A_FOLD in legal_actions:
                return A_FOLD
            if self.rng.random() < 0.25 and A_MIN_RAISE in legal_actions:
                return A_MIN_RAISE
            return A_CALL if A_CALL in legal_actions else legal_actions[0]

        # Weak / High card facing bet
        if self.rng.random() < 0.15 and A_POT in legal_actions:
            return A_POT
        if A_FOLD in legal_actions:
            return A_FOLD
        return legal_actions[0]

    def draw_action(self, hand: List[int]) -> List[int]:
        return standard_math_discard(hand)


class RandomCyclingBot(MultiPlayerOpponent):
    """Heuristic bot that randomly cycles between different archetype behaviors."""
    def __init__(self, rng: np.random.Generator | None = None, rng_seed: int | None = None, cycle_frequency: int = 1):
        if rng is None and rng_seed is not None:
            r = np.random.default_rng(rng_seed)
        else:
            r = rng or np.random.default_rng()
        super().__init__(opponent_id=99, name="Chameleon", rng=r)
        self.cycle_frequency = cycle_frequency  # Cycle every N hands
        self.hands_in_current_style = 0
        self.archetypes = [cls(rng=self.rng) for cls in ARCHETYPE_CLASSES]
        idx = int(self.rng.integers(0, len(self.archetypes)))
        self.active_bot = self.archetypes[idx]
        self.name = f"Chameleon ({self.active_bot.name})"

    def get_style_name(self) -> str:
        return self.active_bot.name

    def rotate_style(self):
        idx = int(self.rng.integers(0, len(self.archetypes)))
        self.active_bot = self.archetypes[idx]
        self.name = f"Chameleon ({self.active_bot.name})"
        self.hands_in_current_style = 0

    def on_hand_end(self):
        self.hands_in_current_style += 1
        if self.hands_in_current_style >= self.cycle_frequency:
            self.rotate_style()

    def bet_action(
        self,
        hand: List[int],
        pot: int,
        bet_to_call: int,
        phase: str,
        stack: int,
        legal_actions: List[int],
    ) -> int:
        return self.active_bot.bet_action(hand, pot, bet_to_call, phase, stack, legal_actions)

    def draw_action(self, hand: List[int]) -> List[int]:
        return self.active_bot.draw_action(hand)


ARCHETYPE_CLASSES = [CallingStation, Maniac, Rock, TAG, LAG, AdversarialExploiter]
NUM_ARCHETYPES = len(ARCHETYPE_CLASSES)


def make_opponent_by_id(opp_id: int, rng=None) -> MultiPlayerOpponent:
    cls = ARCHETYPE_CLASSES[opp_id % NUM_ARCHETYPES]
    return cls(rng=rng)


def make_random_archetype(rng: np.random.Generator | None = None) -> MultiPlayerOpponent:
    r = rng or np.random.default_rng()
    opp_id = int(r.integers(0, NUM_ARCHETYPES))
    return make_opponent_by_id(opp_id, rng=r)
# ---------------------------------------------------------------------------
# Multi-Agent League Sparring (Fictitious Play)
# Moved to rl/league.py to keep game/ pure poker simulation logic.
# ---------------------------------------------------------------------------

def __getattr__(name: str):
    if name in ("LeagueOpponent", "LeaguePool", "get_cached_model"):
        try:
            import rl.league as _league
            return getattr(_league, name)
        except ImportError:
            raise AttributeError(f"'{name}' has been moved to 'rl.league' and 'rl' is not available.")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


