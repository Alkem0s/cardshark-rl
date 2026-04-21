"""
opponents.py — Three hard-coded opponent archetypes for 5-Card Draw Poker.

Each archetype implements:
    bet_action(hand, pot, bet_to_call, phase, can_raise) → int   (0=Fold, 1=Call/Check, 2=Raise)
    draw_action(hand) → List[int]   (indices of cards to discard, 0-4)
"""

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import List

from card_utils import (
    rank_of, suit_of, hand_category, get_pairs_info,
    has_flush_draw, has_straight_draw, evaluate_hand,
)


class Opponent(ABC):
    """Base class for hard-coded opponent policies."""

    FOLD = 0
    CALL = 1
    RAISE = 2

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
        can_raise: bool,
    ) -> int:
        """Return 0 (fold), 1 (call/check), or 2 (raise/bet)."""
        ...

    @abstractmethod
    def draw_action(self, hand: List[int]) -> List[int]:
        """Return list of card *indices* (0-4) to discard."""
        ...


# ---------------------------------------------------------------------------
# 1. The Calling Station
# ---------------------------------------------------------------------------

class CallingStation(Opponent):
    """Never bluffs, never raises, always calls.  Draws mathematically."""

    def __init__(self, rng=None):
        super().__init__(opponent_id=0, name="CallingStation", rng=rng)

    def bet_action(self, hand, pot, bet_to_call, phase, can_raise) -> int:
        # Always call (or check if nothing to call)
        return self.CALL

    def draw_action(self, hand: List[int]) -> List[int]:
        """Mathematically optimal draw: keep best group, discard kickers."""
        info = get_pairs_info(hand)
        cat = info["category"]

        # Straight / Flush / Full House+ → stand pat
        if cat >= 4:
            return []

        # Three of a kind → discard 2 kickers
        if cat == 3:
            return info["kicker_indices"]

        # Two pair → discard the 1 kicker
        if cat == 2:
            return info["kicker_indices"]

        # One pair → discard 3 kickers
        if cat == 1:
            return info["kicker_indices"]

        # High card → keep the two highest ranked cards, discard 3
        ranks_with_idx = [(rank_of(hand[i]), i) for i in range(5)]
        ranks_with_idx.sort(reverse=True)
        keep = {ranks_with_idx[0][1], ranks_with_idx[1][1]}
        return [i for i in range(5) if i not in keep]


# ---------------------------------------------------------------------------
# 2. The Maniac / Gambler
# ---------------------------------------------------------------------------

class Maniac(Opponent):
    """Bets aggressively.  Draws 1 card chasing straights/flushes frequently."""

    def __init__(self, rng=None):
        super().__init__(opponent_id=1, name="Maniac", rng=rng)

    def bet_action(self, hand, pot, bet_to_call, phase, can_raise) -> int:
        r = self.rng.random()
        if can_raise and r < 0.70:
            return self.RAISE
        if r < 0.95:
            return self.CALL
        return self.FOLD  # Rare fold (5%)

    def draw_action(self, hand: List[int]) -> List[int]:
        info = get_pairs_info(hand)
        cat = info["category"]

        # Made hand (trips+) → stand pat
        if cat >= 3:
            return []

        # 60% of the time, chase a flush/straight draw by drawing 1 card
        if self.rng.random() < 0.60:
            # Try flush draw first
            is_fd, fd_suit = has_flush_draw(hand)
            if is_fd:
                # Discard the one card that doesn't match the flush suit
                return [i for i in range(5) if suit_of(hand[i]) != fd_suit]

            # Try straight draw
            is_sd, keep_idx = has_straight_draw(hand)
            if is_sd and len(keep_idx) == 4:
                return [i for i in range(5) if i not in keep_idx]

            # No real draw — just discard 1 random card anyway (bluff draw)
            return [int(self.rng.integers(0, 5))]

        # 40% of the time, play normally
        if cat == 2:  # Two pair
            return info["kicker_indices"]
        if cat == 1:  # Pair
            return info["kicker_indices"]

        # High card: keep highest 2, toss 3
        ranks_with_idx = [(rank_of(hand[i]), i) for i in range(5)]
        ranks_with_idx.sort(reverse=True)
        keep = {ranks_with_idx[0][1], ranks_with_idx[1][1]}
        return [i for i in range(5) if i not in keep]


# ---------------------------------------------------------------------------
# 3. The Rock / Nit
# ---------------------------------------------------------------------------

class Rock(Opponent):
    """Only plays strong hands.  Tight and predictable.

    Pre-draw: Folds junk, calls with a pair, raises with high pair (JJ+).
    Post-draw: Folds to aggression unless improved.  Raises with trips+.
    Draw: Stands pat with two-pair+.  Draws 3 to a pair.
    """

    HIGH_PAIR_THRESHOLD = 9  # Rank of Jack (0=2,...,9=J,10=Q,11=K,12=A)

    def __init__(self, rng=None):
        super().__init__(opponent_id=2, name="Rock", rng=rng)

    def bet_action(self, hand, pot, bet_to_call, phase, can_raise) -> int:
        info = get_pairs_info(hand)
        cat = info["category"]

        if phase == "pre_draw":
            # No pair → fold if there's a bet, check if free
            if cat == 0:
                return self.CALL if bet_to_call == 0 else self.FOLD

            # One pair below JJ → call but don't raise
            if cat == 1:
                best_pair_rank = max(info["pair_ranks"])
                if best_pair_rank < self.HIGH_PAIR_THRESHOLD:
                    return self.CALL if bet_to_call <= 2 else self.FOLD
                # High pair (JJ+): raise if possible
                if can_raise:
                    return self.RAISE
                return self.CALL

            # Two pair+ → raise
            if can_raise:
                return self.RAISE
            return self.CALL

        else:  # post_draw
            # Junk → fold to any bet
            if cat == 0:
                return self.CALL if bet_to_call == 0 else self.FOLD

            # One pair → call small bets, fold to raises
            if cat == 1:
                return self.CALL if bet_to_call <= 4 else self.FOLD

            # Two pair → call
            if cat == 2:
                return self.CALL

            # Trips+ → raise
            if can_raise:
                return self.RAISE
            return self.CALL

    def draw_action(self, hand: List[int]) -> List[int]:
        info = get_pairs_info(hand)
        cat = info["category"]

        # Two pair+ or straight/flush → stand pat
        if cat >= 2:
            return []

        # One pair → draw 3 (discard kickers)
        if cat == 1:
            return info["kicker_indices"]

        # High card (shouldn't really be here pre-draw, but handle it)
        # Keep the highest card, discard 4
        ranks_with_idx = [(rank_of(hand[i]), i) for i in range(5)]
        ranks_with_idx.sort(reverse=True)
        keep = {ranks_with_idx[0][1]}
        return [i for i in range(5) if i not in keep]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

OPPONENT_CLASSES = [CallingStation, Maniac, Rock]
NUM_ARCHETYPES = len(OPPONENT_CLASSES)

def make_opponent(opponent_id: int, rng=None) -> Opponent:
    """Factory: create an opponent by its integer ID."""
    return OPPONENT_CLASSES[opponent_id](rng=rng)

def make_random_opponent(rng: np.random.Generator) -> Opponent:
    """Create a random archetype."""
    idx = int(rng.integers(0, NUM_ARCHETYPES))
    return make_opponent(idx, rng=rng)
