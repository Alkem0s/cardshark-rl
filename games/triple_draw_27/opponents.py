"""
games/triple_draw_27/opponents.py — Heuristic Opponent Bots for 2-7 Triple Draw.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional
from core.card import rank_of, suit_of
from games.triple_draw_27.rules import (
    evaluate_27_hand,
    hand_category_27,
    get_lowball_draw_candidates,
    is_wheel_27,
)
from games.triple_draw_27.env import (
    A_FOLD,
    A_CALL,
    A_MIN_RAISE,
    A_HALF_POT,
    A_POT,
    A_ALL_IN,
    A_DRAW_START,
)


class LowballOpponent:
    """Base class for 2-7 Triple Draw rule-based heuristic agents."""

    def __init__(self, opponent_id: int, name: str, rng: Optional[np.random.Generator] = None):
        self.opponent_id = opponent_id
        self.name = name
        self.rng = rng or np.random.default_rng()

    def draw_action(self, hand: List[int], draw_round: int = 0) -> List[int]:
        """Returns list of card indices (0..4) to discard."""
        raise NotImplementedError

    def bet_action(
        self,
        hand: List[int],
        pot: int,
        bet_to_call: int,
        phase: str,
        stack: int,
        legal_actions: List[int],
        opponents_draw_counts: Optional[List[int]] = None,
    ) -> int:
        """Returns integer action index."""
        raise NotImplementedError


class PatStandBot(LowballOpponent):
    """
    Tight-Aggressive 2-7 Lowball Bot.
    Stands pat on any 8-or-better low, otherwise draws aggressively toward 7-5-4-3-2.
    Value-bets when pat, folds weak pairs to heavy aggression.
    """

    def __init__(self, opponent_id: int = 1, name: str = "PatStandBot", rng=None):
        super().__init__(opponent_id, name, rng)

    def draw_action(self, hand: List[int], draw_round: int = 0) -> List[int]:
        tier = hand_category_27(hand)
        highest_r = max(rank_of(c) for c in hand)
        # Stand pat on 8-low or better
        if tier == 8 and highest_r <= 6:
            return []
        # On 3rd draw, stand pat even on 9-low
        if draw_round == 2 and tier == 8 and highest_r <= 7:
            return []
        return get_lowball_draw_candidates(hand)

    def bet_action(
        self,
        hand: List[int],
        pot: int,
        bet_to_call: int,
        phase: str,
        stack: int,
        legal_actions: List[int],
        opponents_draw_counts: Optional[List[int]] = None,
    ) -> int:
        tier = hand_category_27(hand)
        highest_r = max(rank_of(c) for c in hand)
        is_pat = (tier == 8 and highest_r <= 7)

        if is_pat:
            # Bet or raise with made low
            for a in [A_POT, A_HALF_POT, A_MIN_RAISE]:
                if a in legal_actions:
                    return a
            return A_CALL

        if bet_to_call == 0:
            return A_CALL

        # Facing bet
        if bet_to_call <= pot * 0.4 and (tier >= 7 or highest_r <= 8):
            return A_CALL
        if A_FOLD in legal_actions:
            return A_FOLD
        return A_CALL


class SmoothDrawBot(LowballOpponent):
    """
    Methodical drawing bot.
    Systematically discards pairs and high cards to construct smooth lows.
    """

    def __init__(self, opponent_id: int = 2, name: str = "SmoothDrawBot", rng=None):
        super().__init__(opponent_id, name, rng)

    def draw_action(self, hand: List[int], draw_round: int = 0) -> List[int]:
        tier = hand_category_27(hand)
        highest_r = max(rank_of(c) for c in hand)
        if tier == 8 and highest_r <= 7:
            return []
        return get_lowball_draw_candidates(hand)

    def bet_action(
        self,
        hand: List[int],
        pot: int,
        bet_to_call: int,
        phase: str,
        stack: int,
        legal_actions: List[int],
        opponents_draw_counts: Optional[List[int]] = None,
    ) -> int:
        tier = hand_category_27(hand)
        highest_r = max(rank_of(c) for c in hand)

        if tier == 8 and highest_r <= 6:  # 8-low or better
            if A_HALF_POT in legal_actions:
                return A_HALF_POT
            if A_MIN_RAISE in legal_actions:
                return A_MIN_RAISE
            return A_CALL

        if bet_to_call == 0:
            return A_CALL

        if bet_to_call <= pot * 0.35:
            return A_CALL
        return A_FOLD if A_FOLD in legal_actions else A_CALL


class LowballExploiter(LowballOpponent):
    """
    Adversarial 2-7 Nemesis.
    Tracks draw counts of opponents: if opponents draw 2+ cards on Draw 2 or 3,
    fires aggressive probe bets to force folds. Occasionally 'snows' (stands pat on a bluff).
    """

    def __init__(self, opponent_id: int = 3, name: str = "LowballExploiter", rng=None):
        super().__init__(opponent_id, name, rng)
        self.snowing = False

    def draw_action(self, hand: List[int], draw_round: int = 0) -> List[int]:
        # Snowing bluff: 10% chance to stand pat on Draw 2 or 3 even with pair of high cards
        if draw_round >= 1 and self.rng.random() < 0.10:
            self.snowing = True
            return []

        self.snowing = False
        tier = hand_category_27(hand)
        highest_r = max(rank_of(c) for c in hand)
        if tier == 8 and highest_r <= 7:
            return []
        return get_lowball_draw_candidates(hand)

    def bet_action(
        self,
        hand: List[int],
        pot: int,
        bet_to_call: int,
        phase: str,
        stack: int,
        legal_actions: List[int],
        opponents_draw_counts: Optional[List[int]] = None,
    ) -> int:
        tier = hand_category_27(hand)
        highest_r = max(rank_of(c) for c in hand)

        # If snowing or holding a strong low, bet heavily
        if self.snowing or (tier == 8 and highest_r <= 6):
            for a in [A_POT, A_HALF_POT, A_MIN_RAISE]:
                if a in legal_actions:
                    return a
            return A_CALL

        # Check if opponents showed weakness by drawing 2+ cards
        opp_drawing_heavy = False
        if opponents_draw_counts:
            opp_drawing_heavy = any(cnt >= 2 for cnt in opponents_draw_counts if cnt >= 0)

        if opp_drawing_heavy and bet_to_call == 0 and self.rng.random() < 0.65:
            # Probe bet
            if A_HALF_POT in legal_actions:
                return A_HALF_POT
            if A_MIN_RAISE in legal_actions:
                return A_MIN_RAISE

        if bet_to_call == 0:
            return A_CALL

        if bet_to_call <= pot * 0.3:
            return A_CALL
        return A_FOLD if A_FOLD in legal_actions else A_CALL


def make_random_lowball_opponent(opp_id: int = 4, rng=None) -> LowballOpponent:
    r = rng or np.random.default_rng()
    choice = r.choice(["pat", "smooth", "exploiter"])
    if choice == "pat":
        return PatStandBot(opponent_id=opp_id, rng=r)
    elif choice == "smooth":
        return SmoothDrawBot(opponent_id=opp_id, rng=r)
    else:
        return LowballExploiter(opponent_id=opp_id, rng=r)
