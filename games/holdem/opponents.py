"""
games/holdem/opponents.py — Heuristic Opponent Archetypes for Texas Hold'em.

Archetypes:
1. TAGBot (Tight-Aggressive): Disciplined preflop ranges, aggressive value bets.
2. LAGBot (Loose-Aggressive): Wide ranges, high bluff frequency and pressure.
3. CallingStationBot: Passive calling station, calls down with weak pairs.
4. RandomHoldemBot: Stochastic baseline for environment exploration.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np

from core.card import rank_of, suit_of
from games.holdem.action_abstraction import (
    A_FOLD,
    A_CHECK_CALL,
    A_MIN_RAISE,
    A_HALF_POT,
    A_POT,
    A_ALL_IN,
    STREET_PREFLOP,
    STREET_FLOP,
    STREET_TURN,
    STREET_RIVER,
)
from games.holdem.evaluator import evaluate_holdem_hand, holdem_category


class HoldemOpponent:
    def __init__(self, opponent_id: int, name: str, rng: Optional[np.random.Generator] = None):
        self.opponent_id = opponent_id
        self.name = name
        self.rng = rng or np.random.default_rng()

    def act(
        self,
        hole_cards: List[int],
        board: List[int],
        pot: int,
        bet_to_call: int,
        street: str,
        stack: int,
        legal_actions: List[int],
    ) -> int:
        raise NotImplementedError


class TAGBot(HoldemOpponent):
    """Tight-Aggressive (TAG): Selective preflop, aggressive postflop."""

    def __init__(self, opponent_id: int = 1, name: str = "TAGBot", rng=None):
        super().__init__(opponent_id, name, rng)

    def act(
        self,
        hole_cards: List[int],
        board: List[int],
        pot: int,
        bet_to_call: int,
        street: str,
        stack: int,
        legal_actions: List[int],
    ) -> int:
        if len(hole_cards) < 2:
            return A_CHECK_CALL if A_CHECK_CALL in legal_actions else legal_actions[0]

        r1, r2 = rank_of(hole_cards[0]), rank_of(hole_cards[1])
        s1, s2 = suit_of(hole_cards[0]), suit_of(hole_cards[1])
        is_pair = (r1 == r2)
        is_suited = (s1 == s2)
        high_r = max(r1, r2)
        low_r = min(r1, r2)

        if street == STREET_PREFLOP:
            # Premium pairs (TT+) or big cards (AK, AQ)
            if (is_pair and high_r >= 8) or (high_r >= 11 and low_r >= 9):
                for a in [A_POT, A_HALF_POT, A_MIN_RAISE]:
                    if a in legal_actions:
                        return a
                return A_CHECK_CALL
            # Playable speculative hands (pairs, suited broadway)
            elif is_pair or (is_suited and high_r >= 9):
                if bet_to_call <= pot * 0.35:
                    return A_CHECK_CALL
                return A_FOLD if A_FOLD in legal_actions else A_CHECK_CALL
            else:
                if bet_to_call == 0:
                    return A_CHECK_CALL
                return A_FOLD if A_FOLD in legal_actions else A_CHECK_CALL

        # Postflop
        all_cards = hole_cards + board
        category = holdem_category(all_cards)  # 0=High card, 1=Pair, 2=Two pair, etc.

        if category >= 2:  # Two pair or better
            for a in [A_POT, A_HALF_POT, A_MIN_RAISE]:
                if a in legal_actions:
                    return a
            return A_CHECK_CALL
        elif category == 1:  # One pair
            if bet_to_call <= pot * 0.4:
                return A_CHECK_CALL
            return A_FOLD if A_FOLD in legal_actions else A_CHECK_CALL
        else:
            if bet_to_call == 0:
                return A_CHECK_CALL
            return A_FOLD if A_FOLD in legal_actions else A_CHECK_CALL


class LAGBot(HoldemOpponent):
    """Loose-Aggressive (LAG): Plays many hands, bets and raises frequently."""

    def __init__(self, opponent_id: int = 2, name: str = "LAGBot", rng=None):
        super().__init__(opponent_id, name, rng)

    def act(
        self,
        hole_cards: List[int],
        board: List[int],
        pot: int,
        bet_to_call: int,
        street: str,
        stack: int,
        legal_actions: List[int],
    ) -> int:
        if len(hole_cards) < 2:
            return A_CHECK_CALL if A_CHECK_CALL in legal_actions else legal_actions[0]

        r1, r2 = rank_of(hole_cards[0]), rank_of(hole_cards[1])
        high_r = max(r1, r2)

        # Preflop aggression
        if street == STREET_PREFLOP:
            if high_r >= 7 or r1 == r2:
                if self.rng.random() < 0.6:
                    for a in [A_HALF_POT, A_MIN_RAISE]:
                        if a in legal_actions:
                            return a
                return A_CHECK_CALL
            if bet_to_call <= pot * 0.2:
                return A_CHECK_CALL
            return A_FOLD if A_FOLD in legal_actions else A_CHECK_CALL

        # Postflop: frequent c-bets and floats
        all_cards = hole_cards + board
        category = holdem_category(all_cards)

        if category >= 1:
            if self.rng.random() < 0.5:
                for a in [A_HALF_POT, A_MIN_RAISE]:
                    if a in legal_actions:
                        return a
            return A_CHECK_CALL
        else:
            # 25% pure bluff
            if self.rng.random() < 0.25 and bet_to_call == 0:
                if A_HALF_POT in legal_actions:
                    return A_HALF_POT
            if bet_to_call == 0:
                return A_CHECK_CALL
            return A_FOLD if A_FOLD in legal_actions else A_CHECK_CALL


class CallingStationBot(HoldemOpponent):
    """Passive Calling Station: Rarely folds, rarely raises."""

    def __init__(self, opponent_id: int = 3, name: str = "CallingStation", rng=None):
        super().__init__(opponent_id, name, rng)

    def act(
        self,
        hole_cards: List[int],
        board: List[int],
        pot: int,
        bet_to_call: int,
        street: str,
        stack: int,
        legal_actions: List[int],
    ) -> int:
        return A_CHECK_CALL if A_CHECK_CALL in legal_actions else legal_actions[0]


class RandomHoldemBot(HoldemOpponent):
    """Uniform random choices across legal actions."""

    def __init__(self, opponent_id: int = 4, name: str = "RandomBot", rng=None):
        super().__init__(opponent_id, name, rng)

    def act(
        self,
        hole_cards: List[int],
        board: List[int],
        pot: int,
        bet_to_call: int,
        street: str,
        stack: int,
        legal_actions: List[int],
    ) -> int:
        if not legal_actions:
            return A_CHECK_CALL
        return int(self.rng.choice(legal_actions))


def make_random_holdem_opponent(opp_id: int, rng: Optional[np.random.Generator] = None) -> HoldemOpponent:
    r = rng or np.random.default_rng()
    roll = r.random()
    if roll < 0.35:
        return TAGBot(opponent_id=opp_id, rng=r)
    elif roll < 0.65:
        return LAGBot(opponent_id=opp_id, rng=r)
    elif roll < 0.85:
        return CallingStationBot(opponent_id=opp_id, rng=r)
    else:
        return RandomHoldemBot(opponent_id=opp_id, rng=r)
