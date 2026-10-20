"""
games/triple_draw_27/env.py — 5-Seat 2-7 Triple Draw (Deuce-to-Seven Lowball) Table Engine.

Game Structure:
- 5 Seats with rotating Dealer Button, Small Blind, and Big Blind.
- 4 Betting Rounds:
    1. Bet 1 (Pre-Draw)
    2. Bet 2 (Post-Draw 1)
    3. Bet 3 (Post-Draw 2)
    4. Bet 4 (Post-Draw 3 / River)
- 3 Draw Rounds:
    1. Draw 1
    2. Draw 2
    3. Draw 3 (Final Draw)
- Showdown:
    Remaining hands evaluated via evaluate_27_hand() where 7-5-4-3-2 unsuited is the nuts.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Set, Tuple, Optional

from core.card import Deck, rank_of, suit_of
from games.triple_draw_27.rules import evaluate_27_hand, hand_category_27, is_wheel_27
from game.side_pot import calculate_side_pots, resolve_showdown_payouts, PotTier


# Action constants (Identical 38-dim discrete action space)
A_FOLD = 0
A_CALL = 1
A_MIN_RAISE = 2
A_HALF_POT = 3
A_POT = 4
A_ALL_IN = 5
A_DRAW_START = 6
NUM_DRAW_ACTIONS = 32
TOTAL_ACTIONS = A_DRAW_START + NUM_DRAW_ACTIONS  # 38

# Game phases
PHASE_BET_1 = "bet_1"        # Pre-Draw 1
PHASE_DRAW_1 = "draw_1"      # Draw 1
PHASE_BET_2 = "bet_2"        # Post-Draw 1
PHASE_DRAW_2 = "draw_2"      # Draw 2
PHASE_BET_3 = "bet_3"        # Post-Draw 2
PHASE_DRAW_3 = "draw_3"      # Draw 3 (Final Draw)
PHASE_BET_4 = "bet_4"        # Post-Draw 3 (River)
PHASE_SHOWDOWN = "showdown"


class PlayerState27:
    """State of a single table seat in 2-7 Triple Draw."""

    def __init__(self, seat_idx: int, starting_chips: int):
        self.seat_idx = seat_idx
        self.chips = starting_chips
        self.is_alive = True
        self.hand: List[int] = []
        self.folded = False
        self.total_invested = 0
        self.round_invested = 0
        self.bet_to_call = 0
        self.is_all_in = False
        self.has_acted = False

        # 3-Draw tracking: -1 if not yet drawn, 0-5 after draw
        self.draw_counts: List[int] = [-1, -1, -1]

        # Multi-street action memory
        self.street_actions: List[int] = [-1, -1, -1, -1]
        self.street_bet_sizes: List[int] = [0, 0, 0, 0]

    def reset_for_hand(self):
        self.hand = []
        self.folded = False
        self.total_invested = 0
        self.round_invested = 0
        self.bet_to_call = 0
        self.has_acted = False
        self.is_all_in = (self.chips == 0)
        self.draw_counts = [-1, -1, -1]
        self.street_actions = [-1, -1, -1, -1]
        self.street_bet_sizes = [0, 0, 0, 0]


class TripleDrawPokerEnv:
    """Core table engine for 5-Seat 2-7 Triple Draw Lowball."""

    def __init__(
        self,
        num_seats: int = 5,
        starting_chips: int = 200,
        small_blind: int = 1,
        big_blind: int = 2,
        rng_seed: Optional[int] = None,
        max_hands_per_session: int = 150,
        blind_escalation_interval: Optional[int] = None,
    ):
        self.num_seats = num_seats
        self.starting_chips = starting_chips
        self.base_small_blind = small_blind
        self.base_big_blind = big_blind
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.blind_escalation_interval = blind_escalation_interval
        self.max_hands_per_session = max_hands_per_session

        self.rng = np.random.default_rng(rng_seed)
        self.deck = Deck(rng=self.rng)

        self.seats: List[PlayerState27] = [
            PlayerState27(i, starting_chips) for i in range(num_seats)
        ]

        self.dealer_seat = 0
        self.current_seat = 0
        self.phase = PHASE_BET_1
        self.pot = 0
        self.current_bet = 0
        self.min_raise = big_blind
        self.hands_played = 0
        self.session_done = False
        self.draw_seat_queue: List[int] = []

    @property
    def alive_seats(self) -> List[int]:
        return [i for i, s in enumerate(self.seats) if s.is_alive and s.chips > 0]

    @property
    def active_seats(self) -> List[int]:
        return [i for i, s in enumerate(self.seats) if s.is_alive and not s.folded]

    def reset_tournament(self):
        for s in self.seats:
            s.chips = self.starting_chips
            s.is_alive = True
            s.reset_for_hand()
        self.small_blind = self.base_small_blind
        self.big_blind = self.base_big_blind
        self.hands_played = 0
        self.session_done = False
        self.dealer_seat = int(self.rng.integers(0, self.num_seats))
        self.start_hand()

    def start_hand(self):
        alive = self.alive_seats
        if len(alive) <= 1 or self.hands_played >= self.max_hands_per_session:
            self.session_done = True
            return

        # Blind escalations
        if self.blind_escalation_interval and self.hands_played > 0:
            level = self.hands_played // self.blind_escalation_interval
            mult = 2 ** level
            self.small_blind = self.base_small_blind * mult
            self.big_blind = self.base_big_blind * mult

        # Advance dealer button to next alive seat
        self.dealer_seat = self._next_alive_seat(self.dealer_seat)

        for s in self.seats:
            s.reset_for_hand()

        self.deck.reset()
        self.pot = 0
        self.current_bet = self.big_blind
        self.min_raise = self.big_blind
        self.phase = PHASE_BET_1

        # Deal 5 cards to all alive players
        for s_idx in self.alive_seats:
            self.seats[s_idx].hand = self.deck.deal(5)

        # Post Blinds
        sb_seat = self._next_alive_seat(self.dealer_seat)
        bb_seat = self._next_alive_seat(sb_seat)

        self._post_blind(sb_seat, self.small_blind)
        self._post_blind(bb_seat, self.big_blind)

        # Update bet_to_call for all active players
        for s_idx in self.active_seats:
            p = self.seats[s_idx]
            p.bet_to_call = max(0, self.current_bet - p.round_invested)

        # Pre-draw action starts left of Big Blind (UTG)
        self.current_seat = self._next_alive_seat(bb_seat)

    def _next_alive_seat(self, seat: int) -> int:
        for step in range(1, self.num_seats + 1):
            nxt = (seat + step) % self.num_seats
            if self.seats[nxt].is_alive and self.seats[nxt].chips > 0:
                return nxt
        return seat

    def _post_blind(self, seat_idx: int, amount: int):
        p = self.seats[seat_idx]
        actual = min(p.chips, amount)
        p.chips -= actual
        p.total_invested += actual
        p.round_invested += actual
        self.pot += actual
        p.is_all_in = (p.chips == 0)

    def get_legal_actions(self, seat_idx: int) -> List[int]:
        if self.session_done or self.phase == PHASE_SHOWDOWN:
            return []

        p = self.seats[seat_idx]
        if not p.is_alive or p.folded:
            return []

        # Draw Phase (Draw 1, Draw 2, or Draw 3)
        if self.phase in (PHASE_DRAW_1, PHASE_DRAW_2, PHASE_DRAW_3):
            return [A_DRAW_START + bitmask for bitmask in range(NUM_DRAW_ACTIONS)]

        # Betting Phase (Bet 1, Bet 2, Bet 3, Bet 4)
        if p.is_all_in:
            return []

        legal = [A_FOLD, A_CALL]
        bet_to_call = p.bet_to_call

        if p.chips > bet_to_call:
            legal.append(A_MIN_RAISE)
            legal.append(A_HALF_POT)
            legal.append(A_POT)
            legal.append(A_ALL_IN)
        elif p.chips > 0:
            legal.append(A_ALL_IN)

        return legal

    def get_action_mask(self, seat_idx: int) -> np.ndarray:
        mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
        for a in self.get_legal_actions(seat_idx):
            mask[a] = 1
        return mask

    def step_action(self, seat_idx: int, action: int):
        assert seat_idx == self.current_seat, f"Not seat {seat_idx}'s turn to act (current: {self.current_seat})"
        legal = self.get_legal_actions(seat_idx)
        if not legal:
            return
        if action not in legal:
            action = A_CALL if A_CALL in legal else legal[0]

        p = self.seats[seat_idx]
        p.has_acted = True

        # Process Draw Action
        if self.phase in (PHASE_DRAW_1, PHASE_DRAW_2, PHASE_DRAW_3):
            bitmask = action - A_DRAW_START
            draw_round_idx = 0 if self.phase == PHASE_DRAW_1 else (1 if self.phase == PHASE_DRAW_2 else 2)
            self._execute_draw(seat_idx, bitmask, draw_round_idx)
            self._advance_draw_queue()
            return

        # Process Betting Action
        street_idx = 0 if self.phase == PHASE_BET_1 else (1 if self.phase == PHASE_BET_2 else (2 if self.phase == PHASE_BET_3 else 3))
        p.street_actions[street_idx] = action

        if action == A_FOLD:
            p.folded = True
        elif action == A_CALL:
            call_cost = min(p.chips, p.bet_to_call)
            p.chips -= call_cost
            p.total_invested += call_cost
            p.round_invested += call_cost
            p.street_bet_sizes[street_idx] += call_cost
            self.pot += call_cost
            p.bet_to_call = 0
            p.is_all_in = (p.chips == 0)
        elif action in (A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN):
            raise_amount = self._calculate_raise_amount(seat_idx, action)
            chips_needed = p.bet_to_call + raise_amount
            chips_committed = min(p.chips, chips_needed)

            p.chips -= chips_committed
            p.total_invested += chips_committed
            p.round_invested += chips_committed
            p.street_bet_sizes[street_idx] += chips_committed
            self.pot += chips_committed
            p.bet_to_call = 0
            p.is_all_in = (p.chips == 0)

            if p.round_invested > self.current_bet:
                actual_raise = p.round_invested - self.current_bet
                self.current_bet = p.round_invested
                self.min_raise = max(self.big_blind, actual_raise)

                # Reopen betting for other active non-all-in players
                for s_i in self.active_seats:
                    if s_i != seat_idx and not self.seats[s_i].is_all_in:
                        self.seats[s_i].bet_to_call = self.current_bet - self.seats[s_i].round_invested
                        self.seats[s_i].has_acted = False

        self._advance_betting()

    def _calculate_raise_amount(self, seat_idx: int, action: int) -> int:
        p = self.seats[seat_idx]
        if action == A_MIN_RAISE:
            return self.min_raise
        elif action == A_HALF_POT:
            return max(self.min_raise, int(self.pot * 0.5))
        elif action == A_POT:
            return max(self.min_raise, self.pot)
        else:  # A_ALL_IN
            return p.chips - p.bet_to_call

    def _execute_draw(self, seat_idx: int, bitmask: int, round_idx: int):
        p = self.seats[seat_idx]
        discards = [i for i in range(5) if (bitmask & (1 << i))]
        num_discard = len(discards)
        p.draw_counts[round_idx] = num_discard

        if num_discard > 0:
            held = [c for s in self.seats if s.is_alive and not s.folded for c in s.hand if c != -1]
            replacements = self.deck.draw(num_discard, in_use=held)
            for d_idx, rep_card in zip(discards, replacements):
                p.hand[d_idx] = rep_card

    def _advance_betting(self):
        active = self.active_seats
        # Check if only 1 player remains
        if len(active) <= 1:
            self._award_uncontested_winner(active[0] if active else 0)
            return

        # Check if betting round complete
        unacted_seats = [
            i for i in active
            if not self.seats[i].is_all_in and (not self.seats[i].has_acted or self.seats[i].bet_to_call > 0)
        ]

        if not unacted_seats:
            self._transition_to_next_phase()
        else:
            self.current_seat = self._find_next_acting_seat(self.current_seat)

    def _find_next_acting_seat(self, from_seat: int) -> int:
        for step in range(1, self.num_seats + 1):
            s = (from_seat + step) % self.num_seats
            p = self.seats[s]
            if p.is_alive and not p.folded and not p.is_all_in:
                if not p.has_acted or p.bet_to_call > 0:
                    return s
        return from_seat

    def _transition_to_next_phase(self):
        # Reset round bets
        for s in self.seats:
            s.round_invested = 0
            s.bet_to_call = 0
            s.has_acted = False

        self.current_bet = 0
        self.min_raise = self.big_blind

        if self.phase == PHASE_BET_1:
            self._enter_draw_phase(PHASE_DRAW_1)
        elif self.phase == PHASE_BET_2:
            self._enter_draw_phase(PHASE_DRAW_2)
        elif self.phase == PHASE_BET_3:
            self._enter_draw_phase(PHASE_DRAW_3)
        elif self.phase == PHASE_BET_4:
            self._resolve_showdown()

    def _enter_draw_phase(self, draw_phase: str):
        self.phase = draw_phase
        # Queue active players who are not folded (even all-in players draw)
        sb_seat = self._next_alive_seat(self.dealer_seat)
        queue = []
        for step in range(self.num_seats):
            s = (sb_seat + step) % self.num_seats
            if self.seats[s].is_alive and not self.seats[s].folded:
                queue.append(s)

        self.draw_seat_queue = queue
        self._advance_draw_queue()

    def _advance_draw_queue(self):
        if not self.draw_seat_queue:
            # All active players drew, enter next betting round
            if self.phase == PHASE_DRAW_1:
                self.phase = PHASE_BET_2
            elif self.phase == PHASE_DRAW_2:
                self.phase = PHASE_BET_3
            else:
                self.phase = PHASE_BET_4

            # Betting action starts at first active player left of button
            sb_seat = self._next_alive_seat(self.dealer_seat)
            active_acting = [
                (sb_seat + step) % self.num_seats
                for step in range(self.num_seats)
                if self.seats[(sb_seat + step) % self.num_seats].is_alive
                and not self.seats[(sb_seat + step) % self.num_seats].folded
                and not self.seats[(sb_seat + step) % self.num_seats].is_all_in
            ]

            if not active_acting:
                # All players are all-in, transition to next phase
                self._transition_to_next_phase()
            else:
                self.current_seat = active_acting[0]
        else:
            self.current_seat = self.draw_seat_queue.pop(0)

    def _resolve_showdown(self):
        self.phase = PHASE_SHOWDOWN
        active = self.active_seats
        if not active:
            self._end_hand()
            return

        # Evaluate all hands using 2-7 lowball scoring
        investments = {s.seat_idx: s.total_invested for s in self.seats}
        folded_players = {s.seat_idx for s in self.seats if s.folded}
        hand_scores = {s_idx: evaluate_27_hand(self.seats[s_idx].hand) for s_idx in active}

        tiers = calculate_side_pots(investments, folded_players)
        payouts, details = resolve_showdown_payouts(tiers, hand_scores)

        for s_idx, payout in payouts.items():
            self.seats[s_idx].chips += payout

        self._end_hand()

    def _award_uncontested_winner(self, winner_seat: int):
        self.phase = PHASE_SHOWDOWN
        self.seats[winner_seat].chips += self.pot
        self._end_hand()

    def _end_hand(self):
        self.hands_played += 1
        # Eliminate busted players
        for s in self.seats:
            if s.chips <= 0:
                s.is_alive = False

        if len(self.alive_seats) <= 1 or self.hands_played >= self.max_hands_per_session:
            self.session_done = True
        else:
            self.start_hand()
