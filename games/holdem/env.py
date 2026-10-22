"""
games/holdem/env.py — 5-Seat No-Limit Texas Hold'em Environment.

Core engine features:
1. 5-seat rotating dealer button table structure.
2. Complete multi-street betting rounds: Preflop, Flop (3 cards), Turn (1 card), River (1 card).
3. Fast 7-card showdown evaluation using games.holdem.evaluator.
4. Exact side-pot partitioning and multi-way payout resolution.
5. Action masking and blind escalation support.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

from core.card import Deck, card_str, hand_str
from core.math_utils import calculate_side_pots, resolve_showdown_payouts
from games.holdem.action_abstraction import (
    A_FOLD,
    A_CHECK_CALL,
    A_MIN_RAISE,
    A_HALF_POT,
    A_POT,
    A_ALL_IN,
    NUM_ACTIONS,
    STREET_PREFLOP,
    STREET_FLOP,
    STREET_TURN,
    STREET_RIVER,
    STREET_SHOWDOWN,
    STREETS,
)
from games.holdem.evaluator import evaluate_holdem_hand, holdem_category_name


@dataclass
class HoldemPlayer:
    seat_idx: int
    chips: int
    hand: List[int] = field(default_factory=list)  # 2 hole cards
    is_alive: bool = True
    folded: bool = False
    is_all_in: bool = False
    round_invested: int = 0
    total_invested: int = 0
    bet_to_call: int = 0
    has_acted: bool = False
    street_actions: Dict[str, int] = field(default_factory=lambda: {s: -1 for s in STREETS})
    street_bet_sizes: Dict[str, int] = field(default_factory=lambda: {s: 0 for s in STREETS})

    def reset_for_hand(self):
        self.hand = []
        self.folded = False
        self.is_all_in = (self.chips == 0)
        self.round_invested = 0
        self.total_invested = 0
        self.bet_to_call = 0
        self.has_acted = False
        self.street_actions = {s: -1 for s in STREETS}
        self.street_bet_sizes = {s: 0 for s in STREETS}


class HoldemPokerEnv:
    """
    Standard No-Limit Texas Hold'em table engine.
    """

    def __init__(
        self,
        num_seats: int = 5,
        starting_chips: int = 200,
        small_blind: int = 1,
        big_blind: int = 2,
        max_hands_per_session: int = 150,
        blind_escalation_interval: int = 15,
        dominated_masking: bool = True,
        rng: Optional[np.random.Generator] = None,
        rng_seed: Optional[int] = None,
    ):
        self.num_seats = num_seats
        self.starting_chips = starting_chips
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.base_small_blind = small_blind
        self.base_big_blind = big_blind
        self.max_hands_per_session = max_hands_per_session
        self.blind_escalation_interval = blind_escalation_interval
        self.dominated_masking = dominated_masking
        if rng is None and rng_seed is not None:
            self.rng = np.random.default_rng(rng_seed)
        else:
            self.rng = rng or np.random.default_rng()

        self.deck = Deck(rng=self.rng)
        self.seats: List[HoldemPlayer] = [
            HoldemPlayer(seat_idx=i, chips=starting_chips) for i in range(num_seats)
        ]
        self.board: List[int] = []
        self.dealer_seat: int = 0
        self.current_seat: int = 0
        self.street: str = STREET_PREFLOP
        self.pot: int = 0
        self.current_bet: int = 0
        self.min_raise: int = big_blind
        self.hands_played: int = 0
        self.session_done: bool = False

    @property
    def alive_seats(self) -> List[int]:
        return [s.seat_idx for s in self.seats if s.is_alive and s.chips > 0]

    @property
    def active_seats(self) -> List[int]:
        return [s.seat_idx for s in self.seats if s.is_alive and not s.folded]

    def reset_session(self, randomize_stacks: bool = False):
        self.session_done = False
        self.hands_played = 0
        self.small_blind = self.base_small_blind
        self.big_blind = self.base_big_blind
        self.dealer_seat = int(self.rng.integers(0, self.num_seats))

        for s in self.seats:
            s.is_alive = True
            if randomize_stacks:
                s.chips = int(self.rng.integers(max(10, self.starting_chips // 2), self.starting_chips * 2))
            else:
                s.chips = self.starting_chips

        self.start_hand()

    def start_hand(self):
        alive = self.alive_seats
        if len(alive) <= 1 or self.hands_played >= self.max_hands_per_session:
            self.session_done = True
            return

        # Blind Escalation
        if self.blind_escalation_interval > 0:
            level = self.hands_played // self.blind_escalation_interval
            mult = 2 ** min(level, 4)
            self.small_blind = self.base_small_blind * mult
            self.big_blind = self.base_big_blind * mult

        # Advance Dealer Button
        self.dealer_seat = self._next_alive_seat(self.dealer_seat)
        self.deck.reset()
        self.board = []
        self.pot = 0
        self.current_bet = 0
        self.min_raise = self.big_blind
        self.street = STREET_PREFLOP

        # Deal 2 hole cards to each alive player
        for s in self.seats:
            if s.is_alive and s.chips > 0:
                s.reset_for_hand()
                s.hand = self.deck.deal(2)
            else:
                s.is_alive = False
                s.hand = []
                s.folded = True

        # Post Blinds
        if len(alive) == 2:
            # Heads-up: Dealer posts SB, other player posts BB
            sb_seat = self.dealer_seat
            bb_seat = self._next_alive_seat(self.dealer_seat)
        else:
            sb_seat = self._next_alive_seat(self.dealer_seat)
            bb_seat = self._next_alive_seat(sb_seat)

        self._post_blind(sb_seat, self.small_blind)
        self._post_blind(bb_seat, self.big_blind)

        self.current_bet = max(s.round_invested for s in self.seats)

        # Set bet_to_call for all active players
        for s in self.seats:
            if s.is_alive and not s.folded and not s.is_all_in:
                s.bet_to_call = max(0, self.current_bet - s.round_invested)

        # Preflop action starts:
        # Heads-up: Dealer (SB) acts first preflop
        # Multiway: UTG (left of BB) acts first
        if len(alive) == 2:
            self.current_seat = sb_seat
        else:
            self.current_seat = self._next_alive_seat(bb_seat)

        # If current seat happens to be all-in (rare from posting blind), advance
        if self.seats[self.current_seat].is_all_in:
            self._advance_betting()

    def _next_alive_seat(self, from_seat: int) -> int:
        for step in range(1, self.num_seats + 1):
            s = (from_seat + step) % self.num_seats
            if self.seats[s].is_alive and self.seats[s].chips > 0:
                return s
        return from_seat

    def _post_blind(self, seat_idx: int, amount: int):
        p = self.seats[seat_idx]
        actual = min(p.chips, amount)
        p.chips -= actual
        p.total_invested += actual
        p.round_invested += actual
        self.pot += actual
        p.is_all_in = (p.chips == 0)

    def get_legal_actions(self, seat_idx: int) -> List[int]:
        if self.session_done or self.street == STREET_SHOWDOWN:
            return []

        p = self.seats[seat_idx]
        if not p.is_alive or p.folded or p.is_all_in:
            return []

        legal = [A_CHECK_CALL]
        bet_to_call = p.bet_to_call

        if bet_to_call > 0:
            legal.append(A_FOLD)

        if not self.dominated_masking:
            if p.chips > bet_to_call:
                legal.extend([A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN])
            elif p.chips > 0:
                legal.append(A_ALL_IN)
            return legal

        # --- Dominated & Redundant Action Masking ---
        bb_stack = p.chips / max(1, self.big_blind)
        is_short_stack = (bb_stack <= 20.0)

        # 1. Preflop Discipline
        if self.street == STREET_PREFLOP and len(p.hand) == 2:
            from games.holdem.evaluator import evaluate_preflop_hand_chen
            chen = evaluate_preflop_hand_chen(p.hand[0], p.hand[1])

            # Trash hands (Chen < 0.25) facing an actual raise above the big blind: MUST FOLD
            if chen < 0.25 and bet_to_call > self.big_blind:
                return [A_FOLD]

            # Unplayable trash (Chen < 0.18: 7-2o, 8-3o, etc.) in unraised pot: Cannot open-raise (can only fold/call)
            if chen < 0.18 and bet_to_call <= self.big_blind and not is_short_stack:
                return [A_CHECK_CALL, A_FOLD] if bet_to_call > 0 else [A_CHECK_CALL]

            # Marginal hands facing a significant 3-bet / re-raise: Cannot re-raise
            if chen < 0.40 and bet_to_call >= self.big_blind * 3:
                return [A_CHECK_CALL, A_FOLD] if bet_to_call > 0 else [A_CHECK_CALL]

        # 2. Postflop Air Pruning (Flop, Turn, River)
        elif self.street in (STREET_FLOP, STREET_TURN, STREET_RIVER) and len(self.board) >= 3 and len(p.hand) == 2:
            from games.holdem.board_texture import evaluate_board_texture_and_draws
            tex = evaluate_board_texture_and_draws(p.hand, self.board)
            # Made hands: 5: overpair, 6: top_pair, 7: mid/bottom_pair, 8: two_pair, 9: set/trips, 10: straight+
            has_made = any(tex[k] > 0 for k in range(5, 11))
            # Draws: 11: flush_draw, 12: nut_flush_draw, 13: oesd, 14: gutshot, 15: overcards
            has_draw = any(tex[k] > 0 for k in range(11, 15))
            has_overcards = tex[15] > 0

            if self.street in (STREET_TURN, STREET_RIVER):
                # On Turn/River: only made hands or strong turn draws (flush draw or OESD on turn) can call
                has_strong_turn_draw = (self.street == STREET_TURN and (tex[11] > 0 or tex[13] > 0))
                if not has_made and not has_strong_turn_draw:
                    if bet_to_call > 0:
                        return [A_FOLD]
                    else:
                        # Checked to player with pure air: MUST CHECK (cannot bet air)
                        return [A_CHECK_CALL]
            elif self.street == STREET_FLOP:
                # On Flop: complete air facing a bet must fold
                if not has_made and not has_draw and not has_overcards:
                    if bet_to_call > 0:
                        return [A_FOLD]

        if p.chips <= bet_to_call:
            # Calling already commits all remaining chips (all-in call).
            # A separate all-in shove is completely redundant with call.
            return legal

        chips_after_call = p.chips - bet_to_call
        min_raise_amt = self._calculate_raise_amount(seat_idx, A_MIN_RAISE)
        half_pot_amt = self._calculate_raise_amount(seat_idx, A_HALF_POT)
        pot_amt = self._calculate_raise_amount(seat_idx, A_POT)

        # 1. Min-Raise (only if player has enough chips above bet_to_call)
        if chips_after_call >= min_raise_amt:
            legal.append(A_MIN_RAISE)

        # 2. Half-Pot (must exceed min-raise and player must have enough chips)
        if half_pot_amt > min_raise_amt and chips_after_call >= half_pot_amt:
            legal.append(A_HALF_POT)

        # 3. Pot (must exceed half-pot and player must have enough chips)
        if pot_amt > half_pot_amt and chips_after_call >= pot_amt:
            legal.append(A_POT)

        # 4. All-In Shove Integrity
        can_shove = False
        bb_stack = p.chips / max(1, self.big_blind)
        is_short_stack = (bb_stack <= 20.0)
        facing_heavy_action = (bet_to_call >= p.chips * 0.35) or (bet_to_call >= self.big_blind * 8)
        opponent_is_all_in = any(s.is_all_in for s in self.seats if s.is_alive and not s.folded)
        pot_committed = (p.chips <= self.pot * 1.5)

        if self.street == STREET_PREFLOP and len(p.hand) == 2:
            from games.holdem.evaluator import evaluate_preflop_hand_chen
            chen = evaluate_preflop_hand_chen(p.hand[0], p.hand[1])
            # Only strong hands (Chen >= 0.40), short stacks, or facing heavy action can shove preflop
            if is_short_stack or opponent_is_all_in or facing_heavy_action:
                can_shove = True
            elif chen >= 0.40 and pot_committed:
                can_shove = True
        else:
            # Postflop: permit shove when short-stacked, pot-committed, opponent all-in, or facing heavy action
            if is_short_stack or pot_committed or opponent_is_all_in or facing_heavy_action:
                can_shove = True
            else:
                # Deep-stacked into unraised/small pot: only shove with strong made hand or monster draw
                from games.holdem.board_texture import evaluate_board_texture_and_draws
                tex = evaluate_board_texture_and_draws(p.hand, self.board)
                has_strong_made = (tex[5] > 0 or tex[6] > 0 or tex[8] > 0 or tex[9] > 0 or tex[10] > 0)
                has_monster_draw = (tex[11] > 0 and (tex[13] > 0 or tex[7] > 0))
                if has_strong_made or has_monster_draw:
                    can_shove = True

            # Over-bet shove prevention: deep stacks (> 12 BB) cannot open-shove all-in into tiny pots (> 2x pot)
            if bb_stack > 12.0 and p.chips > self.pot * 2.0:
                can_shove = False

        if can_shove and p.chips > bet_to_call:
            legal.append(A_ALL_IN)

        return legal

    def get_action_mask(self, seat_idx: int) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        for a in self.get_legal_actions(seat_idx):
            mask[a] = 1
        return mask

    def step_action(self, seat_idx: int, action: int):
        assert seat_idx == self.current_seat, f"Not seat {seat_idx}'s turn to act (current: {self.current_seat})"
        legal = self.get_legal_actions(seat_idx)
        if not legal:
            return
        if action not in legal:
            action = A_CHECK_CALL if A_CHECK_CALL in legal else legal[0]

        p = self.seats[seat_idx]
        p.has_acted = True
        p.street_actions[self.street] = action

        if action == A_FOLD:
            p.folded = True
        elif action == A_CHECK_CALL:
            call_cost = min(p.chips, p.bet_to_call)
            p.chips -= call_cost
            p.total_invested += call_cost
            p.round_invested += call_cost
            p.street_bet_sizes[self.street] += call_cost
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
            p.street_bet_sizes[self.street] += chips_committed
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
            if self.street == STREET_PREFLOP and self.current_bet <= self.big_blind:
                # Preflop unraised: standard open is 2.5x BB (add 1.5x BB above big blind)
                return max(self.min_raise, int(self.big_blind * 1.5))
            else:
                return max(self.min_raise, int(self.pot * 0.5))
        elif action == A_POT:
            if self.street == STREET_PREFLOP and self.current_bet <= self.big_blind:
                # Preflop unraised: pot-sized open is 3.5x BB (add 2.5x BB above big blind)
                return max(self.min_raise, int(self.big_blind * 2.5))
            else:
                return max(self.min_raise, self.pot)
        else:  # A_ALL_IN
            return max(0, p.chips - p.bet_to_call)

    def _advance_betting(self):
        active = self.active_seats
        if len(active) <= 1:
            self._award_uncontested_winner(active[0] if active else 0)
            return

        unacted_seats = [
            i for i in active
            if not self.seats[i].is_all_in and (not self.seats[i].has_acted or self.seats[i].bet_to_call > 0)
        ]

        if not unacted_seats:
            self._transition_to_next_street()
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

    def _transition_to_next_street(self):
        # Reset street bets
        for s in self.seats:
            s.round_invested = 0
            s.bet_to_call = 0
            s.has_acted = False

        self.current_bet = 0
        self.min_raise = self.big_blind

        active = self.active_seats
        non_all_in = [i for i in active if not self.seats[i].is_all_in]

        # Check if all or all-but-one players are all-in -> run out board
        if len(non_all_in) <= 1:
            # Deal whatever board cards remain
            cards_needed = 5 - len(self.board)
            if cards_needed > 0:
                held = [c for s in self.seats if s.is_alive and not s.folded for c in s.hand]
                self.board.extend(self.deck.deal(cards_needed, in_use=held + self.board))
            self._resolve_showdown()
            return

        if self.street == STREET_PREFLOP:
            self.street = STREET_FLOP
            held = [c for s in self.seats if s.is_alive and not s.folded for c in s.hand]
            self.board.extend(self.deck.deal(3, in_use=held + self.board))
        elif self.street == STREET_FLOP:
            self.street = STREET_TURN
            held = [c for s in self.seats if s.is_alive and not s.folded for c in s.hand]
            self.board.extend(self.deck.deal(1, in_use=held + self.board))
        elif self.street == STREET_TURN:
            self.street = STREET_RIVER
            held = [c for s in self.seats if s.is_alive and not s.folded for c in s.hand]
            self.board.extend(self.deck.deal(1, in_use=held + self.board))
        elif self.street == STREET_RIVER:
            self._resolve_showdown()
            return

        # Postflop action starts at first active player left of button
        sb_seat = self._next_alive_seat(self.dealer_seat)
        active_acting = [
            (sb_seat + step) % self.num_seats
            for step in range(self.num_seats)
            if self.seats[(sb_seat + step) % self.num_seats].is_alive
            and not self.seats[(sb_seat + step) % self.num_seats].folded
            and not self.seats[(sb_seat + step) % self.num_seats].is_all_in
        ]

        if not active_acting:
            self._transition_to_next_street()
        else:
            self.current_seat = active_acting[0]

    def _resolve_showdown(self):
        self.street = STREET_SHOWDOWN
        active = self.active_seats
        if not active:
            self._end_hand()
            return

        # Ensure board has 5 cards
        cards_needed = 5 - len(self.board)
        if cards_needed > 0:
            held = [c for s in self.seats if s.is_alive and not s.folded for c in s.hand]
            self.board.extend(self.deck.deal(cards_needed, in_use=held + self.board))

        investments = {s.seat_idx: s.total_invested for s in self.seats}
        folded_players = {s.seat_idx for s in self.seats if s.folded}
        hand_scores = {
            s_idx: evaluate_holdem_hand(self.seats[s_idx].hand + self.board)[0]
            for s_idx in active
        }

        tiers = calculate_side_pots(investments, folded_players)
        payouts, details = resolve_showdown_payouts(tiers, hand_scores)

        for s_idx, payout in payouts.items():
            self.seats[s_idx].chips += payout

        self._end_hand()

    def _award_uncontested_winner(self, winner_seat: int):
        self.street = STREET_SHOWDOWN
        self.seats[winner_seat].chips += self.pot
        self._end_hand()

    def _end_hand(self):
        self.hands_played += 1
        for s in self.seats:
            if s.chips <= 0:
                s.is_alive = False

        if len(self.alive_seats) <= 1 or self.hands_played >= self.max_hands_per_session:
            self.session_done = True
        else:
            self.start_hand()
