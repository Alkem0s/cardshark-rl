"""
multi_draw_poker_env.py — 5-Seat Multi-Player 5-Card Draw Poker Environment.

Features:
- Up to 5 persistent seats (Hero can be at any seat).
- Rotating Dealer Button, Small Blind, and Big Blind.
- Deck reshuffle safety when discards exhaust the deck stub.
- Discrete(38) scale-invariant action space (Fold, Call, Min, 0.5x, 1.0x, All-in, 32 Draws).
- Side-pot resolution for arbitrary stack sizes and all-in situations.
- Chip exhaustion / elimination (busted players marked inactive, table shrinks dynamically).
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Set, Tuple, Optional

from card_utils import Deck, evaluate_hand, rank_of, suit_of, hand_category
from side_pot import calculate_side_pots, resolve_showdown_payouts, PotTier


# Action constants
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
PHASE_PRE_DRAW = "pre_draw"
PHASE_DRAW = "draw"
PHASE_POST_DRAW = "post_draw"
PHASE_SHOWDOWN = "showdown"


class PlayerState:
    """State of a single table seat."""
    def __init__(self, seat_idx: int, starting_chips: int):
        self.seat_idx = seat_idx
        self.chips = starting_chips
        self.is_alive = True          # False when eliminated (chips == 0)
        self.hand: List[int] = []
        self.folded = False
        self.total_invested = 0       # Total chips put in pot across the whole hand
        self.round_invested = 0       # Chips put in during current betting round
        self.bet_to_call = 0
        self.draw_count = -1          # -1 before draw phase, 0-5 after
        self.has_acted = False
        self.is_all_in = False
        # Model C Intra-Hand Sequence Memory
        self.pre_draw_action = -1     # Last pre-draw action index (-1 if not acted)
        self.post_draw_action = -1    # Last post-draw action index (-1 if not acted)
        self.pre_draw_bet_size = 0    # Cumulative chips committed pre-draw
        self.post_draw_bet_size = 0   # Cumulative chips committed post-draw

    def reset_for_hand(self):
        self.hand = []
        self.folded = False
        self.total_invested = 0
        self.round_invested = 0
        self.bet_to_call = 0
        self.draw_count = -1
        self.has_acted = False
        self.is_all_in = (self.chips == 0)
        self.pre_draw_action = -1
        self.post_draw_action = -1
        self.pre_draw_bet_size = 0
        self.post_draw_bet_size = 0


class MultiDrawPokerEnv:
    """Core table engine for 5-Seat 5-Card Draw Poker."""

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

        # Seats
        self.seats: List[PlayerState] = [
            PlayerState(i, starting_chips) for i in range(num_seats)
        ]

        self.button_seat = 0
        self.hands_played = 0
        self.phase = PHASE_PRE_DRAW
        self.pot = 0
        self.current_bet = 0  # Highest total bet level in current betting round
        self.min_raise_amount = self.big_blind

        # Turn management
        self.action_order: List[int] = []
        self.current_turn_idx = 0
        self.current_seat = 0

        # Discards pool for reshuffle safety
        self.discarded_cards_this_hand: List[int] = []

        # Session tracking
        self.session_done = False

    @property
    def total_table_chips(self) -> int:
        return sum(s.chips for s in self.seats) + self.pot

    @property
    def alive_seats(self) -> List[int]:
        return [s.seat_idx for s in self.seats if s.is_alive and s.chips > 0]

    @property
    def active_seats_in_hand(self) -> List[int]:
        return [s.seat_idx for s in self.seats if s.is_alive and not s.folded]

    def reset_session(
        self,
        starting_chips: Optional[int] = None,
        rng_seed: Optional[int] = None,
        randomize_stacks: bool = False,
    ):
        """Resets the entire tournament/session (all chips restored)."""
        if rng_seed is not None:
            self.rng = np.random.default_rng(rng_seed)
            self.deck = Deck(rng=self.rng)

        if starting_chips is not None:
            self.starting_chips = starting_chips

        if randomize_stacks:
            # Generate diverse initial stack depths (short, medium, deep) conserving total chips
            total_target = self.starting_chips * self.num_seats
            weights = self.rng.uniform(0.08, 0.40, size=self.num_seats)
            weights /= weights.sum()
            stacks = [int(w * total_target) for w in weights]
            diff = total_target - sum(stacks)
            stacks[-1] += diff
            for i, s in enumerate(self.seats):
                s.chips = max(self.base_big_blind * 5, stacks[i])
                s.is_alive = True
        else:
            for i, s in enumerate(self.seats):
                s.chips = self.starting_chips
                s.is_alive = True

        self.button_seat = int(self.rng.integers(0, self.num_seats))
        self.hands_played = 0
        self.session_done = False

        self._start_hand()

    def _next_alive_seat(self, from_seat: int) -> int:
        """Find next seat clockwise that is alive."""
        for step in range(1, self.num_seats + 1):
            s = (from_seat + step) % self.num_seats
            if self.seats[s].is_alive and self.seats[s].chips > 0:
                return s
        return from_seat

    def _start_hand(self):
        """Deals cards, posts blinds, sets up pre-draw betting."""
        self.hands_played += 1
        self.phase = PHASE_PRE_DRAW
        self.pot = 0
        self.discarded_cards_this_hand = []

        # Blind escalation (doubles every interval hands)
        if self.blind_escalation_interval and self.blind_escalation_interval > 0:
            level = (self.hands_played - 1) // self.blind_escalation_interval
            mult = 2 ** min(level, 7)  # Cap at 128x
            self.small_blind = self.base_small_blind * mult
            self.big_blind = self.base_big_blind * mult
        else:
            self.small_blind = self.base_small_blind
            self.big_blind = self.base_big_blind

        self.current_bet = 0
        self.min_raise_amount = self.big_blind

        alive = self.alive_seats
        if len(alive) <= 1:
            self.session_done = True
            return

        # Advance button to next alive seat
        self.button_seat = self._next_alive_seat(self.button_seat)

        # Reset seat hand states
        for s in self.seats:
            s.reset_for_hand()

        # Deal 5 cards to each alive player
        self.deck.reset()
        for seat_idx in alive:
            self.seats[seat_idx].hand = self.deck.deal(5)

        # Assign Blinds
        if len(alive) == 2:
            # Heads-up: Button posts SB, other posts BB
            sb_seat = self.button_seat
            bb_seat = self._next_alive_seat(sb_seat)
        else:
            sb_seat = self._next_alive_seat(self.button_seat)
            bb_seat = self._next_alive_seat(sb_seat)

        self._post_blind(sb_seat, self.small_blind)
        self._post_blind(bb_seat, self.big_blind)

        self.current_bet = self.big_blind
        self._update_bets_to_call()

        # Pre-draw turn order starts at seat after Big Blind (UTG)
        utg_seat = self._next_alive_seat(bb_seat)
        self._build_action_order(start_seat=utg_seat)

    def _post_blind(self, seat_idx: int, amount: int):
        player = self.seats[seat_idx]
        actual = min(player.chips, amount)
        player.chips -= actual
        player.total_invested += actual
        player.round_invested += actual
        self.pot += actual
        if player.chips == 0:
            player.is_all_in = True

    def _update_bets_to_call(self):
        for s in self.seats:
            if s.is_alive and not s.folded and not s.is_all_in:
                s.bet_to_call = max(0, self.current_bet - s.round_invested)
            else:
                s.bet_to_call = 0

    def _build_action_order(self, start_seat: int):
        """Builds action sequence starting from start_seat clockwise among alive players."""
        order = []
        for i in range(self.num_seats):
            seat = (start_seat + i) % self.num_seats
            p = self.seats[seat]
            if p.is_alive and not p.folded and not p.is_all_in:
                order.append(seat)

        self.action_order = order
        self.current_turn_idx = 0
        if self.action_order:
            self.current_seat = self.action_order[0]

    def get_legal_actions(self, seat_idx: int) -> List[int]:
        """Returns list of legal action integers for the given seat."""
        player = self.seats[seat_idx]
        if not player.is_alive or player.folded or player.is_all_in:
            return []

        if self.phase in (PHASE_PRE_DRAW, PHASE_POST_DRAW):
            legal = [A_FOLD, A_CALL]
            btc = player.bet_to_call

            # If can afford more than call, raises are possible
            chips_after_call = player.chips - btc
            if chips_after_call > 0:
                # All-In is always legal if you have chips beyond calling
                legal.append(A_ALL_IN)

                # Min-raise: raise by at least min_raise_amount
                min_total = btc + self.min_raise_amount
                if chips_after_call >= self.min_raise_amount:
                    legal.append(A_MIN_RAISE)

                # Half-Pot raise
                half_pot_raise = max(self.min_raise_amount, self.pot // 2)
                if chips_after_call >= half_pot_raise and half_pot_raise > self.min_raise_amount:
                    legal.append(A_HALF_POT)

                # Pot raise
                pot_raise = max(self.min_raise_amount, self.pot)
                if chips_after_call >= pot_raise and pot_raise > half_pot_raise:
                    legal.append(A_POT)

            return sorted(list(set(legal)))

        elif self.phase == PHASE_DRAW:
            # 32 draw bitmask actions
            return list(range(A_DRAW_START, TOTAL_ACTIONS))

        return []

    def get_action_mask(self, seat_idx: int) -> np.ndarray:
        """Returns binary mask of shape (38,) for sb3-contrib MaskablePPO."""
        mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
        legal = self.get_legal_actions(seat_idx)
        for a in legal:
            mask[a] = 1
        return mask

    def step_action(self, seat_idx: int, action: int):
        """Execute action taken by player at seat_idx."""
        player = self.seats[seat_idx]
        player.has_acted = True

        if self.phase in (PHASE_PRE_DRAW, PHASE_POST_DRAW):
            self._execute_bet_action(player, action)
        elif self.phase == PHASE_DRAW:
            self._execute_draw_action(player, action)

        self._advance_turn()

    def _execute_bet_action(self, player: PlayerState, action: int):
        legal = self.get_legal_actions(player.seat_idx)
        if action not in legal:
            # Fallback to check/call
            action = A_CALL if A_CALL in legal else legal[0]

        if action == A_FOLD:
            player.folded = True
            player.bet_to_call = 0
            if self.phase == PHASE_PRE_DRAW:
                player.pre_draw_action = A_FOLD
            elif self.phase == PHASE_POST_DRAW:
                player.post_draw_action = A_FOLD
            return

        btc = player.bet_to_call
        added_chips = 0
        is_raise = False

        if action == A_CALL:
            added_chips = min(player.chips, btc)

        elif action == A_MIN_RAISE:
            raise_size = self.min_raise_amount
            added_chips = min(player.chips, btc + raise_size)
            is_raise = True

        elif action == A_HALF_POT:
            raise_size = max(self.min_raise_amount, self.pot // 2)
            added_chips = min(player.chips, btc + raise_size)
            is_raise = True

        elif action == A_POT:
            raise_size = max(self.min_raise_amount, self.pot)
            added_chips = min(player.chips, btc + raise_size)
            is_raise = True

        elif action == A_ALL_IN:
            added_chips = player.chips
            if added_chips > btc:
                is_raise = True

        player.chips -= added_chips
        player.total_invested += added_chips
        player.round_invested += added_chips
        self.pot += added_chips

        if player.chips == 0:
            player.is_all_in = True

        if self.phase == PHASE_PRE_DRAW:
            player.pre_draw_action = action
            player.pre_draw_bet_size = player.round_invested
        elif self.phase == PHASE_POST_DRAW:
            player.post_draw_action = action
            player.post_draw_bet_size = player.round_invested

        if is_raise:
            new_bet = player.round_invested
            raise_diff = new_bet - self.current_bet
            if raise_diff > self.min_raise_amount:
                self.min_raise_amount = raise_diff
            self.current_bet = new_bet
            self._update_bets_to_call()

            # A raise re-opens action for all other active, non-all-in players
            self._reopen_action_order_after_raise(raiser_seat=player.seat_idx)

    def _reopen_action_order_after_raise(self, raiser_seat: int):
        order = []
        for i in range(1, self.num_seats):
            s = (raiser_seat + i) % self.num_seats
            p = self.seats[s]
            if p.is_alive and not p.folded and not p.is_all_in:
                order.append(s)

        self.action_order = order
        self.current_turn_idx = 0

    def _execute_draw_action(self, player: PlayerState, action: int):
        if action < A_DRAW_START or action >= TOTAL_ACTIONS:
            action = A_DRAW_START  # Stand pat fallback

        bitmask = action - A_DRAW_START
        discard_indices = [i for i in range(5) if bitmask & (1 << i)]
        num_discard = len(discard_indices)
        player.draw_count = num_discard

        if num_discard > 0:
            # Store discarded cards
            discarded_cards = [player.hand[i] for i in discard_indices]
            self.discarded_cards_this_hand.extend(discarded_cards)

            # Check deck stub availability; reshuffle previous discards if needed
            if self.deck.remaining < num_discard:
                # Pool available cards from previous discards (excluding current discard)
                reusable = [c for c in self.discarded_cards_this_hand if c not in discarded_cards]
                if reusable:
                    self.deck.cards.extend(reusable)
                    self.rng.shuffle(self.deck.cards)

            # Draw replacements
            drawn_cards = self.deck.draw(min(num_discard, self.deck.remaining))
            for idx, new_card in zip(sorted(discard_indices), drawn_cards):
                player.hand[idx] = new_card

    def _advance_turn(self):
        # 1. Check if only 1 player remains unfolded
        active = self.active_seats_in_hand
        if len(active) <= 1:
            self._award_uncontested_pot()
            return

        # 2. Check if betting round is complete
        if self.phase in (PHASE_PRE_DRAW, PHASE_POST_DRAW):
            # Advance index in current action order
            self.current_turn_idx += 1
            if self.current_turn_idx >= len(self.action_order):
                # Round finished
                self._transition_to_next_phase()
            else:
                self.current_seat = self.action_order[self.current_turn_idx]

        elif self.phase == PHASE_DRAW:
            self.current_turn_idx += 1
            if self.current_turn_idx >= len(self.action_order):
                self._transition_to_next_phase()
            else:
                self.current_seat = self.action_order[self.current_turn_idx]

    def _transition_to_next_phase(self):
        active = self.active_seats_in_hand
        if len(active) <= 1:
            self._award_uncontested_pot()
            return

        if self.phase == PHASE_PRE_DRAW:
            self.phase = PHASE_DRAW
            # Setup draw order: starts at first active seat to left of button
            start = self._next_alive_seat(self.button_seat)
            order = []
            for i in range(self.num_seats):
                s = (start + i) % self.num_seats
                p = self.seats[s]
                if p.is_alive and not p.folded:
                    order.append(s)
            self.action_order = order
            self.current_turn_idx = 0
            if self.action_order:
                self.current_seat = self.action_order[0]
            else:
                self._transition_to_next_phase()

        elif self.phase == PHASE_DRAW:
            self.phase = PHASE_POST_DRAW
            self.current_bet = 0
            self.min_raise_amount = self.big_blind
            for s in self.seats:
                s.round_invested = 0
                s.bet_to_call = 0
                s.has_acted = False

            # Post-draw betting starts at first active to left of button
            start = self._next_alive_seat(self.button_seat)
            self._build_action_order(start_seat=start)

            # If all or all-but-one players are all-in, skip straight to showdown
            can_bet = [s for s in self.action_order if not self.seats[s].is_all_in]
            if len(can_bet) <= 1:
                self._resolve_showdown()

        elif self.phase == PHASE_POST_DRAW:
            self._resolve_showdown()

    def _award_uncontested_pot(self):
        """When all opponents fold, the lone survivor takes the entire pot."""
        active = self.active_seats_in_hand
        if active:
            winner = active[0]
            self.seats[winner].chips += self.pot
            self.pot = 0

        self._check_eliminations_and_end_hand()

    def _resolve_showdown(self):
        """Multi-way showdown using side_pot.py."""
        self.phase = PHASE_SHOWDOWN
        active = self.active_seats_in_hand

        investments = {s.seat_idx: s.total_invested for s in self.seats}
        folded = {s.seat_idx for s in self.seats if s.folded}

        pots = calculate_side_pots(investments, folded)
        if not pots and self.pot > 0 and active:
            # Fallback single pot
            pots = [PotTier(self.pot, set(active))]

        # Score hands for active players
        scores = {}
        for s_idx in active:
            scores[s_idx] = evaluate_hand(self.seats[s_idx].hand)

        payouts, details = resolve_showdown_payouts(pots, scores)
        for s_idx, payout in payouts.items():
            self.seats[s_idx].chips += payout

        self.pot = 0
        self._check_eliminations_and_end_hand()

    def _check_eliminations_and_end_hand(self):
        """Prune busted players (chips == 0) and check session termination."""
        for s in self.seats:
            if s.chips <= 0:
                s.is_alive = False

        alive = self.alive_seats
        if len(alive) <= 1 or self.hands_played >= self.max_hands_per_session:
            self.session_done = True
        else:
            self._start_hand()
