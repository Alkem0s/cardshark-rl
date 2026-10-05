"""
multi_gym_wrapper.py — Single-agent Gymnasium wrapper around MultiDrawPokerEnv for MaskablePPO.

Key Capabilities:
- Wraps 5-seat multi-player poker into a standard gym.Env interface for Hero.
- Scale-invariant 63-dimensional observation vector.
- Discrete(38) action space with hard action masking.
- Multi-hand session credit assignment: tournament survival rewards & bust penalties.
- Embeds a diverse pool of bot opponents and drives their turns automatically.
- Model B Bayesian Opponent Profiler updates automatically at each hand conclusion.
"""

from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

try:
    from game.card_utils import (
        normalize_rank, normalize_suit, normalize_hand_score,
        hand_category, evaluate_hand
    )
    from game.multi_draw_poker_env import (
        MultiDrawPokerEnv, TOTAL_ACTIONS,
        A_FOLD, A_CALL, A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN, A_DRAW_START,
        PHASE_PRE_DRAW, PHASE_DRAW, PHASE_POST_DRAW, PHASE_SHOWDOWN
    )
    from game.multi_opponents import MultiPlayerOpponent, make_random_archetype, make_opponent_by_id
    from rl.opponent_tracker import TableOpponentTracker
except ImportError:
    from card_utils import (
        normalize_rank, normalize_suit, normalize_hand_score,
        hand_category, evaluate_hand
    )
    from multi_draw_poker_env import (
        MultiDrawPokerEnv, TOTAL_ACTIONS,
        A_FOLD, A_CALL, A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN, A_DRAW_START,
        PHASE_PRE_DRAW, PHASE_DRAW, PHASE_POST_DRAW, PHASE_SHOWDOWN
    )
    from multi_opponents import MultiPlayerOpponent, make_random_archetype, make_opponent_by_id
    from opponent_tracker import TableOpponentTracker

from enum import IntEnum

class ModelObsDim(IntEnum):
    """Standardized observation dimensions keyed by model version designation."""
    MODEL_B = 63  # Baseline tabular/MLP representation
    MODEL_C = 87  # Model C sequence/action history representation
    MODEL_D = 87  # Model D attention representation (same feature schema as C)
    MODEL_E = 91  # Model E ICM & tournament blind dynamics representation
    MODEL_F = 97  # Model F bluff & range asymmetry representation (future)

# Global canonical dimension constants
OBS_DIM_B = ModelObsDim.MODEL_B.value
OBS_DIM_C = ModelObsDim.MODEL_C.value
OBS_DIM_D = ModelObsDim.MODEL_D.value
OBS_DIM_E = ModelObsDim.MODEL_E.value
OBS_DIM_F = ModelObsDim.MODEL_F.value

# Backward-compatibility aliases
OBS_DIM = OBS_DIM_B
SUPERHUMAN_OBS_DIM = OBS_DIM_C
MODEL_E_OBS_DIM = OBS_DIM_E


class MultiDrawPokerGymEnv(gym.Env):
    """Single-agent Gymnasium wrapper for 5-Seat Multi-Player 5-Card Draw Poker."""

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        num_seats: int = 5,
        starting_chips: int = 200,
        small_blind: int = 1,
        big_blind: int = 2,
        hero_seat: Optional[int] = None,  # None = randomized per session
        max_hands_per_session: int = 150,
        blind_escalation_interval: Optional[int] = None,
        randomize_stacks: bool = False,
        fold_penalty: float = 0.2,
        rng_seed: Optional[int] = None,
        fixed_opponent_archetypes: Optional[List[int]] = None,
        obs_dim: int | ModelObsDim = ModelObsDim.MODEL_B,
        superhuman_obs: Optional[bool] = None,
        model_e_obs: Optional[bool] = None,
        discard_masking: bool = False,
        league_pool: Optional[Any] = None,
    ):
        super().__init__()

        self.num_seats = num_seats
        self.starting_chips = starting_chips
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.fixed_hero_seat = hero_seat
        self.hero_seat = hero_seat if hero_seat is not None else 0
        self.max_hands_per_session = max_hands_per_session
        self.blind_escalation_interval = blind_escalation_interval
        self.randomize_stacks = randomize_stacks
        self.fold_penalty = fold_penalty
        self.fixed_opponent_archetypes = fixed_opponent_archetypes

        # Standardized observation dimension resolution
        if model_e_obs is True:
            self.obs_dim = int(ModelObsDim.MODEL_E)
        elif superhuman_obs is True:
            self.obs_dim = int(ModelObsDim.MODEL_C)
        else:
            self.obs_dim = int(obs_dim)

        # Retain backward-compatible attributes for external callers
        self.superhuman_obs = (self.obs_dim >= ModelObsDim.MODEL_C.value)
        self.model_e_obs = (self.obs_dim >= ModelObsDim.MODEL_E.value)
        self.discard_masking = discard_masking
        self.league_pool = league_pool

        self.rng = np.random.default_rng(rng_seed)

        # Core table environment
        self.env = MultiDrawPokerEnv(
            num_seats=num_seats,
            starting_chips=starting_chips,
            small_blind=small_blind,
            big_blind=big_blind,
            rng_seed=rng_seed,
            max_hands_per_session=max_hands_per_session,
            blind_escalation_interval=blind_escalation_interval,
        )

        # Model B Bayesian opponent tracker
        self.tracker = TableOpponentTracker(num_seats=num_seats)

        # Opponent bot instances
        self.opponents: Dict[int, MultiPlayerOpponent] = {}

        # Observation & action spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(TOTAL_ACTIONS)

        # Per-hand tracking for telemetry & Bayesian updates
        self._raised_pre_draw: Dict[int, bool] = {i: False for i in range(num_seats)}
        self._post_draw_raised: Dict[int, bool] = {i: False for i in range(num_seats)}
        self._faced_raise_and_folded: Dict[int, bool] = {i: False for i in range(num_seats)}

        # Session tracking
        self.prev_hero_chips = starting_chips
        self.session_hands = 0

    def action_masks(self) -> np.ndarray:
        """Returns binary mask for sb3-contrib MaskablePPO."""
        if self.env.session_done or not self.env.seats[self.hero_seat].is_alive:
            mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
            mask[A_CALL] = 1 # Fallback
            return mask
        if self.discard_masking and self.env.phase == PHASE_DRAW:
            from rl.discard_masker import get_legal_discard_mask
            hero_hand = self.env.seats[self.hero_seat].hand
            discard_mask = get_legal_discard_mask(hero_hand)
            mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
            for bitmask_idx in range(len(discard_mask)):
                if discard_mask[bitmask_idx]:
                    mask[A_DRAW_START + bitmask_idx] = 1
            if not np.any(mask):
                mask[A_DRAW_START] = 1
            return mask
        return self.env.get_action_mask(self.hero_seat)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Choose Hero seat
        if self.fixed_hero_seat is not None:
            self.hero_seat = self.fixed_hero_seat
        else:
            self.hero_seat = int(self.rng.integers(0, self.num_seats))

        # Reset environment & tracker
        self.env.reset_session(
            starting_chips=self.starting_chips,
            randomize_stacks=self.randomize_stacks,
        )
        self.tracker.reset_all()

        # Seed opponent bots
        self.opponents.clear()
        for s in range(self.num_seats):
            if s != self.hero_seat:
                if self.league_pool is not None:
                    self.opponents[s] = self.league_pool.sample_opponent(seat_idx=s, rng=self.rng)
                elif self.fixed_opponent_archetypes and len(self.fixed_opponent_archetypes) > s:
                    self.opponents[s] = make_opponent_by_id(self.fixed_opponent_archetypes[s], rng=self.rng)
                else:
                    self.opponents[s] = make_random_archetype(rng=self.rng)

        self._reset_hand_tracking()
        self.prev_hero_chips = self.env.seats[self.hero_seat].chips
        self.session_hands = 0

        # Run opponents until it is Hero's turn (or session ends)
        self._step_opponents_until_hero_turn()

        obs = self._build_observation()
        info = self._build_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        hero = self.env.seats[self.hero_seat]
        reward = 0.0

        # Unforced fold penalty check
        if hero.is_alive and not hero.folded:
            if action == A_FOLD and hero.bet_to_call == 0:
                reward -= self.fold_penalty

            # Execute Hero action
            self._record_action_telemetry(self.hero_seat, action)
            self.env.step_action(self.hero_seat, action)

        # Continue stepping opponents until Hero's turn again, or hand ends
        self._step_opponents_until_hero_turn()

        # Check if Hero's chips changed (per-hand profit/loss)
        current_hero_chips = hero.chips
        total_chips = max(1, self.env.total_table_chips)
        chip_delta = current_hero_chips - self.prev_hero_chips
        normalized_chip_reward = chip_delta / total_chips
        reward += normalized_chip_reward
        self.prev_hero_chips = current_hero_chips

        # Check termination
        terminated = False
        truncated = False

        if not hero.is_alive or hero.chips == 0:
            terminated = True
            reward -= 1.0  # Elimination penalty
        elif len(self.env.alive_seats) == 1 and self.env.alive_seats[0] == self.hero_seat:
            terminated = True
            reward += 2.0  # Tournament victory bonus (1st place elimination)
        elif self.env.session_done or self.env.hands_played >= self.max_hands_per_session:
            truncated = True

        obs = self._build_observation()
        info = self._build_info()
        return obs, reward, terminated, truncated, info

    def _step_opponents_until_hero_turn(self):
        """Drives the table state by stepping opponent bots until Hero's turn or session ends."""
        prev_hand_idx = self.env.hands_played

        while not self.env.session_done:
            # Check if hand changed
            if self.env.hands_played != prev_hand_idx:
                # Previous hand finished: update Bayesian tracker
                self._record_hand_to_tracker()
                self._reset_hand_tracking()
                prev_hand_idx = self.env.hands_played

            # If Hero is dead, finish the session or return
            if not self.env.seats[self.hero_seat].is_alive:
                break

            # If it's Hero's turn, stop and let RL agent decide
            if self.env.current_seat == self.hero_seat and not self.env.seats[self.hero_seat].folded:
                legal = self.env.get_legal_actions(self.hero_seat)
                if legal:
                    break  # Wait for Hero's action

            # It's an opponent's turn: decide and step
            opp_seat = self.env.current_seat
            opp_player = self.env.seats[opp_seat]

            if opp_player.is_alive and not opp_player.folded and not opp_player.is_all_in:
                bot = self.opponents.get(opp_seat)
                legal = self.env.get_legal_actions(opp_seat)

                if legal:
                    if self.env.phase in (PHASE_PRE_DRAW, PHASE_POST_DRAW):
                        if bot:
                            if hasattr(bot, "act_with_env"):
                                action = bot.act_with_env(self.env, opp_seat, legal)
                            else:
                                action = bot.bet_action(
                                    hand=opp_player.hand,
                                    pot=self.env.pot,
                                    bet_to_call=opp_player.bet_to_call,
                                    phase=self.env.phase,
                                    stack=opp_player.chips,
                                    legal_actions=legal,
                                )
                        else:
                            action = A_CALL if A_CALL in legal else legal[0]
                    else: # Draw phase
                        if bot:
                            if hasattr(bot, "draw_with_env"):
                                action = bot.draw_with_env(self.env, opp_seat)
                            else:
                                discards = bot.draw_action(opp_player.hand)
                                bitmask = sum((1 << i) for i in discards)
                                action = A_DRAW_START + bitmask
                        else:
                            action = A_DRAW_START

                    self._record_action_telemetry(opp_seat, action)
                    self.env.step_action(opp_seat, action)
                else:
                    self.env.step_action(opp_seat, A_CALL)
            else:
                self.env.step_action(opp_seat, A_CALL)

        if self.env.hands_played != prev_hand_idx:
            self._record_hand_to_tracker()
            self._reset_hand_tracking()

    def step_table_until_winner(self):
        """Continues stepping the tournament among remaining alive bots until only 1 remains or max hands hit."""
        prev_hand_idx = self.env.hands_played

        while not self.env.session_done and len(self.env.alive_seats) > 1 and self.env.hands_played < self.max_hands_per_session:
            if self.env.hands_played != prev_hand_idx:
                self._record_hand_to_tracker()
                self._reset_hand_tracking()
                prev_hand_idx = self.env.hands_played

            opp_seat = self.env.current_seat
            opp_player = self.env.seats[opp_seat]

            if opp_player.is_alive and not opp_player.folded and not opp_player.is_all_in:
                bot = self.opponents.get(opp_seat)
                legal = self.env.get_legal_actions(opp_seat)

                if legal:
                    if self.env.phase in (PHASE_PRE_DRAW, PHASE_POST_DRAW):
                        if bot:
                            if hasattr(bot, "act_with_env"):
                                action = bot.act_with_env(self.env, opp_seat, legal)
                            else:
                                action = bot.bet_action(
                                    hand=opp_player.hand,
                                    pot=self.env.pot,
                                    bet_to_call=opp_player.bet_to_call,
                                    phase=self.env.phase,
                                    stack=opp_player.chips,
                                    legal_actions=legal,
                                )
                        else:
                            action = A_CALL if A_CALL in legal else legal[0]
                    else: # Draw phase
                        if bot:
                            if hasattr(bot, "draw_with_env"):
                                action = bot.draw_with_env(self.env, opp_seat)
                            else:
                                discards = bot.draw_action(opp_player.hand)
                                bitmask = sum((1 << i) for i in discards)
                                action = A_DRAW_START + bitmask
                        else:
                            action = A_DRAW_START

                    self._record_action_telemetry(opp_seat, action)
                    self.env.step_action(opp_seat, action)
                else:
                    self.env.step_action(opp_seat, A_CALL)
            else:
                self.env.step_action(opp_seat, A_CALL)

        if self.env.hands_played != prev_hand_idx:
            self._record_hand_to_tracker()
            self._reset_hand_tracking()

    def _record_action_telemetry(self, seat_idx: int, action: int):
        player = self.env.seats[seat_idx]
        if self.env.phase == PHASE_PRE_DRAW:
            if action in (A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN):
                self._raised_pre_draw[seat_idx] = True
        elif self.env.phase == PHASE_POST_DRAW:
            if action in (A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN):
                self._post_draw_raised[seat_idx] = True
        if action == A_FOLD and player.bet_to_call > 0:
            self._faced_raise_and_folded[seat_idx] = True

    def _record_hand_to_tracker(self):
        """Update Beta-Binomial models for each seat after a hand finishes."""
        for s in range(self.num_seats):
            player = self.env.seats[s]
            vpip = player.total_invested > (self.big_blind if s == self.env.button_seat else 0)
            pfr = self._raised_pre_draw.get(s, False)
            post_draw_raised = self._post_draw_raised.get(s, False)
            faced_raise_and_folded = self._faced_raise_and_folded.get(s, False)
            cards_drawn = player.draw_count if player.draw_count >= 0 else None

            self.tracker.record_seat_hand(
                seat_idx=s,
                vpip=vpip,
                pfr=pfr,
                post_draw_raised=post_draw_raised,
                faced_raise_and_folded=faced_raise_and_folded,
                cards_drawn=cards_drawn,
            )

    def _reset_hand_tracking(self):
        self._raised_pre_draw = {i: False for i in range(self.num_seats)}
        self._post_draw_raised = {i: False for i in range(self.num_seats)}
        self._faced_raise_and_folded = {i: False for i in range(self.num_seats)}

    def _build_observation(self) -> np.ndarray:
        return build_observation_vector(
            env=self.env,
            tracker=self.tracker,
            seat_idx=self.hero_seat,
            obs_dim=self.obs_dim,
        )

    def get_privileged_belief(self) -> Tuple[int, float]:
        """
        Returns privileged ground-truth table belief for auxiliary representation learning:
        - target_cat: int (0..8) hand category of primary active opponent
        - target_bluff: float (0.0 or 1.0) whether primary active opponent is betting with air (cat == 0)
        """
        primary_opp = -1
        max_round_bet = 0
        for s in range(self.num_seats):
            if s != self.hero_seat and self.env.seats[s].is_alive and not self.env.seats[s].folded:
                if self.env.seats[s].round_invested > max_round_bet:
                    max_round_bet = self.env.seats[s].round_invested
                    primary_opp = s

        if primary_opp == -1:
            active = [s for s in range(self.num_seats) if s != self.hero_seat and self.env.seats[s].is_alive and not self.env.seats[s].folded]
            if active:
                primary_opp = active[0]

        if primary_opp != -1 and self.env.seats[primary_opp].hand and len(self.env.seats[primary_opp].hand) == 5:
            cat = int(hand_category(self.env.seats[primary_opp].hand))
            is_bluff = 1.0 if (cat == 0 and max_round_bet > 0) else 0.0
            return cat, is_bluff

        return 0, 0.0

    def _build_info(self) -> dict:
        is_winner = bool(len(self.env.alive_seats) == 1 and self.env.alive_seats[0] == self.hero_seat)
        cat, is_bluff = self.get_privileged_belief()
        return {
            "hero_seat": self.hero_seat,
            "hero_chips": self.env.seats[self.hero_seat].chips,
            "pot": self.env.pot,
            "hands_played": self.env.hands_played,
            "alive_players": len(self.env.alive_seats),
            "button_seat": self.env.button_seat,
            "is_winner": is_winner,
            "opp_hand_cat": cat,
            "is_bluff": is_bluff,
        }


def build_observation_vector(
    env: MultiDrawPokerEnv,
    tracker: TableOpponentTracker,
    seat_idx: int,
    obs_dim: int | ModelObsDim = ModelObsDim.MODEL_B,
    superhuman_obs: Optional[bool] = None,
    model_e_obs: Optional[bool] = None,
) -> np.ndarray:
    """
    Constructs scale-invariant state vector for the requested model tier:
    - Model B: 63 dims (Hand tokens, scoring, stack/pot, position, survival, opponent slots)
    - Model C / D: 87 dims (+ 24 sequence memory & recency tilt features)
    - Model E: 91 dims (+ 4 tournament ICM dynamics: Stack/BB, Pot/BB, Blind pressure, Clock)
    """
    # Standardize target dimension
    if model_e_obs is True:
        target_dim = ModelObsDim.MODEL_E.value
    elif superhuman_obs is True:
        target_dim = ModelObsDim.MODEL_C.value
    elif superhuman_obs is False and model_e_obs is False and obs_dim == ModelObsDim.MODEL_B:
        target_dim = ModelObsDim.MODEL_B.value
    else:
        target_dim = int(obs_dim)

    hero = env.seats[seat_idx]
    total_chips = max(1.0, float(env.total_table_chips))
    pot = float(env.pot)

    # 1. Hero Hand Encoding (10)
    hand_features = []
    if hero.hand and len(hero.hand) == 5:
        for c in hero.hand:
            hand_features.extend([normalize_rank(c), normalize_suit(c)])
    else:
        hand_features = [0.0] * 10

    # 2. Hero Hand Strength (2)
    if hero.hand and len(hero.hand) == 5:
        cat_norm = float(hand_category(hero.hand)) / 8.0
        score_norm = normalize_hand_score(evaluate_hand(hero.hand))
    else:
        cat_norm = 0.0
        score_norm = 0.0

    # 3. Table Pot & Stack State (5)
    hero_stack_ratio = float(hero.chips) / total_chips
    pot_ratio = pot / total_chips
    btc = float(hero.bet_to_call)
    pot_odds = btc / max(1.0, pot + btc)
    spr = min(float(hero.chips) / max(1.0, pot), 10.0) / 10.0
    invested_ratio = float(hero.total_invested) / max(1.0, pot)

    # 4. Game State & Position (4)
    phase_vec = [0.0, 0.0, 0.0]
    if env.phase == PHASE_PRE_DRAW:
        phase_vec[0] = 1.0
    elif env.phase == PHASE_DRAW:
        phase_vec[1] = 1.0
    elif env.phase == PHASE_POST_DRAW:
        phase_vec[2] = 1.0

    btn_dist = float((seat_idx - env.button_seat) % env.num_seats) / max(1.0, env.num_seats - 1.0)

    # 5. Table Survival (2)
    alive_count = len(env.alive_seats)
    active_in_hand_count = len(env.active_seats_in_hand)
    alive_ratio = float(alive_count) / float(env.num_seats)
    in_hand_ratio = float(active_in_hand_count) / max(1.0, float(alive_count))

    # 6. Opponent Slots (40: 4 slots x 10 features)
    is_alive = [s.is_alive for s in env.seats]
    in_hand = [not s.folded for s in env.seats]
    stacks = [s.chips for s in env.seats]
    investments = [s.total_invested for s in env.seats]
    draw_counts = [s.draw_count for s in env.seats]

    opp_vector = tracker.get_relative_opponent_vector(
        hero_seat=seat_idx,
        is_alive=is_alive,
        in_hand=in_hand,
        stacks=stacks,
        investments=investments,
        pot=int(pot),
        draw_counts_this_hand=draw_counts,
    )

    obs = (
        hand_features +
        [cat_norm, score_norm] +
        [hero_stack_ratio, pot_ratio, pot_odds, spr, invested_ratio] +
        phase_vec + [btn_dist] +
        [alive_ratio, in_hand_ratio] +
        opp_vector
    )

    if target_dim == ModelObsDim.MODEL_B.value:
        assert len(obs) == OBS_DIM_B, f"Model B observation dimension mismatch: {len(obs)} != {OBS_DIM_B}"
        return np.array(obs, dtype=np.float32)

    # 7. Model C/D Action Sequence & Dynamic Recency Memory (24: 4 slots x 6 features)
    act_map = {
        -1: 0.0,
        A_FOLD: -1.0,
        A_CALL: 0.2,
        A_MIN_RAISE: 0.5,
        A_HALF_POT: 0.7,
        A_POT: 0.9,
        A_ALL_IN: 1.0,
    }

    sequence_features: List[float] = []
    for offset in range(1, env.num_seats):
        s = (seat_idx + offset) % env.num_seats
        p = env.seats[s]

        if p.is_alive:
            # 1. Pre-Draw Action History (2)
            pre_act = act_map.get(p.pre_draw_action, 0.0)
            pre_sizing = float(np.clip(p.pre_draw_bet_size / total_chips, 0.0, 1.0))

            # 2. Post-Draw Action History (2)
            post_act = act_map.get(p.post_draw_action, 0.0)
            post_sizing = float(np.clip(p.post_draw_bet_size / total_chips, 0.0, 1.0))

            # 3. Intra-Hand Line Combination & Recency Tilt (2)
            line_code = 0.0
            if p.folded:
                line_code = -1.0
            elif p.post_draw_action in (A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN):
                if p.draw_count == 0:
                    line_code = 0.8  # Pat value bet
                elif p.pre_draw_action in (-1, A_CALL) and p.draw_count in (1, 2):
                    line_code = 1.0  # Trap / check-raise line
                elif p.pre_draw_action in (A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN) and p.draw_count >= 2:
                    line_code = -0.5 # Aggressive multi-card draw bluff
                else:
                    line_code = 0.6  # Standard post-draw aggression
            elif p.post_draw_action == A_CALL:
                line_code = 0.1      # Showdown bound call

            tilt_delta = tracker.get_seat_recency_delta(s)

            slot_seq = [pre_act, pre_sizing, post_act, post_sizing, line_code, tilt_delta]
        else:
            slot_seq = [0.0] * 6

        sequence_features.extend(slot_seq)

    obs = obs + sequence_features

    if target_dim in (ModelObsDim.MODEL_C.value, ModelObsDim.MODEL_D.value):
        assert len(obs) == OBS_DIM_C, f"Model C/D observation dimension mismatch: {len(obs)} != {OBS_DIM_C}"
        return np.array(obs, dtype=np.float32)

    # 8. Model E Tournament ICM Dynamics (4 features)
    bb = max(1.0, float(env.big_blind))
    # Effective stack in BB (capped at 100 BB, normalized [-1.0, 1.0])
    stack_bb_norm = float(np.clip((hero.chips / bb) / 50.0, 0.0, 2.0)) - 1.0
    # Pot in BB (capped at 50 BB, normalized [-1.0, 1.0])
    pot_bb_norm = float(np.clip((pot / bb) / 25.0, 0.0, 2.0)) - 1.0
    # Blind pressure: current BB relative to all table chips
    bb_pressure_norm = float(np.clip((bb / total_chips) * 10.0, 0.0, 2.0)) - 1.0
    # Blind escalation clock: countdown to next blind raise
    if env.blind_escalation_interval and env.blind_escalation_interval > 0:
        hands_left = env.blind_escalation_interval - (env.hands_played % env.blind_escalation_interval)
        escalation_clock = (float(hands_left) / float(env.blind_escalation_interval)) * 2.0 - 1.0
    else:
        escalation_clock = 0.0
    icm_features = [stack_bb_norm, pot_bb_norm, bb_pressure_norm, escalation_clock]
    obs = obs + icm_features

    if target_dim == ModelObsDim.MODEL_E.value:
        assert len(obs) == OBS_DIM_E, f"Model E observation dimension mismatch: {len(obs)} != {OBS_DIM_E}"
        return np.array(obs, dtype=np.float32)

    # 9. Model F Game-Theoretic Geometry & Information Features (6 features)
    pot_odds_raw = btc / max(1.0, pot + btc)
    pot_odds_norm = float(np.clip(pot_odds_raw * 2.0 - 1.0, -1.0, 1.0))

    mdf_raw = pot / max(1.0, pot + btc)
    mdf_norm = float(np.clip(mdf_raw * 2.0 - 1.0, -1.0, 1.0))

    # Aggressor tracking: identify opponent who opened or raised
    aggressor_seat = -1
    max_invest = 0
    for s_idx in range(env.num_seats):
        if s_idx != seat_idx and env.seats[s_idx].is_alive and not env.seats[s_idx].folded:
            if env.seats[s_idx].round_invested > max_invest:
                max_invest = env.seats[s_idx].round_invested
                aggressor_seat = s_idx

    if aggressor_seat != -1 and env.seats[aggressor_seat].draw_count >= 0:
        aggressor_draw_norm = float(np.clip((env.seats[aggressor_seat].draw_count / 5.0) * 2.0 - 1.0, -1.0, 1.0))
        pre_bet = max(1.0, float(env.seats[aggressor_seat].pre_draw_bet_size))
        post_bet = float(env.seats[aggressor_seat].post_draw_bet_size)
        escalation_norm = float(np.clip((post_bet / pre_bet) - 1.0, -1.0, 1.0))
    else:
        aggressor_draw_norm = 0.0
        escalation_norm = 0.0

    if hero.hand and len(hero.hand) == 5:
        hand_pctl_norm = float(np.clip((hand_category(hero.hand) / 8.0) * 2.0 - 1.0, -1.0, 1.0))
    else:
        hand_pctl_norm = -1.0

    facing_probe = 0.0
    if env.phase == PHASE_POST_DRAW and btc > 0:
        if hero.post_draw_action == A_CALL and hero.round_invested == 0:
            facing_probe = 1.0
        elif hero.post_draw_action == -1 and env.current_bet > 0:
            facing_probe = 0.5
    facing_probe_norm = float(facing_probe * 2.0 - 1.0)

    model_f_features = [
        pot_odds_norm,
        mdf_norm,
        aggressor_draw_norm,
        escalation_norm,
        hand_pctl_norm,
        facing_probe_norm,
    ]
    obs = obs + model_f_features

    if target_dim == ModelObsDim.MODEL_F.value:
        assert len(obs) == OBS_DIM_F, f"Model F observation dimension mismatch: {len(obs)} != {OBS_DIM_F}"
        return np.array(obs, dtype=np.float32)

    assert len(obs) == target_dim, f"Observation dimension mismatch: {len(obs)} != {target_dim}"
    return np.array(obs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Factory functions for VecEnv creation
# ---------------------------------------------------------------------------

def make_multi_env(
    num_seats: int = 5,
    starting_chips: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    hero_seat: Optional[int] = None,
    max_hands_per_session: int = 150,
    blind_escalation_interval: Optional[int] = None,
    randomize_stacks: bool = False,
    fold_penalty: float = 0.2,
    seed: int = 0,
    fixed_opponent_archetypes: Optional[List[int]] = None,
    obs_dim: int | ModelObsDim = ModelObsDim.MODEL_B,
    superhuman_obs: Optional[bool] = None,
    model_e_obs: Optional[bool] = None,
    discard_masking: bool = False,
    league_pool: Optional[Any] = None,
):
    """Creates a closure that returns a MultiDrawPokerGymEnv."""
    def _init():
        return MultiDrawPokerGymEnv(
            num_seats=num_seats,
            starting_chips=starting_chips,
            small_blind=small_blind,
            big_blind=big_blind,
            hero_seat=hero_seat,
            max_hands_per_session=max_hands_per_session,
            blind_escalation_interval=blind_escalation_interval,
            randomize_stacks=randomize_stacks,
            fold_penalty=fold_penalty,
            rng_seed=seed,
            fixed_opponent_archetypes=fixed_opponent_archetypes,
            obs_dim=obs_dim,
            superhuman_obs=superhuman_obs,
            model_e_obs=model_e_obs,
            discard_masking=discard_masking,
            league_pool=league_pool,
        )
    return _init


def make_model_b_multi_env(
    num_seats: int = 5,
    starting_chips: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    hero_seat: Optional[int] = None,
    max_hands_per_session: int = 150,
    blind_escalation_interval: Optional[int] = None,
    randomize_stacks: bool = False,
    fold_penalty: float = 0.2,
    seed: int = 0,
    fixed_opponent_archetypes: Optional[List[int]] = None,
    league_pool: Optional[Any] = None,
):
    """Creates a closure that returns a Model B MultiDrawPokerGymEnv (63 dims)."""
    return make_multi_env(
        num_seats=num_seats,
        starting_chips=starting_chips,
        small_blind=small_blind,
        big_blind=big_blind,
        hero_seat=hero_seat,
        max_hands_per_session=max_hands_per_session,
        blind_escalation_interval=blind_escalation_interval,
        randomize_stacks=randomize_stacks,
        fold_penalty=fold_penalty,
        seed=seed,
        fixed_opponent_archetypes=fixed_opponent_archetypes,
        obs_dim=ModelObsDim.MODEL_B,
        discard_masking=False,
        league_pool=league_pool,
    )


def make_model_c_multi_env(
    num_seats: int = 5,
    starting_chips: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    hero_seat: Optional[int] = None,
    max_hands_per_session: int = 150,
    blind_escalation_interval: Optional[int] = None,
    randomize_stacks: bool = False,
    fold_penalty: float = 0.2,
    seed: int = 0,
    fixed_opponent_archetypes: Optional[List[int]] = None,
    league_pool: Optional[Any] = None,
):
    """Creates a closure that returns a Model C MultiDrawPokerGymEnv (87 dims)."""
    return make_multi_env(
        num_seats=num_seats,
        starting_chips=starting_chips,
        small_blind=small_blind,
        big_blind=big_blind,
        hero_seat=hero_seat,
        max_hands_per_session=max_hands_per_session,
        blind_escalation_interval=blind_escalation_interval,
        randomize_stacks=randomize_stacks,
        fold_penalty=fold_penalty,
        seed=seed,
        fixed_opponent_archetypes=fixed_opponent_archetypes,
        obs_dim=ModelObsDim.MODEL_C,
        discard_masking=False,
        league_pool=league_pool,
    )


def make_model_d_multi_env(
    num_seats: int = 5,
    starting_chips: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    hero_seat: Optional[int] = None,
    max_hands_per_session: int = 150,
    blind_escalation_interval: Optional[int] = None,
    randomize_stacks: bool = False,
    fold_penalty: float = 0.2,
    seed: int = 0,
    fixed_opponent_archetypes: Optional[List[int]] = None,
    league_pool: Optional[Any] = None,
):
    """Creates a closure that returns a Model D MultiDrawPokerGymEnv (87 dims)."""
    return make_multi_env(
        num_seats=num_seats,
        starting_chips=starting_chips,
        small_blind=small_blind,
        big_blind=big_blind,
        hero_seat=hero_seat,
        max_hands_per_session=max_hands_per_session,
        blind_escalation_interval=blind_escalation_interval,
        randomize_stacks=randomize_stacks,
        fold_penalty=fold_penalty,
        seed=seed,
        fixed_opponent_archetypes=fixed_opponent_archetypes,
        obs_dim=ModelObsDim.MODEL_D,
        discard_masking=False,
        league_pool=league_pool,
    )


def make_model_e_multi_env(
    num_seats: int = 5,
    starting_chips: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    hero_seat: Optional[int] = None,
    max_hands_per_session: int = 150,
    blind_escalation_interval: Optional[int] = None,
    randomize_stacks: bool = False,
    fold_penalty: float = 0.075,
    seed: int = 0,
    fixed_opponent_archetypes: Optional[List[int]] = None,
    league_pool: Optional[Any] = None,
):
    """Creates a closure that returns a Model E MultiDrawPokerGymEnv (91 dims + discard masking)."""
    return make_multi_env(
        num_seats=num_seats,
        starting_chips=starting_chips,
        small_blind=small_blind,
        big_blind=big_blind,
        hero_seat=hero_seat,
        max_hands_per_session=max_hands_per_session,
        blind_escalation_interval=blind_escalation_interval,
        randomize_stacks=randomize_stacks,
        fold_penalty=fold_penalty,
        seed=seed,
        fixed_opponent_archetypes=fixed_opponent_archetypes,
        obs_dim=ModelObsDim.MODEL_E,
        discard_masking=True,
        league_pool=league_pool,
    )


def make_model_f_multi_env(
    num_seats: int = 5,
    starting_chips: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    hero_seat: Optional[int] = None,
    max_hands_per_session: int = 150,
    blind_escalation_interval: Optional[int] = None,
    randomize_stacks: bool = False,
    fold_penalty: float = 0.075,
    seed: int = 0,
    fixed_opponent_archetypes: Optional[List[int]] = None,
    league_pool: Optional[Any] = None,
):
    """Creates a closure that returns a Model F MultiDrawPokerGymEnv (97 dims + discard masking)."""
    return make_multi_env(
        num_seats=num_seats,
        starting_chips=starting_chips,
        small_blind=small_blind,
        big_blind=big_blind,
        hero_seat=hero_seat,
        max_hands_per_session=max_hands_per_session,
        blind_escalation_interval=blind_escalation_interval,
        randomize_stacks=randomize_stacks,
        fold_penalty=fold_penalty,
        seed=seed,
        fixed_opponent_archetypes=fixed_opponent_archetypes,
        obs_dim=ModelObsDim.MODEL_F,
        discard_masking=True,
        league_pool=league_pool,
    )


# Backward-compatible alias
make_superhuman_multi_env = make_model_c_multi_env


def multi_mask_fn(env) -> np.ndarray:
    """Mask function for sb3-contrib ActionMasker, unwrapping layers if needed."""
    if hasattr(env, "action_masks"):
        return env.action_masks()
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "action_masks"):
        return env.unwrapped.action_masks()
    raise AttributeError(f"Environment {env} has no action_masks method")


