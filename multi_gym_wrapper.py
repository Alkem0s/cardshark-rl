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

OBS_DIM = 63
SUPERHUMAN_OBS_DIM = 87


class MultiDrawPokerGymEnv(gym.Env):
    """Single-agent Gymnasium wrapper for 5-Seat Multi-Player 5-Card Draw Poker."""

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        num_seats: int = 5,
        starting_chips: int = 200,
        small_blind: int = 1,
        big_blind: int = 2,
        hero_seat: Optional[int] = None, # None = randomized per session
        max_hands_per_session: int = 150,
        blind_escalation_interval: Optional[int] = None,
        randomize_stacks: bool = False,
        fold_penalty: float = 0.2,
        rng_seed: Optional[int] = None,
        fixed_opponent_archetypes: Optional[List[int]] = None,
        superhuman_obs: bool = False,
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
        self.superhuman_obs = superhuman_obs
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
        obs_dim = SUPERHUMAN_OBS_DIM if superhuman_obs else OBS_DIM
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
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
            superhuman_obs=self.superhuman_obs,
        )

    def _build_info(self) -> dict:
        is_winner = bool(len(self.env.alive_seats) == 1 and self.env.alive_seats[0] == self.hero_seat)
        return {
            "hero_seat": self.hero_seat,
            "hero_chips": self.env.seats[self.hero_seat].chips,
            "pot": self.env.pot,
            "hands_played": self.env.hands_played,
            "alive_players": len(self.env.alive_seats),
            "button_seat": self.env.button_seat,
            "is_winner": is_winner,
        }


def build_observation_vector(
    env: MultiDrawPokerEnv,
    tracker: TableOpponentTracker,
    seat_idx: int,
    superhuman_obs: bool = False,
) -> np.ndarray:
    """Constructs scale-invariant state vector (63 dims for Model B, 87 dims for Model C Superhuman)."""
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

    if not superhuman_obs:
        assert len(obs) == OBS_DIM, f"Observation dimension mismatch: {len(obs)} != {OBS_DIM}"
        return np.array(obs, dtype=np.float32)

    # 7. Model C Superhuman Sequence & Dynamic Recency Memory (24: 4 slots x 6 features)
    act_map = {
        -1: 0.0,
        A_FOLD: -1.0,
        A_CALL: 0.2,
        A_MIN_RAISE: 0.5,
        A_HALF_POT: 0.7,
        A_POT: 0.9,
        A_ALL_IN: 1.0,
    }

    superhuman_features: List[float] = []
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

        superhuman_features.extend(slot_seq)

    obs = obs + superhuman_features
    assert len(obs) == SUPERHUMAN_OBS_DIM, f"Superhuman observation dimension mismatch: {len(obs)} != {SUPERHUMAN_OBS_DIM}"
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
    superhuman_obs: bool = False,
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
            superhuman_obs=superhuman_obs,
            league_pool=league_pool,
        )
    return _init


def make_superhuman_multi_env(
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
    """Creates a closure that returns a Superhuman MultiDrawPokerGymEnv (87 dims)."""
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
        superhuman_obs=True,
        league_pool=league_pool,
    )


def multi_mask_fn(env) -> np.ndarray:
    """Mask function for sb3-contrib ActionMasker, unwrapping layers if needed."""
    if hasattr(env, "action_masks"):
        return env.action_masks()
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "action_masks"):
        return env.unwrapped.action_masks()
    raise AttributeError(f"Environment {env} has no action_masks method")

