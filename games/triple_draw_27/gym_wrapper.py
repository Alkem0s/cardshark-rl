"""
games/triple_draw_27/gym_wrapper.py — Gymnasium Multi-Player Vectorized Wrapper for 2-7 Triple Draw.

Observation Vector Architecture (95 Dimensions):
- [0..9]   Hero Cards (5 cards x 2 [rank_norm, suit_norm])
- [10..24] 3-Draw Tracking across 5 seats (5 seats x 3 draw counts / 5.0)
- [25..29] Seat Chip Ratios (chips / starting_chips / 5.0)
- [30..34] Active / Folded Indicators (1.0 = alive & active, 0.0 = folded/out)
- [35..38] Street Betting Indicators (Bet 1, Bet 2, Bet 3, Bet 4 one-hot)
- [39..42] Draw Round Indicators (Pre-Draw, Draw 1, Draw 2, Draw 3 one-hot)
- [43..47] Pot, Blinds, Hero Bet to Call, Round Current Bet, Hero Chips (normalized)
- [48..51] Position Features (Dealer distance, Is Button, Is Small Blind, Is Big Blind)
- [52..55] ICM Risk Ratios (Hero stack share, shortest stack ratio, etc.)
- [56..61] 6 Raw Game Geometry Features:
           - pot_odds
           - min_defense_frequency
           - aggressor_draw_count
           - aggressor_sizing_ratio
           - hero_lowball_percentile
           - facing_probe_bet
- [62..94] Multi-Street Sequence Memory across all 5 seats (33 features)
Total Observation Dimension: 95
"""

from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from core.card import rank_of, suit_of, normalize_rank, normalize_suit
from core.math_utils import pot_odds, minimum_defense_frequency
from games.triple_draw_27.rules import evaluate_27_hand, hand_category_27, is_wheel_27
from games.triple_draw_27.discard_masker import get_legal_discard_mask_27
from games.triple_draw_27.env import (
    TripleDrawPokerEnv,
    PlayerState27,
    A_FOLD,
    A_CALL,
    A_MIN_RAISE,
    A_HALF_POT,
    A_POT,
    A_ALL_IN,
    A_DRAW_START,
    TOTAL_ACTIONS,
    PHASE_BET_1,
    PHASE_DRAW_1,
    PHASE_BET_2,
    PHASE_DRAW_2,
    PHASE_BET_3,
    PHASE_DRAW_3,
    PHASE_BET_4,
    PHASE_SHOWDOWN,
)
from games.triple_draw_27.opponents import LowballOpponent, make_random_lowball_opponent

OBS_DIM_27 = 95


def build_observation_vector_27(
    env: TripleDrawPokerEnv,
    seat_idx: int = 0,
) -> np.ndarray:
    """Constructs the canonical 95-dimensional observation vector for 2-7 Triple Draw."""
    obs = np.zeros(OBS_DIM_27, dtype=np.float32)
    p = env.seats[seat_idx]
    total_table_chips = max(1, sum(s.chips for s in env.seats))

    # [0..9] Hero Cards (5 cards x 2)
    if p.hand and len(p.hand) == 5:
        for i, c in enumerate(p.hand):
            obs[i * 2] = normalize_rank(rank_of(c))
            obs[i * 2 + 1] = normalize_suit(suit_of(c))

    # [10..24] 3-Draw Tracking across 5 seats (5 seats x 3 rounds)
    idx = 10
    for s_i in range(5):
        s = env.seats[s_i]
        for round_idx in range(3):
            cnt = s.draw_counts[round_idx]
            obs[idx] = (cnt / 5.0) if cnt >= 0 else -1.0
            idx += 1

    # [25..29] Seat Chip Ratios
    for s_i in range(5):
        obs[25 + s_i] = float(env.seats[s_i].chips) / float(total_table_chips)

    # [30..34] Active / Folded Indicators
    for s_i in range(5):
        s = env.seats[s_i]
        obs[30 + s_i] = 1.0 if (s.is_alive and not s.folded) else 0.0

    # [35..38] Street Betting Indicators (Bet 1, Bet 2, Bet 3, Bet 4)
    if env.phase == PHASE_BET_1:
        obs[35] = 1.0
    elif env.phase == PHASE_BET_2:
        obs[36] = 1.0
    elif env.phase == PHASE_BET_3:
        obs[37] = 1.0
    elif env.phase == PHASE_BET_4:
        obs[38] = 1.0

    # [39..42] Draw Round Indicators
    if env.phase == PHASE_DRAW_1:
        obs[40] = 1.0
    elif env.phase == PHASE_DRAW_2:
        obs[41] = 1.0
    elif env.phase == PHASE_DRAW_3:
        obs[42] = 1.0
    else:
        obs[39] = 1.0  # Not in a draw round

    # [43..47] Pot, Blinds, Bet to Call, Current Bet, Hero Chips
    obs[43] = min(2.0, float(env.pot) / float(total_table_chips))
    obs[44] = float(env.big_blind) / 100.0
    obs[45] = min(1.0, float(p.bet_to_call) / max(1.0, float(p.chips)))
    obs[46] = min(1.0, float(env.current_bet) / float(total_table_chips))
    obs[47] = float(p.chips) / float(total_table_chips)

    # [48..51] Position Features
    dist_from_dealer = (seat_idx - env.dealer_seat) % 5
    obs[48] = float(dist_from_dealer) / 4.0
    obs[49] = 1.0 if dist_from_dealer == 0 else 0.0  # Button
    obs[50] = 1.0 if dist_from_dealer == 1 else 0.0  # SB
    obs[51] = 1.0 if dist_from_dealer == 2 else 0.0  # BB

    # [52..55] ICM Risk Ratios
    opp_chips = [env.seats[s_i].chips for s_i in range(5) if s_i != seat_idx and env.seats[s_i].is_alive]
    max_opp_chips = max(opp_chips) if opp_chips else 1
    min_opp_chips = min(opp_chips) if opp_chips else 1
    obs[52] = float(p.chips) / max(1.0, float(max_opp_chips))
    obs[53] = float(p.chips) / max(1.0, float(min_opp_chips))
    obs[54] = float(len(env.alive_seats)) / 5.0
    obs[55] = float(len(env.active_seats)) / 5.0

    # [56..61] 6 Raw Game Geometry Features
    # 1. Pot odds
    obs[56] = pot_odds(float(p.bet_to_call), float(env.pot))
    # 2. Minimum Defense Frequency
    obs[57] = minimum_defense_frequency(float(env.current_bet), float(env.pot))
    # 3. Aggressor Draw Count (draw count of most recent raiser)
    last_draw = 0.0
    for s_i in range(5):
        if s_i != seat_idx and env.seats[s_i].is_alive and not env.seats[s_i].folded:
            c = env.seats[s_i].draw_counts[0]
            if c >= 0:
                last_draw = float(c) / 5.0
    obs[58] = last_draw
    # 4. Aggressor Sizing Ratio
    obs[59] = min(1.0, float(env.current_bet) / max(1.0, float(p.chips)))
    # 5. Hero Lowball Hand Percentile [0.0, 1.0]
    if p.hand and len(p.hand) == 5:
        score = evaluate_27_hand(p.hand)
        # Lowball score ranges from ~0 to ~8_760_000
        obs[60] = min(1.0, max(0.0, float(score) / 8_760_000.0))
    # 6. Facing Probe Bet Indicator
    hero_drew_heavy = (p.draw_counts[0] >= 2 or p.draw_counts[1] >= 2)
    obs[61] = 1.0 if (hero_drew_heavy and p.bet_to_call > 0) else 0.0

    # [62..94] Action Memory across streets and seats (33 features)
    idx = 62
    for s_i in range(5):
        s = env.seats[s_i]
        for act in s.street_actions:
            if idx < OBS_DIM_27:
                obs[idx] = float(act) / 5.0 if act >= 0 else -1.0
                idx += 1

    return np.clip(obs, -1.0, 1.0)


class TripleDrawPokerGymEnv(gym.Env):
    """
    OpenAI Gymnasium Environment for 2-7 Triple Draw.
    Features:
      - 5-Seat freezeout simulation
      - Native discard action masking
      - Multi-agent heuristic league sparring
      - Discrete(38) action space
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_seats: int = 5,
        starting_chips: int = 200,
        small_blind: int = 1,
        big_blind: int = 2,
        hero_seat: int = 0,
        max_hands_per_session: int = 150,
        blind_escalation_interval: Optional[int] = 15,
        randomize_stacks: bool = True,
        fold_penalty: float = 0.05,
        discard_masking: bool = True,
        rng_seed: Optional[int] = None,
        league_pool: Optional[Any] = None,
    ):
        super().__init__()
        self.num_seats = num_seats
        self.starting_chips = starting_chips
        self.hero_seat = hero_seat
        self.randomize_stacks = randomize_stacks
        self.fold_penalty = fold_penalty
        self.discard_masking = discard_masking
        self.league_pool = league_pool
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)

        self.env = TripleDrawPokerEnv(
            num_seats=num_seats,
            starting_chips=starting_chips,
            small_blind=small_blind,
            big_blind=big_blind,
            rng_seed=rng_seed,
            max_hands_per_session=max_hands_per_session,
            blind_escalation_interval=blind_escalation_interval,
        )

        self.action_space = spaces.Discrete(TOTAL_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBS_DIM_27,),
            dtype=np.float32,
        )

        self.opponents: Dict[int, LowballOpponent] = {}
        self._setup_default_opponents()

    def _setup_default_opponents(self):
        for s in range(self.num_seats):
            if s != self.hero_seat:
                self.opponents[s] = make_random_lowball_opponent(opp_id=s, rng=self.rng)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.env.rng = self.rng
            self.env.deck.rng = self.rng

        self.env.reset_tournament()
        if self.randomize_stacks:
            # Randomize chip distribution within [100, 300] preserving 1,000 chip total
            noise = self.rng.integers(-50, 51, size=self.num_seats)
            noise -= int(np.mean(noise))
            for i, s in enumerate(self.env.seats):
                s.chips = max(20, self.starting_chips + int(noise[i]))

        self._setup_default_opponents()
        self._step_until_hero_turn()

        obs = build_observation_vector_27(self.env, self.hero_seat)
        return obs, self._get_info()

    def action_masks(self) -> np.ndarray:
        p = self.env.seats[self.hero_seat]
        if self.discard_masking and self.env.phase in (PHASE_DRAW_1, PHASE_DRAW_2, PHASE_DRAW_3):
            discard_mask = get_legal_discard_mask_27(p.hand)
            mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
            for bitmask_idx in range(len(discard_mask)):
                if discard_mask[bitmask_idx]:
                    mask[A_DRAW_START + bitmask_idx] = 1
            if not np.any(mask):
                mask[A_DRAW_START] = 1
            return mask
        return self.env.get_action_mask(self.hero_seat)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        p = self.env.seats[self.hero_seat]
        prev_chips = p.chips

        self.env.step_action(self.hero_seat, action)
        self._step_until_hero_turn()

        new_chips = p.chips
        chip_delta = new_chips - prev_chips

        # Reward formulation
        reward = float(chip_delta) / float(self.env.big_blind)
        if action == A_FOLD:
            reward -= self.fold_penalty

        terminated = (p.chips <= 0) or (self.env.session_done)
        truncated = False

        obs = build_observation_vector_27(self.env, self.hero_seat)
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _step_until_hero_turn(self):
        while not self.env.session_done:
            cur = self.env.current_seat
            if cur == self.hero_seat:
                p = self.env.seats[self.hero_seat]
                if p.is_alive and not p.folded:
                    if self.env.phase in (PHASE_DRAW_1, PHASE_DRAW_2, PHASE_DRAW_3):
                        return
                    elif not p.is_all_in:
                        return

            # Opponent acts or non-acting hero is bypassed
            opp_player = self.env.seats[cur]
            legal = self.env.get_legal_actions(cur)
            if not legal:
                if self.env.phase in (PHASE_DRAW_1, PHASE_DRAW_2, PHASE_DRAW_3):
                    self.env._advance_draw_queue()
                else:
                    self.env._advance_betting()
                continue

            bot = self.opponents.get(cur)
            if self.env.phase in (PHASE_BET_1, PHASE_BET_2, PHASE_BET_3, PHASE_BET_4):
                if bot:
                    act = bot.bet_action(
                        hand=opp_player.hand,
                        pot=self.env.pot,
                        bet_to_call=opp_player.bet_to_call,
                        phase=self.env.phase,
                        stack=opp_player.chips,
                        legal_actions=legal,
                    )
                else:
                    act = A_CALL if A_CALL in legal else legal[0]
            else:  # Draw phase
                if bot:
                    draw_round = 0 if self.env.phase == PHASE_DRAW_1 else (1 if self.env.phase == PHASE_DRAW_2 else 2)
                    discards = bot.draw_action(opp_player.hand, draw_round=draw_round)
                    bitmask = sum(1 << d for d in discards)
                    act = A_DRAW_START + bitmask
                else:
                    act = A_DRAW_START

            self.env.step_action(cur, act)

    def _get_info(self) -> dict:
        p = self.env.seats[self.hero_seat]
        chips = [s.chips for s in self.env.seats]
        leader = int(np.argmax(chips))
        is_winner = (p.chips > 0) and (
            (len(self.env.alive_seats) == 1 and self.env.alive_seats[0] == self.hero_seat)
            or (self.env.session_done and leader == self.hero_seat)
        )
        return {
            "hero_chips": p.chips,
            "hands_played": self.env.hands_played,
            "is_winner": is_winner,
            "pot": self.env.pot,
        }
