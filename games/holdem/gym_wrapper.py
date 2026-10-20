"""
games/holdem/gym_wrapper.py — Gymnasium Wrapper for No-Limit Texas Hold'em.

Features:
1. 73-dimensional bounded [-1.0, 1.0] observation vector.
2. 6-action discrete betting space (Fold, Check/Call, Min-Raise, Half-Pot, Pot, All-In).
3. ActionMasker integration for valid betting options.
4. Autonomous table stepping for heuristic opponents.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from core.card import rank_of, suit_of, normalize_rank, normalize_suit
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
from games.holdem.evaluator import evaluate_holdem_hand
from games.holdem.env import HoldemPokerEnv
from games.holdem.opponents import HoldemOpponent, make_random_holdem_opponent

OBS_DIM_HOLDEM = 73


def build_observation_vector_holdem(env: HoldemPokerEnv, hero_seat: int) -> np.ndarray:
    obs = np.zeros(OBS_DIM_HOLDEM, dtype=np.float32)
    idx = 0
    total_chips = max(1, env.num_seats * env.starting_chips)
    hero = env.seats[hero_seat]

    # 1. Hero Hole Cards (2 cards -> 4 floats)
    for c_i in range(2):
        if c_i < len(hero.hand) and hero.hand[c_i] != -1:
            c = hero.hand[c_i]
            obs[idx] = normalize_rank(rank_of(c))
            obs[idx + 1] = normalize_suit(suit_of(c))
        else:
            obs[idx] = -1.0
            obs[idx + 1] = -1.0
        idx += 2

    # 2. Community Board Cards (5 cards -> 10 floats)
    for b_i in range(5):
        if b_i < len(env.board) and env.board[b_i] != -1:
            c = env.board[b_i]
            obs[idx] = normalize_rank(rank_of(c))
            obs[idx + 1] = normalize_suit(suit_of(c))
        else:
            obs[idx] = -1.0
            obs[idx + 1] = -1.0
        idx += 2

    # 3. Current Hand Strength (1 float)
    if len(hero.hand) == 2 and len(env.board) >= 3:
        score, _ = evaluate_holdem_hand(hero.hand + env.board)
        # Normalize score (0..8,000,000) to [0.0, 1.0] -> [-1.0, 1.0]
        strength = min(1.0, score / 8_000_000.0)
        obs[idx] = (strength * 2.0) - 1.0
    elif len(hero.hand) == 2:
        r1, r2 = rank_of(hero.hand[0]), rank_of(hero.hand[1])
        preflop_val = (max(r1, r2) + (3 if r1 == r2 else 0)) / 15.0
        obs[idx] = (min(1.0, preflop_val) * 2.0) - 1.0
    else:
        obs[idx] = -1.0
    idx += 1

    # 4. Pot Odds (1 float)
    pot_odds = hero.bet_to_call / max(1.0, float(env.pot + hero.bet_to_call))
    obs[idx] = (min(1.0, pot_odds) * 2.0) - 1.0
    idx += 1

    # 5. Pot Size normalized (1 float)
    obs[idx] = (min(1.0, env.pot / float(total_chips)) * 2.0) - 1.0
    idx += 1

    # 6. Hero Stack normalized (1 float)
    obs[idx] = (min(1.0, hero.chips / float(total_chips)) * 2.0) - 1.0
    idx += 1

    # 7. Hero Bet to Call normalized (1 float)
    b2c_ratio = hero.bet_to_call / max(1.0, float(hero.chips))
    obs[idx] = (min(1.0, b2c_ratio) * 2.0) - 1.0
    idx += 1

    # 8. Street Indicator (4 one-hot floats)
    street_map = {STREET_PREFLOP: 0, STREET_FLOP: 1, STREET_TURN: 2, STREET_RIVER: 3}
    curr_s_idx = street_map.get(env.street, 0)
    for s_i in range(4):
        obs[idx + s_i] = 1.0 if s_i == curr_s_idx else -1.0
    idx += 4

    # 9. Hero Position relative to Button (5 one-hot floats)
    pos_offset = (hero_seat - env.dealer_seat) % env.num_seats
    for p_i in range(env.num_seats):
        obs[idx + p_i] = 1.0 if p_i == pos_offset else -1.0
    idx += env.num_seats

    # 10. Per-Seat Status (5 seats * 5 features = 25 floats)
    for s in env.seats:
        obs[idx] = (min(1.0, s.chips / float(total_chips)) * 2.0) - 1.0
        obs[idx + 1] = (min(1.0, s.round_invested / max(1.0, float(env.pot))) * 2.0) - 1.0
        obs[idx + 2] = 1.0 if s.is_alive else -1.0
        obs[idx + 3] = 1.0 if s.folded else -1.0
        obs[idx + 4] = 1.0 if s.is_all_in else -1.0
        idx += 5

    # 11. Street Action History (4 streets * 5 seats = 20 floats)
    for s_name in [STREET_PREFLOP, STREET_FLOP, STREET_TURN, STREET_RIVER]:
        for s in env.seats:
            act = s.street_actions.get(s_name, -1)
            if act == -1:
                obs[idx] = -1.0
            else:
                obs[idx] = (act / 2.5) - 1.0  # actions 0..5 normalized to [-1, 1]
            idx += 1

    return obs


class HoldemPokerGymEnv(gym.Env):
    """
    Gymnasium environment for No-Limit Texas Hold'em.
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
        blind_escalation_interval: int = 15,
        randomize_stacks: bool = False,
        fold_penalty: float = 0.05,
        rng_seed: Optional[int] = None,
    ):
        super().__init__()
        self.hero_seat = hero_seat
        self.num_seats = num_seats
        self.starting_chips = starting_chips
        self.randomize_stacks = randomize_stacks
        self.fold_penalty = fold_penalty
        self.rng = np.random.default_rng(rng_seed)

        self.env = HoldemPokerEnv(
            num_seats=num_seats,
            starting_chips=starting_chips,
            small_blind=small_blind,
            big_blind=big_blind,
            max_hands_per_session=max_hands_per_session,
            blind_escalation_interval=blind_escalation_interval,
            rng=self.rng,
        )

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM_HOLDEM,), dtype=np.float32
        )

        self.opponents: Dict[int, HoldemOpponent] = {}
        self._init_opponents()

    def _init_opponents(self):
        for s in range(self.num_seats):
            if s != self.hero_seat:
                self.opponents[s] = make_random_holdem_opponent(opp_id=s, rng=self.rng)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.env.rng = self.rng
            self._init_opponents()

        self.env.reset_session(randomize_stacks=self.randomize_stacks)
        self._step_until_hero_turn()

        obs = build_observation_vector_holdem(self.env, self.hero_seat)
        info = self._get_info()
        return obs, info

    def action_masks(self) -> np.ndarray:
        return self.env.get_action_mask(self.hero_seat)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        p = self.env.seats[self.hero_seat]
        prev_chips = p.chips

        self.env.step_action(self.hero_seat, action)
        self._step_until_hero_turn()

        new_chips = p.chips
        chip_delta = new_chips - prev_chips

        reward = float(chip_delta) / float(self.env.big_blind)
        if action == A_FOLD:
            reward -= self.fold_penalty

        terminated = (p.chips <= 0) or (self.env.session_done)
        truncated = False

        obs = build_observation_vector_holdem(self.env, self.hero_seat)
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _step_until_hero_turn(self):
        while not self.env.session_done:
            cur = self.env.current_seat
            if cur == self.hero_seat:
                p = self.env.seats[self.hero_seat]
                if p.is_alive and not p.folded and not p.is_all_in:
                    return

            opp_player = self.env.seats[cur]
            legal = self.env.get_legal_actions(cur)
            if not legal:
                self.env._advance_betting()
                continue

            bot = self.opponents.get(cur)
            if bot:
                act = bot.act(
                    hole_cards=opp_player.hand,
                    board=self.env.board,
                    pot=self.env.pot,
                    bet_to_call=opp_player.bet_to_call,
                    street=self.env.street,
                    stack=opp_player.chips,
                    legal_actions=legal,
                )
            else:
                act = A_CHECK_CALL if A_CHECK_CALL in legal else legal[0]

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
