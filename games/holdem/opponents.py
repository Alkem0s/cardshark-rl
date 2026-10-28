"""
games/holdem/opponents.py — Heuristic Opponent Archetypes for Texas Hold'em.

Archetypes:
1. TAGBot (Tight-Aggressive): Disciplined preflop ranges, aggressive value bets.
2. LAGBot (Loose-Aggressive): Wide ranges, high bluff frequency and pressure.
3. CallingStationBot: Passive calling station, calls down with weak pairs.
4. RandomHoldemBot: Stochastic baseline for environment exploration.
"""

from __future__ import annotations
import os
from typing import List, Optional, Dict, Any
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
        # Parameter jitter for curriculum generalization (+/- 15%)
        self.call_thresh = float(np.clip(0.35 + self.rng.uniform(-0.08, 0.08), 0.20, 0.50))
        self.postflop_call = float(np.clip(0.40 + self.rng.uniform(-0.10, 0.10), 0.25, 0.55))
        self.aggression_freq = float(np.clip(0.90 + self.rng.uniform(-0.15, 0.10), 0.70, 1.00))

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
                if self.rng.random() < self.aggression_freq:
                    for a in [A_POT, A_HALF_POT, A_MIN_RAISE]:
                        if a in legal_actions:
                            return a
                return A_CHECK_CALL
            # Playable speculative hands (pairs, suited broadway)
            elif is_pair or (is_suited and high_r >= 9):
                if bet_to_call <= pot * self.call_thresh:
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
            if self.rng.random() < self.aggression_freq:
                for a in [A_POT, A_HALF_POT, A_MIN_RAISE]:
                    if a in legal_actions:
                        return a
            return A_CHECK_CALL
        elif category == 1:  # One pair
            if bet_to_call <= pot * self.postflop_call:
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
        # Parameter jitter for curriculum generalization
        self.open_freq = float(np.clip(0.60 + self.rng.uniform(-0.15, 0.15), 0.40, 0.85))
        self.cbet_freq = float(np.clip(0.50 + self.rng.uniform(-0.15, 0.15), 0.35, 0.75))
        self.bluff_freq = float(np.clip(0.25 + self.rng.uniform(-0.10, 0.10), 0.10, 0.40))
        self.call_limit = float(np.clip(0.20 + self.rng.uniform(-0.08, 0.08), 0.10, 0.35))

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
                if self.rng.random() < self.open_freq:
                    for a in [A_HALF_POT, A_MIN_RAISE]:
                        if a in legal_actions:
                            return a
                return A_CHECK_CALL
            if bet_to_call <= pot * self.call_limit:
                return A_CHECK_CALL
            return A_FOLD if A_FOLD in legal_actions else A_CHECK_CALL

        # Postflop: frequent c-bets and floats
        all_cards = hole_cards + board
        category = holdem_category(all_cards)

        if category >= 1:
            if self.rng.random() < self.cbet_freq:
                for a in [A_HALF_POT, A_MIN_RAISE]:
                    if a in legal_actions:
                        return a
            return A_CHECK_CALL
        else:
            # Pure bluff
            if self.rng.random() < self.bluff_freq and bet_to_call == 0:
                if A_HALF_POT in legal_actions:
                    return A_HALF_POT
            if bet_to_call == 0:
                return A_CHECK_CALL
            return A_FOLD if A_FOLD in legal_actions else A_CHECK_CALL


class CallingStationBot(HoldemOpponent):
    """Passive Calling Station: Rarely folds, rarely raises."""

    def __init__(self, opponent_id: int = 3, name: str = "CallingStation", rng=None):
        super().__init__(opponent_id, name, rng)
        self.all_in_fold_prob = float(np.clip(0.05 + self.rng.uniform(-0.05, 0.10), 0.0, 0.20))

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
        if bet_to_call >= pot * 2.0 and self.rng.random() < self.all_in_fold_prob:
            return A_FOLD if A_FOLD in legal_actions else A_CHECK_CALL
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


_HOLDEM_MODEL_CACHE: Dict[str, Any] = {}


def get_cached_holdem_model(model_path: str):
    """Loads and caches neural network checkpoints to prevent duplicate disk reads."""
    if model_path not in _HOLDEM_MODEL_CACHE:
        if not os.path.exists(model_path):
            return None
        try:
            from sb3_contrib import MaskablePPO
            _HOLDEM_MODEL_CACHE[model_path] = MaskablePPO.load(model_path)
        except Exception:
            return None
    return _HOLDEM_MODEL_CACHE[model_path]


class HoldemModelOpponent(HoldemOpponent):
    """Neural policy opponent loading a trained CardShark-RL checkpoint."""

    def __init__(
        self,
        opponent_id: int = 1,
        model_path: str = "models/holdem/model_holdem_b.zip",
        name: str = "ModelBOpponent",
        obs_version: Optional[str] = None,
        deterministic: bool = False,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(opponent_id, name, rng)
        self.model_path = model_path
        self.deterministic = deterministic
        self.is_neural = True
        self.model = get_cached_holdem_model(self.model_path)

        if obs_version is not None:
            self.obs_version = obs_version.lower()
        elif self.model is not None and hasattr(self.model, "observation_space"):
            dim = int(self.model.observation_space.shape[0])
            if dim == 158:
                self.obs_version = "e"
            elif dim == 152:
                self.obs_version = "d"
            elif dim == 138:
                self.obs_version = "c"
            elif dim in (119, 125):
                self.obs_version = "b"
            else:
                self.obs_version = "a"
        else:
            self.obs_version = "d"

    def act_with_env(self, env, seat_idx: int, legal_actions: List[int]) -> int:
        """Executes true neural inference using seat-perspective observation and action masking."""
        if not legal_actions:
            return A_CHECK_CALL
        if self.model is not None:
            try:
                from games.holdem.gym_wrapper import build_observation_vector_holdem
                obs = build_observation_vector_holdem(
                    env,
                    hero_seat=seat_idx,
                    obs_version=self.obs_version,
                    tracker=getattr(env, "tracker", None),
                )
                mask = env.get_action_mask(seat_idx)
                act, _ = self.model.predict(obs, action_masks=mask, deterministic=self.deterministic)
                return int(act)
            except Exception:
                pass

        seat = env.seats[seat_idx]
        return self.act(
            hole_cards=seat.hand,
            board=env.board,
            pot=env.pot,
            bet_to_call=seat.bet_to_call,
            street=env.street,
            stack=seat.chips,
            legal_actions=legal_actions,
        )

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
        if len(hole_cards) == 2:
            try:
                from games.holdem.evaluator import evaluate_holdem_hand, evaluate_preflop_hand_chen
                if street == STREET_PREFLOP:
                    chen = evaluate_preflop_hand_chen(hole_cards[0], hole_cards[1])
                    if chen >= 0.45:
                        for a in [A_POT, A_MIN_RAISE]:
                            if a in legal_actions:
                                return a
                    elif chen >= 0.25:
                        return A_CHECK_CALL if A_CHECK_CALL in legal_actions else legal_actions[0]
                    else:
                        return A_FOLD if (A_FOLD in legal_actions and bet_to_call > 0) else A_CHECK_CALL
                else:
                    if len(board) >= 3:
                        score, _ = evaluate_holdem_hand(hole_cards + board)
                        if score >= 3_000_000:
                            for a in [A_POT, A_MIN_RAISE]:
                                if a in legal_actions:
                                    return a
                        elif score >= 1_500_000:
                            return A_CHECK_CALL if A_CHECK_CALL in legal_actions else legal_actions[0]
                        else:
                            if bet_to_call > 0:
                                return A_FOLD if A_FOLD in legal_actions else A_CHECK_CALL
                            return A_CHECK_CALL
            except Exception:
                pass
        return A_CHECK_CALL if A_CHECK_CALL in legal_actions else legal_actions[0]

    def act_with_obs(self, obs: np.ndarray, mask: np.ndarray) -> int:
        if self.model is not None:
            try:
                act, _ = self.model.predict(obs, action_masks=mask, deterministic=self.deterministic)
                return int(act)
            except Exception:
                pass
        valid = np.where(mask == 1)[0]
        return int(valid[0]) if len(valid) > 0 else A_CHECK_CALL


def make_random_holdem_opponent(
    opp_id: int,
    rng: Optional[np.random.Generator] = None,
    league_mode: bool = False,
    model_b_path: str = "models/holdem/model_holdem_b.zip",
    model_c_path: str = "models/holdem/model_holdem_c.zip",
    model_d_path: str = "models/holdem/model_holdem_d_best.zip",
) -> HoldemOpponent:
    r = rng or np.random.default_rng()
    roll = r.random()

    # Discover available neural checkpoints for multi-agent fictitious play
    available_models = [p for p in [model_d_path, model_c_path, model_b_path] if os.path.exists(p)]

    if league_mode and available_models:
        # 40% neural sparring against league checkpoints
        if roll < 0.40:
            chosen_model = str(r.choice(available_models))
            return HoldemModelOpponent(opponent_id=opp_id, model_path=chosen_model, rng=r)
        elif roll < 0.65:
            return TAGBot(opponent_id=opp_id, rng=r)
        elif roll < 0.85:
            return LAGBot(opponent_id=opp_id, rng=r)
        elif roll < 0.95:
            return CallingStationBot(opponent_id=opp_id, rng=r)
        else:
            return RandomHoldemBot(opponent_id=opp_id, rng=r)

    if roll < 0.35:
        return TAGBot(opponent_id=opp_id, rng=r)
    elif roll < 0.65:
        return LAGBot(opponent_id=opp_id, rng=r)
    elif roll < 0.85:
        return CallingStationBot(opponent_id=opp_id, rng=r)
    else:
        return RandomHoldemBot(opponent_id=opp_id, rng=r)

