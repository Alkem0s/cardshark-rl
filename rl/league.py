"""
rl/league.py — Multi-Agent League Sparring (Fictitious Play) for Model C & D.

Manages frozen historical checkpoints and enables self-play sparring during training.
Cleanly decoupled from pure poker game rules.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Any
import numpy as np

from game.multi_opponents import MultiPlayerOpponent, TAG, make_random_archetype
from rl.opponent_tracker import TableOpponentTracker

_MODEL_CACHE: Dict[str, Any] = {}

def get_cached_model(model_path: str):
    """Loads and caches neural network checkpoints to prevent duplicate disk reads."""
    if model_path not in _MODEL_CACHE:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"League checkpoint not found: {model_path}")
        from sb3_contrib import MaskablePPO
        _MODEL_CACHE[model_path] = MaskablePPO.load(model_path)
    return _MODEL_CACHE[model_path]


class LeagueOpponent(MultiPlayerOpponent):
    """Neural network opponent loaded from a frozen checkpoint (Model B, C, or D snapshot)."""

    def __init__(
        self,
        model_path: str,
        opponent_id: int = 100,
        name: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
        deterministic: bool = False,
    ):
        base_name = name or f"League_{os.path.basename(model_path).replace('.zip', '')}"
        super().__init__(opponent_id=opponent_id, name=base_name, rng=rng)
        self.model_path = model_path
        self.deterministic = deterministic
        self.model = get_cached_model(model_path)
        self.is_superhuman = (self.model.observation_space.shape[0] == 87)
        self.tracker = TableOpponentTracker(num_seats=5)
        self._fallback_bot = TAG(rng=self.rng)

    def act_with_env(self, env, seat_idx: int, legal_actions: List[int]) -> int:
        from rl.multi_gym_wrapper import build_observation_vector
        try:
            obs = build_observation_vector(
                env=env,
                tracker=self.tracker,
                seat_idx=seat_idx,
                superhuman_obs=self.is_superhuman,
            )
            mask = env.get_action_mask(seat_idx)
            action_id, _ = self.model.predict(obs, action_masks=mask, deterministic=self.deterministic)
            action_id = int(action_id)
            if action_id in legal_actions:
                return action_id
        except Exception:
            pass
        return self._fallback_bot.bet_action(
            hand=env.seats[seat_idx].hand,
            pot=env.pot,
            bet_to_call=env.seats[seat_idx].bet_to_call,
            phase=env.phase,
            stack=env.seats[seat_idx].chips,
            legal_actions=legal_actions,
        )

    def draw_with_env(self, env, seat_idx: int) -> int:
        legal = env.get_legal_actions(seat_idx)
        return self.act_with_env(env, seat_idx, legal)

    def bet_action(
        self,
        hand: List[int],
        pot: int,
        bet_to_call: int,
        phase: str,
        stack: int,
        legal_actions: List[int],
    ) -> int:
        return self._fallback_bot.bet_action(
            hand=hand,
            pot=pot,
            bet_to_call=bet_to_call,
            phase=phase,
            stack=stack,
            legal_actions=legal_actions,
        )

    def draw_action(self, hand: List[int]) -> List[int]:
        return self._fallback_bot.draw_action(hand)


class LeaguePool:
    """Manages the pool of sparring agents (frozen Model B, historical Model C/D checkpoints, and heuristics)."""

    def __init__(
        self,
        base_model_path: str = "models/model_b.zip",
        league_dir: str = "models/league",
        neural_opponent_prob: float = 0.50,
    ):
        self.base_model_path = base_model_path
        self.league_dir = league_dir
        self.neural_opponent_prob = neural_opponent_prob
        self.checkpoints: List[str] = []
        os.makedirs(self.league_dir, exist_ok=True)
        self.refresh_checkpoints()

    def refresh_checkpoints(self):
        """Scans the filesystem for valid model checkpoints."""
        self.checkpoints = []
        # Support both model_b.zip and legacy model_b_multiplayer.zip
        for candidate in [self.base_model_path, "models/model_b_multiplayer.zip"]:
            if os.path.exists(candidate) and candidate not in self.checkpoints:
                self.checkpoints.append(candidate)
        if os.path.exists(self.league_dir):
            for fname in sorted(os.listdir(self.league_dir)):
                if fname.endswith(".zip"):
                    full_p = os.path.join(self.league_dir, fname)
                    if full_p not in self.checkpoints:
                        self.checkpoints.append(full_p)

    def add_snapshot(self, model_path: str):
        """Adds a newly created training snapshot to the sparring pool."""
        if os.path.exists(model_path) and model_path not in self.checkpoints:
            self.checkpoints.append(model_path)

    def sample_opponent(
        self,
        seat_idx: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> MultiPlayerOpponent:
        """Samples either a neural LeagueOpponent or a heuristic archetype."""
        r = rng or np.random.default_rng()
        if self.checkpoints and r.random() < self.neural_opponent_prob:
            chosen_path = str(r.choice(self.checkpoints))
            try:
                return LeagueOpponent(
                    model_path=chosen_path,
                    opponent_id=100 + seat_idx,
                    rng=r,
                    deterministic=False,
                )
            except Exception:
                return make_random_archetype(rng=r)
        else:
            return make_random_archetype(rng=r)
