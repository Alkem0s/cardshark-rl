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
    """Neural network opponent loaded from a frozen checkpoint (Model B, C, D, or E snapshot)."""

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
        self.obs_dim = int(self.model.observation_space.shape[0])
        # Backwards compatibility flag for legacy tests
        self.is_superhuman = (self.obs_dim in (87, 91, 97))
        # Discard masking is enabled for Model E and newer
        self.discard_masking = (self.obs_dim >= 91)
        self.tracker = TableOpponentTracker(num_seats=5)
        self._fallback_bot = TAG(rng=self.rng)

    def get_action_mask(self, env, seat_idx: int) -> np.ndarray:
        """Returns action mask for seat, applying dominated discard action masking if supported."""
        from game.multi_draw_poker_env import PHASE_DRAW, TOTAL_ACTIONS, A_DRAW_START
        if self.discard_masking and env.phase == PHASE_DRAW:
            from rl.discard_masker import get_legal_discard_mask
            hand = env.seats[seat_idx].hand
            discard_mask = get_legal_discard_mask(hand)
            mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
            for bitmask_idx in range(len(discard_mask)):
                if discard_mask[bitmask_idx]:
                    mask[A_DRAW_START + bitmask_idx] = 1
            if not np.any(mask):
                mask[A_DRAW_START] = 1
            return mask
        return env.get_action_mask(seat_idx)

    def act_with_env(self, env, seat_idx: int, legal_actions: List[int]) -> int:
        from rl.multi_gym_wrapper import build_observation_vector
        try:
            obs = build_observation_vector(
                env=env,
                tracker=self.tracker,
                seat_idx=seat_idx,
                obs_dim=self.obs_dim,
            )
            mask = self.get_action_mask(env, seat_idx)
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
    """
    Manages the multi-agent co-evolutionary league sparring pool (Prioritized Fictitious Self-Play).
    Features:
    1. Historical Baseline Anchors: Model B & C historical checkpoints prevent catastrophic forgetting.
    2. Active Self-Play Snapshots: Rolling Model D snapshots sampled with recency weighting to push the meta.
    3. Curriculum Neural Probability Warmup: Ramps neural opponent density from min_prob to target_prob.
    4. Heuristic Anchors: Retains diverse archetypes (TAG, Maniac, Rock, CallingStation, AdversarialExploiter).
    """

    def __init__(
        self,
        base_model_path: str = "models/model_b.zip",
        league_dir: str = "models/league",
        neural_opponent_prob: float = 0.50,
        min_neural_prob: float = 0.30,
        curriculum_warmup_steps: int = 250_000,
        self_play_prob: float = 0.50,
        exploiter_prob: float = 0.20,
        exclude_prefix: Optional[str] = None,
    ):
        self.base_model_path = base_model_path
        self.league_dir = league_dir
        self.target_neural_prob = neural_opponent_prob
        self.min_neural_prob = min_neural_prob
        self.curriculum_warmup_steps = curriculum_warmup_steps
        self.self_play_prob = self_play_prob
        self.exploiter_prob = exploiter_prob
        self.exclude_prefix = exclude_prefix

        self.historical_checkpoints: List[str] = []
        self.active_snapshots: List[str] = []
        self.checkpoints: List[str] = []
        self.current_step: int = 0

        os.makedirs(self.league_dir, exist_ok=True)
        self.refresh_checkpoints()

    @property
    def neural_opponent_prob(self) -> float:
        """Dynamic neural opponent probability based on curriculum step."""
        if self.curriculum_warmup_steps <= 0:
            return self.target_neural_prob
        progress = min(1.0, max(0.0, self.current_step / self.curriculum_warmup_steps))
        return self.min_neural_prob + progress * (self.target_neural_prob - self.min_neural_prob)

    def set_step(self, step: int):
        """Updates current training timestep for curriculum scheduling."""
        self.current_step = step

    def refresh_checkpoints(self):
        """Scans the filesystem for valid model checkpoints."""
        self.historical_checkpoints = []

        # Seed base models (Model B baseline, Model C superhuman, Model D attention champion)
        for candidate in [
            self.base_model_path,
            "models/model_b.zip",
            "models/model_c.zip",
            "models/model_d.zip",
        ]:
            if os.path.exists(candidate) and candidate not in self.historical_checkpoints:
                self.historical_checkpoints.append(candidate)

        if os.path.exists(self.league_dir):
            for fname in sorted(os.listdir(self.league_dir)):
                if fname.endswith(".zip"):
                    if self.exclude_prefix and fname.startswith(self.exclude_prefix):
                        continue
                    full_p = os.path.join(self.league_dir, fname)
                    if full_p not in self.historical_checkpoints:
                        self.historical_checkpoints.append(full_p)

        self.checkpoints = list(self.historical_checkpoints) + list(self.active_snapshots)

    def add_snapshot(self, model_path: str):
        """Adds a newly created training snapshot to the self-play sparring pool."""
        if os.path.exists(model_path) and model_path not in self.active_snapshots:
            self.active_snapshots.append(model_path)
        if model_path not in self.checkpoints:
            self.checkpoints.append(model_path)

    def sample_opponent(
        self,
        seat_idx: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> MultiPlayerOpponent:
        """
        Samples an opponent from the mixed ecosystem:
        1. Dedicated Adversarial Exploiter sparring (exploiter_prob, e.g. 20%) to prevent over-folding.
        2. Neural LeagueOpponent (PFSP active self-play or historical champions).
        3. Diverse heuristic archetypes (CallingStation, TAG, LAG, Maniac, Rock).
        """
        r = rng or np.random.default_rng()
        roll = r.random()

        # Dedicated exploiter sparring seat
        if self.exploiter_prob > 0 and roll < self.exploiter_prob:
            from game.multi_opponents import AdversarialExploiter
            return AdversarialExploiter(rng=r)

        prob = self.neural_opponent_prob
        rem_roll = (roll - self.exploiter_prob) / max(1e-6, 1.0 - self.exploiter_prob)

        if self.checkpoints and rem_roll < prob:
            # Decide between Active Self-Play vs Historical Anchor Pool
            if self.active_snapshots and r.random() < self.self_play_prob:
                # Recency-weighted sampling favoring recent self-play checkpoints
                n = len(self.active_snapshots)
                weights = np.arange(1, n + 1, dtype=float) ** 1.5
                probs = weights / weights.sum()
                chosen_path = str(r.choice(self.active_snapshots, p=probs))
            elif self.historical_checkpoints:
                chosen_path = str(r.choice(self.historical_checkpoints))
            else:
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
