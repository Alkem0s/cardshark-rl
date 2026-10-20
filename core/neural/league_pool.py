"""
core/neural/league_pool.py — Universal Prioritized Fictitious Self-Play (PFSP) & League Manager.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Any
import numpy as np

_MODEL_CACHE: Dict[str, Any] = {}

def get_cached_model(model_path: str):
    """Loads and caches neural network checkpoints to prevent duplicate disk reads."""
    if model_path not in _MODEL_CACHE:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"League checkpoint not found: {model_path}")
        from sb3_contrib import MaskablePPO
        _MODEL_CACHE[model_path] = MaskablePPO.load(model_path)
    return _MODEL_CACHE[model_path]


class LeaguePool:
    """
    Manages a pool of historical checkpoints for Prioritized Fictitious Self-Play (PFSP).
    Selects opponent checkpoints according to curriculum stages and self-play ratios.
    """

    def __init__(
        self,
        base_model_path: Optional[str] = None,
        league_dir: str = "models/league",
        neural_opponent_prob: float = 0.55,
        min_neural_prob: float = 0.35,
        curriculum_warmup_steps: int = 250_000,
        self_play_prob: float = 0.40,
        exploiter_prob: float = 0.30,
        exclude_prefix: Optional[str] = None,
    ):
        self.base_model_path = base_model_path
        self.league_dir = league_dir
        self.neural_opponent_prob = neural_opponent_prob
        self.min_neural_prob = min_neural_prob
        self.curriculum_warmup_steps = curriculum_warmup_steps
        self.self_play_prob = self_play_prob
        self.exploiter_prob = exploiter_prob
        self.exclude_prefix = exclude_prefix
        self.current_step = 0

        self.checkpoints: List[str] = []
        self.active_snapshots: List[str] = []
        self._refresh_checkpoints()

    def _refresh_checkpoints(self):
        if self.base_model_path and os.path.exists(self.base_model_path):
            if self.base_model_path not in self.checkpoints:
                self.checkpoints.append(self.base_model_path)

        if os.path.exists(self.league_dir):
            files = sorted(os.listdir(self.league_dir))
            for f in files:
                if f.endswith(".zip"):
                    full_p = os.path.join(self.league_dir, f)
                    if self.exclude_prefix and f.startswith(self.exclude_prefix):
                        continue
                    if full_p not in self.checkpoints:
                        self.checkpoints.append(full_p)

    def add_snapshot(self, snapshot_path: str):
        if os.path.exists(snapshot_path):
            if snapshot_path not in self.checkpoints:
                self.checkpoints.append(snapshot_path)
            if snapshot_path not in self.active_snapshots:
                self.active_snapshots.append(snapshot_path)

    def set_step(self, step: int):
        self.current_step = step

    def get_neural_probability(self) -> float:
        if self.curriculum_warmup_steps <= 0:
            return self.neural_opponent_prob
        progress = min(1.0, self.current_step / float(self.curriculum_warmup_steps))
        return self.min_neural_prob + progress * (self.neural_opponent_prob - self.min_neural_prob)
