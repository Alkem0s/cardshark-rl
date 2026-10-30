"""
rl/holdem_curriculum.py — Two-Stage Curriculum Reward Annealing for Hold'em Model E.

Smoothly transitions the agent from early guided exploration (heavy Bayesian exploit
bonuses, ICM shaping, station penalties) to pure unshaped chip-EV and tournament equity
maximization, eliminating long-horizon reward-shaping distortion.
"""

from __future__ import annotations
import math
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback


class HoldemCurriculumRewardCallback(BaseCallback):
    """
    Two-Stage Curriculum Callback:
    - Stage 1 (0 to scaffold_steps): shaping_mult = 1.0 (Full guidance)
    - Stage 2 (scaffold_steps to total_timesteps): Smooth cosine anneal from 1.0 to min_shaping_mult.
    """

    def __init__(
        self,
        scaffold_steps: int = 500_000,
        total_timesteps: int = 2_500_000,
        min_shaping_mult: float = 0.0,
        log_freq: int = 25_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.scaffold_steps = int(scaffold_steps)
        self.total_timesteps = max(self.scaffold_steps + 1, int(total_timesteps))
        self.min_shaping_mult = float(min_shaping_mult)
        self.log_freq = int(log_freq)
        self.current_mult = 1.0
        self.last_logged_step = 0

    def _on_step(self) -> bool:
        t = self.num_timesteps
        if t <= self.scaffold_steps:
            mult = 1.0
        else:
            anneal_span = float(self.total_timesteps - self.scaffold_steps)
            progress = min(1.0, max(0.0, (t - self.scaffold_steps) / anneal_span))
            # Smooth cosine decay
            mult = self.min_shaping_mult + 0.5 * (1.0 - self.min_shaping_mult) * (
                1.0 + math.cos(math.pi * progress)
            )

        self.current_mult = float(mult)
        overall_progress = min(1.0, max(0.0, float(t) / float(self.total_timesteps)))

        # Broadcast to vectorized environment workers
        if getattr(self, "model", None) is not None:
            try:
                env = self.training_env
                if env is not None and hasattr(env, "set_attr"):
                    env.set_attr("curriculum_shaping_mult", self.current_mult)
                    env.set_attr("league_progress", overall_progress)
            except Exception:
                pass

        if t - self.last_logged_step >= self.log_freq and self.verbose > 0:
            self.last_logged_step = t
            stage = "Stage 1 (Scaffolding)" if t <= self.scaffold_steps else "Stage 2 (Pure-Equity)"
            p_neural = (0.15 + 0.60 * overall_progress) * 100.0
            p_heur = max(0.0, 100.0 - p_neural - 5.0)
            print(
                f"  [Curriculum {t:,} steps] {stage} | Shaping: {self.current_mult:.4f} | Sparring: {p_heur:.0f}% Heuristics / {p_neural:.0f}% League"
            )

        return True
