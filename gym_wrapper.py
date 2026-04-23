"""
gym_wrapper.py — Gymnasium wrapper around DrawPokerEnv for SB3 MaskablePPO.

Wraps the PettingZoo AEC environment into a standard single-agent
Gymnasium env. The opponent (hardcoded archetype) acts automatically
inside step(). Exposes action_mask() for sb3-contrib's MaskablePPO.
"""

from __future__ import annotations
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple

from draw_poker_env import (
    DrawPokerEnv, TOTAL_ACTIONS, PHASE_SHOWDOWN,
    A_FOLD, A_CALL, A_RAISE, A_DRAW_START, A_DRAW_END,
    PHASE_PRE_DRAW, PHASE_POST_DRAW,
)
from opponents import NUM_ARCHETYPES
from card_utils import hand_category


class DrawPokerGymEnv(gym.Env):
    """Single-agent Gymnasium wrapper for the Draw Poker environment.

    The RL agent is always player_0. The opponent (player_1) is
    controlled by a hardcoded archetype embedded inside the env.

    Important: This wrapper handles the case where the opponent acts
    first and ends the hand during reset (e.g., if the opponent is
    the Rock and folds immediately). In these cases, reset() keeps
    retrying until a hand is dealt where the RL agent gets to act.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        use_implicit_modeling: bool = False,
        opponent_id: int | None = None,
        rolling_window: int = 50,
        render_mode: str | None = None,
        rng_seed: int | None = None,
        opponent_schedule: str = "random",
        block_size: int = 200,
    ):
        super().__init__()

        self.use_implicit_modeling = use_implicit_modeling
        self.opponent_id = opponent_id
        self.rolling_window = rolling_window
        self.render_mode = render_mode

        # Create the inner AEC env
        self._env = DrawPokerEnv(
            use_implicit_modeling=use_implicit_modeling,
            opponent_id=opponent_id,
            rolling_window=rolling_window,
            render_mode=render_mode,
            rng_seed=rng_seed,
            opponent_schedule=opponent_schedule,
            block_size=block_size,
        )

        # Determine obs size
        obs_size = self._env._compute_obs_size()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(TOTAL_ACTIONS)

        # Tracking for evaluation
        self.episode_reward = 0.0
        self.episode_length = 0
        self.total_hands = 0

        # Behavioral tracking for "The Tell" analysis
        self._post_draw_actions: list = []   # (opp_draw_count, agent_action, opp_id)

        # Cache the last reward from auto-resolved hands
        self._pending_reward = 0.0

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self._env.rng = np.random.default_rng(seed)
            self._env.deck.rng = self._env.rng

        self._pending_reward = 0.0

        # Keep resetting until we get a hand where the RL agent can act.
        # This handles the edge case where the opponent acts first and
        # the hand ends (e.g., Rock folds pre-draw when out of position).
        max_retries = 100
        for _ in range(max_retries):
            self._env.reset(seed=None, options=options)
            if not self._env.terminations.get("player_0", False):
                break
            # Hand ended during opponent's auto-play — accumulate reward
            self._pending_reward += self._env.rewards.get("player_0", 0)

        self.episode_reward = 0.0
        self.episode_length = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step as the RL agent (player_0).

        The inner env handles opponent auto-stepping.
        """
        # Track post-draw actions for behavioral analysis
        if self._env.phase == PHASE_POST_DRAW:
            opp_draw = self._env.draw_counts.get("player_1", 0)
            opp_id = (
                self._env.current_opponent.opponent_id
                if self._env.current_opponent
                else -1
            )
            self._post_draw_actions.append((opp_draw, action, opp_id))

        # Execute the action in the inner env
        self._env.step(action)

        # Collect results
        terminated = self._env.terminations.get("player_0", False)
        truncated = self._env.truncations.get("player_0", False)
        reward = self._env.rewards.get("player_0", 0) + self._pending_reward
        self._pending_reward = 0.0

        # Reward shaping
        if not terminated and action == A_FOLD and self._env._bet_to_call.get("player_0", 0) == 0:
            # Penalty for folding when checking was an option
            reward -= 0.3

        # Steal-attempt bonus: if agent raised and opponent folded, reward the steal
        if terminated and action == A_RAISE and self._env.folded.get("player_1", False):
            reward += 0.2

        self.episode_reward += reward
        self.episode_length += 1

        if terminated or truncated:
            self.total_hands += 1

        obs = self._get_obs()
        info = self._get_info()

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Get observation for the RL agent."""
        if self._env.terminations.get("player_0", False):
            # Return zero obs on terminal
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        obs_dict = self._env.observe("player_0")
        return obs_dict["observation"]

    def _get_info(self) -> dict:
        return {
            "phase": self._env.phase,
            "pot": self._env.pot,
            "opponent": (
                self._env.current_opponent.name
                if self._env.current_opponent
                else "none"
            ),
        }

    def action_mask(self) -> np.ndarray:
        """Return the action mask for the current state.

        Used by sb3-contrib's ActionMasker wrapper.
        """
        if self._env.terminations.get("player_0", False):
            # All actions valid on terminal (won't be used)
            return np.ones(TOTAL_ACTIONS, dtype=np.int8)
        obs_dict = self._env.observe("player_0")
        return obs_dict["action_mask"]

    def render(self):
        self._env.render()

    def close(self):
        pass

    def get_post_draw_actions(self) -> list:
        """Return the recorded post-draw action history for analysis."""
        return list(self._post_draw_actions)

    def clear_post_draw_actions(self):
        """Clear the behavioral tracking buffer."""
        self._post_draw_actions.clear()


# ---------------------------------------------------------------------------
# Factory functions for vectorised env creation
# ---------------------------------------------------------------------------

def make_env(
    use_implicit: bool = False,
    opponent_id: int | None = None,
    rolling_window: int = 50,
    seed: int = 0,
):
    """Create a closure that returns a DrawPokerGymEnv."""
    def _init():
        env = DrawPokerGymEnv(
            use_implicit_modeling=use_implicit,
            opponent_id=opponent_id,
            rolling_window=rolling_window,
            rng_seed=seed,
        )
        return env
    return _init


def mask_fn(env: DrawPokerGymEnv) -> np.ndarray:
    """Mask function for sb3-contrib's ActionMasker."""
    return env.action_mask()
