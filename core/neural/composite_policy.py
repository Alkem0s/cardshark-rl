"""
core/neural/composite_policy.py — Composite Regime-Switching Policy Architecture.

Addresses the mathematical objective mismatch between:
1. Multiway Tournament Play (3-5 players): Non-linear ICM risk aversion, stack preservation.
2. Heads-Up Endgame Play (2 players): Linear zero-sum chip-EV, aggressive button exploitation.

HoldemCompositeModel maintains explicit, configurable slots for:
- tournament_model: Any trained policy (e.g. Model F, Model E, baseline PPO).
- heads_up_model: Any dedicated 1v1 policy (e.g. Model G, specialized heads-up policy).

During inference, it dynamically inspects the observation vector's heads-up regime feature
(feature index 163 in the 168-dimensional state vector) and routes the decision to the
appropriate specialist with zero latency (<1 ms).
"""

from __future__ import annotations
import os
from typing import Optional, Tuple, Dict, Any, Union, List
import numpy as np


class HoldemCompositeModel:
    """
    Plug-and-play composite regime-switching agent.
    
    Routes decisions between a tournament model and a heads-up model dynamically
    based on the observation's regime indicator.
    """

    def __init__(
        self,
        tournament_model: Any,
        heads_up_model: Any,
        heads_up_feature_idx: int = 163,
        name: str = "Composite_Regime_Switching_Model",
    ):
        self.tournament_model = tournament_model
        self.heads_up_model = heads_up_model
        self.heads_up_feature_idx = heads_up_feature_idx
        self.name = name

    @classmethod
    def load(
        cls,
        tournament_model_path: str,
        heads_up_model_path: str,
        device: str = "cpu",
        heads_up_feature_idx: int = 163,
        name: Optional[str] = None,
    ) -> "HoldemCompositeModel":
        """
        Loads pre-trained MaskablePPO checkpoints from disk into the corresponding slots.
        """
        from sb3_contrib import MaskablePPO

        if not os.path.exists(tournament_model_path):
            raise FileNotFoundError(f"Tournament model checkpoint not found: {tournament_model_path}")
        if not os.path.exists(heads_up_model_path):
            raise FileNotFoundError(f"Heads-up model checkpoint not found: {heads_up_model_path}")

        t_model = MaskablePPO.load(tournament_model_path, device=device)
        hu_model = MaskablePPO.load(heads_up_model_path, device=device)

        model_name = name or f"Composite_{os.path.basename(tournament_model_path).replace('.zip', '')}+{os.path.basename(heads_up_model_path).replace('.zip', '')}"
        return cls(
            tournament_model=t_model,
            heads_up_model=hu_model,
            heads_up_feature_idx=heads_up_feature_idx,
            name=model_name,
        )

    def is_heads_up(self, obs: np.ndarray) -> Union[bool, np.ndarray]:
        """
        Checks whether the given observation corresponds to a heads-up state.
        In the 168-dim state vector, feature 163 is 1.0 if num_alive == 2, -1.0 otherwise.
        """
        if obs.ndim == 1:
            if len(obs) > self.heads_up_feature_idx:
                return bool(obs[self.heads_up_feature_idx] > 0.0)
            return False
        else:
            if obs.shape[1] > self.heads_up_feature_idx:
                return obs[:, self.heads_up_feature_idx] > 0.0
            return np.zeros(obs.shape[0], dtype=bool)

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Routes inference to either tournament_model or heads_up_model.
        Supports both single observations (shape: (168,)) and batched inputs (shape: (B, 168)).
        """
        obs = np.asarray(observation)

        # Single environment step
        if obs.ndim == 1:
            hu_active = self.is_heads_up(obs)
            active_model = self.heads_up_model if hu_active else self.tournament_model
            return active_model.predict(
                obs,
                state=state,
                episode_start=episode_start,
                deterministic=deterministic,
                action_masks=action_masks,
            )

        # Vectorized batch of environments
        batch_size = obs.shape[0]
        actions = np.zeros(batch_size, dtype=np.int64)
        hu_mask = self.is_heads_up(obs)
        t_mask = ~hu_mask

        # Route tournament environments
        if np.any(t_mask):
            t_indices = np.where(t_mask)[0]
            t_obs = obs[t_indices]
            t_masks = action_masks[t_indices] if action_masks is not None else None
            t_ep_start = episode_start[t_indices] if episode_start is not None else None
            t_acts, _ = self.tournament_model.predict(
                t_obs,
                episode_start=t_ep_start,
                deterministic=deterministic,
                action_masks=t_masks,
            )
            actions[t_indices] = t_acts

        # Route heads-up environments
        if np.any(hu_mask):
            hu_indices = np.where(hu_mask)[0]
            hu_obs = obs[hu_indices]
            hu_masks = action_masks[hu_indices] if action_masks is not None else None
            hu_ep_start = episode_start[hu_indices] if episode_start is not None else None
            hu_acts, _ = self.heads_up_model.predict(
                hu_obs,
                episode_start=hu_ep_start,
                deterministic=deterministic,
                action_masks=hu_masks,
            )
            actions[hu_indices] = hu_acts

        return actions, state

    def __repr__(self) -> str:
        t_cls = self.tournament_model.__class__.__name__
        hu_cls = self.heads_up_model.__class__.__name__
        return f"<HoldemCompositeModel '{self.name}' [Tournament Slot: {t_cls}, Heads-Up Slot: {hu_cls}]>"

