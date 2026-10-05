"""
auxiliary_belief.py — Auxiliary Supervised Belief Head & Representation Learning for Model F.

Key Capabilities:
1. AuxiliaryBeliefHead:
   - Evaluates latent Card Attention representations to predict:
     a) Opponent hand category (9 classes: High Card to Straight Flush)
     b) Opponent bluff probability (binary: bet size > 0 with air)
2. AuxiliaryBeliefCallback:
   - Captures privileged environment targets during vectorized rollouts.
   - Executes auxiliary cross-entropy backpropagation through the Card Attention backbone
     at the conclusion of each rollout buffer.
3. Representation Shaping:
   - Teaches the Transformer attention heads to internalize range asymmetry and bluff tells
     without hardcoding any brittle decision rules.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Any

from stable_baselines3.common.callbacks import BaseCallback


class AuxiliaryBeliefHead(nn.Module):
    """
    Auxiliary prediction network predicting opponent hand category (9 classes)
    and binary bluff indicator from latent Card Attention representations.
    """

    def __init__(self, feature_dim: int = 216, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.category_head = nn.Linear(hidden_dim, 9)
        self.bluff_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            cat_logits: (B, 9) logits for hand categories 0..8
            bluff_logits: (B,) logit for binary bluff prediction
        """
        h = self.trunk(features)
        cat_logits = self.category_head(h)
        bluff_logits = self.bluff_head(h).squeeze(-1)
        return cat_logits, bluff_logits


class AuxiliaryBeliefCallback(BaseCallback):
    """
    Rollout callback that collects privileged environment targets during training
    and backpropagates auxiliary belief losses into the Card Attention feature extractor.
    """

    def __init__(
        self,
        belief_head: AuxiliaryBeliefHead,
        lr: float = 1.0e-4,
        bluff_weight: float = 0.5,
        max_batch_size: int = 2048,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.belief_head = belief_head
        self.lr = lr
        self.bluff_weight = bluff_weight
        self.max_batch_size = max_batch_size

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._obs_buffer: List[np.ndarray] = []
        self._cat_buffer: List[int] = []
        self._bluff_buffer: List[float] = []

    def _init_callback(self) -> None:
        device = self.model.device
        self.belief_head.to(device)

        # Optimize both the belief head AND the shared features_extractor
        extractor = self.model.policy.features_extractor
        params = list(self.belief_head.parameters()) + list(extractor.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        new_obs = self.locals.get("new_obs", None)

        if new_obs is not None and len(infos) > 0:
            for i, info in enumerate(infos):
                if "opp_hand_cat" in info and "is_bluff" in info:
                    self._obs_buffer.append(new_obs[i])
                    self._cat_buffer.append(int(info["opp_hand_cat"]))
                    self._bluff_buffer.append(float(info["is_bluff"]))

        return True

    def _on_rollout_end(self) -> None:
        if not self._obs_buffer or self.optimizer is None:
            return

        device = self.model.device
        n_samples = len(self._obs_buffer)

        # Subsample if buffer exceeds max_batch_size
        if n_samples > self.max_batch_size:
            indices = np.random.choice(n_samples, size=self.max_batch_size, replace=False)
            obs_batch = np.array([self._obs_buffer[idx] for idx in indices], dtype=np.float32)
            cat_batch = np.array([self._cat_buffer[idx] for idx in indices], dtype=np.int64)
            bluff_batch = np.array([self._bluff_buffer[idx] for idx in indices], dtype=np.float32)
        else:
            obs_batch = np.array(self._obs_buffer, dtype=np.float32)
            cat_batch = np.array(self._cat_buffer, dtype=np.int64)
            bluff_batch = np.array(self._bluff_buffer, dtype=np.float32)

        # Clear buffer for next rollout
        self._obs_buffer.clear()
        self._cat_buffer.clear()
        self._bluff_buffer.clear()

        # Convert to PyTorch tensors
        obs_tensor = torch.as_tensor(obs_batch, device=device)
        cat_tensor = torch.as_tensor(cat_batch, device=device)
        bluff_tensor = torch.as_tensor(bluff_batch, device=device)

        # Forward pass through shared features_extractor
        extractor = self.model.policy.features_extractor
        extractor.train()
        self.belief_head.train()

        features = extractor(obs_tensor)
        cat_logits, bluff_logits = self.belief_head(features)

        # Compute multi-task losses
        loss_cat = F.cross_entropy(cat_logits, cat_tensor)
        loss_bluff = F.binary_cross_entropy_with_logits(bluff_logits, bluff_tensor)
        total_aux_loss = loss_cat + self.bluff_weight * loss_bluff

        # Backpropagation
        self.optimizer.zero_grad()
        total_aux_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.belief_head.parameters()) + list(extractor.parameters()), 1.0)
        self.optimizer.step()

        if self.verbose > 0:
            print(f"  [Auxiliary Belief] Rollout update: Loss={total_aux_loss.item():.4f} (Cat={loss_cat.item():.4f}, Bluff={loss_bluff.item():.4f})")
