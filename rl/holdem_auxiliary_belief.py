"""
rl/holdem_auxiliary_belief.py — Auxiliary Supervised Belief Head & Representation Learning for Texas Hold'em Model D.

Key Capabilities:
1. HoldemAuxiliaryBeliefHead:
   Evaluates latent Card Attention / Policy features to predict:
   a) Primary adversary archetype (4 classes: TAG, LAG, Station, Random)
   b) Primary adversary hand strength tier (4 classes: Air, Weak Pair, Top Pair, Monster)
   c) Adversary fold probability (binary logit: whether adversary folds)
2. HoldemAuxiliaryBeliefCallback:
   - Captures privileged environment targets (adv_archetype, adv_hand_tier, adv_folded) during vectorized rollouts.
   - Executes auxiliary cross-entropy backpropagation directly through the Card Attention backbone
     at the conclusion of each rollout buffer.
3. Representation Shaping:
   Forces the Transformer attention heads to internalize villain behaviors and range textures
   without hardcoding brittle decision heuristics.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Any

from stable_baselines3.common.callbacks import BaseCallback


class HoldemAuxiliaryBeliefHead(nn.Module):
    """
    Auxiliary prediction network predicting adversary archetype, hand strength tier,
    and fold probability from latent representation features.
    """

    def __init__(self, feature_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.archetype_head = nn.Linear(hidden_dim, 4)
        self.hand_tier_head = nn.Linear(hidden_dim, 4)
        self.fold_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            arch_logits: (B, 4) logits for adversary archetype [TAG, LAG, Station, Random]
            tier_logits: (B, 4) logits for adversary hand tier [Air, Weak Pair, Top Pair, Monster]
            fold_logits: (B,) logit for binary fold prediction
        """
        h = self.trunk(features)
        arch_logits = self.archetype_head(h)
        tier_logits = self.hand_tier_head(h)
        fold_logits = self.fold_head(h).squeeze(-1)
        return arch_logits, tier_logits, fold_logits


class HoldemAuxiliaryBeliefCallback(BaseCallback):
    """
    Rollout callback that collects privileged environment targets during training
    and backpropagates auxiliary belief losses into the Card Attention feature extractor.
    """

    def __init__(
        self,
        belief_head: HoldemAuxiliaryBeliefHead,
        lr: float = 1.0e-4,
        tier_weight: float = 0.5,
        fold_weight: float = 0.5,
        max_batch_size: int = 2048,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.belief_head = belief_head
        self.lr = lr
        self.tier_weight = tier_weight
        self.fold_weight = fold_weight
        self.max_batch_size = max_batch_size

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._obs_buffer: List[np.ndarray] = []
        self._arch_buffer: List[int] = []
        self._tier_buffer: List[int] = []
        self._fold_buffer: List[float] = []

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
                if "adv_archetype" in info and "adv_hand_tier" in info and "adv_folded" in info:
                    self._obs_buffer.append(new_obs[i])
                    self._arch_buffer.append(int(info["adv_archetype"]))
                    self._tier_buffer.append(int(info["adv_hand_tier"]))
                    self._fold_buffer.append(float(info["adv_folded"]))

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
            arch_batch = np.array([self._arch_buffer[idx] for idx in indices], dtype=np.int64)
            tier_batch = np.array([self._tier_buffer[idx] for idx in indices], dtype=np.int64)
            fold_batch = np.array([self._fold_buffer[idx] for idx in indices], dtype=np.float32)
        else:
            obs_batch = np.array(self._obs_buffer, dtype=np.float32)
            arch_batch = np.array(self._arch_buffer, dtype=np.int64)
            tier_batch = np.array(self._tier_buffer, dtype=np.int64)
            fold_batch = np.array(self._fold_buffer, dtype=np.float32)

        # Clear buffer for next rollout
        self._obs_buffer.clear()
        self._arch_buffer.clear()
        self._tier_buffer.clear()
        self._fold_buffer.clear()

        # Convert to PyTorch tensors
        obs_tensor = torch.as_tensor(obs_batch, device=device)
        arch_tensor = torch.as_tensor(arch_batch, device=device)
        tier_tensor = torch.as_tensor(tier_batch, device=device)
        fold_tensor = torch.as_tensor(fold_batch, device=device)

        # Forward pass through shared features_extractor
        extractor = self.model.policy.features_extractor
        extractor.train()
        self.belief_head.train()

        features = extractor(obs_tensor)
        arch_logits, tier_logits, fold_logits = self.belief_head(features)

        # Compute multi-task losses
        loss_arch = F.cross_entropy(arch_logits, arch_tensor)
        loss_tier = F.cross_entropy(tier_logits, tier_tensor)
        loss_fold = F.binary_cross_entropy_with_logits(fold_logits, fold_tensor)
        total_aux_loss = loss_arch + self.tier_weight * loss_tier + self.fold_weight * loss_fold

        # Backpropagation into shared feature representation
        self.optimizer.zero_grad()
        total_aux_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.belief_head.parameters()) + list(extractor.parameters()), 1.0
        )
        self.optimizer.step()

        if self.verbose > 0:
            print(
                f"  [Holdem Belief] Rollout update: Total={total_aux_loss.item():.4f} "
                f"(Arch={loss_arch.item():.4f}, Tier={loss_tier.item():.4f}, Fold={loss_fold.item():.4f})"
            )
