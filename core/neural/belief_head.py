"""
core/neural/belief_head.py — Auxiliary Supervised Belief Head & Representation Learning.
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

    def __init__(self, feature_dim: int = 216, hidden_dim: int = 128, num_categories: int = 9):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_categories = num_categories

        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.category_head = nn.Linear(hidden_dim, num_categories)
        self.bluff_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(features)
        cat_logits = self.category_head(h)
        bluff_logits = self.bluff_head(h).squeeze(-1)
        return cat_logits, bluff_logits


class AuxiliaryBeliefCallback(BaseCallback):
    """
    Rollout callback capturing privileged opponent targets and executing
    self-supervised representation learning backprop through CardAttentionExtractor.
    """

    def __init__(
        self,
        belief_head: AuxiliaryBeliefHead,
        lr: float = 1e-4,
        bluff_weight: float = 0.5,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.belief_head = belief_head
        self.lr = lr
        self.bluff_weight = bluff_weight
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.device = torch.device("cpu")

        self.recorded_obs: List[np.ndarray] = []
        self.recorded_cats: List[int] = []
        self.recorded_bluffs: List[float] = []

    def _on_training_start(self) -> None:
        self.device = self.model.device
        self.belief_head.to(self.device)

        features_extractor = self.model.policy.features_extractor
        params = list(self.belief_head.parameters()) + list(features_extractor.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        obs_tensor = self.locals.get("new_obs", None)

        if obs_tensor is not None and len(infos) > 0:
            if isinstance(obs_tensor, torch.Tensor):
                obs_np = obs_tensor.detach().cpu().numpy()
            else:
                obs_np = np.array(obs_tensor)

            for i, info in enumerate(infos):
                if "privileged_opp_cat" in info:
                    opp_cat = int(info["privileged_opp_cat"])
                    is_bluff = float(info.get("privileged_is_bluff", 0.0))
                    self.recorded_obs.append(obs_np[i])
                    self.recorded_cats.append(opp_cat)
                    self.recorded_bluffs.append(is_bluff)

        return True

    def _on_rollout_end(self) -> None:
        if len(self.recorded_obs) < 32 or self.optimizer is None:
            self.recorded_obs.clear()
            self.recorded_cats.clear()
            self.recorded_bluffs.clear()
            return

        obs_t = torch.tensor(np.array(self.recorded_obs), dtype=torch.float32, device=self.device)
        cat_t = torch.tensor(np.array(self.recorded_cats), dtype=torch.long, device=self.device)
        bluff_t = torch.tensor(np.array(self.recorded_bluffs), dtype=torch.float32, device=self.device)

        features = self.model.policy.features_extractor(obs_t)
        cat_logits, bluff_logits = self.belief_head(features)

        loss_cat = F.cross_entropy(cat_logits, cat_t)
        loss_bluff = F.binary_cross_entropy_with_logits(bluff_logits, bluff_t)
        total_loss = loss_cat + self.bluff_weight * loss_bluff

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.belief_head.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.model.policy.features_extractor.parameters(), max_norm=0.5)
        self.optimizer.step()

        if self.verbose > 0:
            print(f"  [Auxiliary Belief] Rollout update: Loss={total_loss.item():.4f} (Cat={loss_cat.item():.4f}, Bluff={loss_bluff.item():.4f})")

        self.recorded_obs.clear()
        self.recorded_cats.clear()
        self.recorded_bluffs.clear()
