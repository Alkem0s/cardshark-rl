"""
rl/holdem_decoupled_actor.py — Decoupled Hierarchical Action-Sizing Policy for Texas Hold'em Model D.

Key Innovation:
In No-Limit Texas Hold'em, flat-logit softmax policies cause severe bet-sizing starvation
(e.g., Half-Pot was chosen 0.0% of the time in Model B/C tournament benchmarks).

HoldemDecoupledActionNet eliminates this failure mode by decoupling decision making into:
1. Core Action Stream: Decides strategic intent among 4 high-level choices:
   - Fold (Action 0)
   - Check / Call (Action 1)
   - Raise Base (Intent to bet / raise)
   - All-In Shove (Action 5)
2. Sizing Stream: Specializes exclusively in geometric sizing allocation:
   - Min-Raise (Action 2)
   - Half-Pot (Action 3)
   - Full-Pot (Action 4)

Composed Logits:
- Logit[0] = Fold
- Logit[1] = Check/Call
- Logit[2] = Raise Base + Min-Raise Sizing
- Logit[3] = Raise Base + Half-Pot Sizing
- Logit[4] = Raise Base + Full-Pot Sizing
- Logit[5] = All-In Shove

100% Drop-in Compatibility:
- Subclasses MaskableActorCriticPolicy.
- Fully compatible with MaskablePPO, action masks, model saving, loading, and evaluation.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Type, Any, Tuple, Union

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule


class HoldemDecoupledActionNet(nn.Module):
    """
    Decoupled Hierarchical Action Head for Texas Hold'em.
    Separates high-level strategic action choice (Fold, Call, Raise, All-In)
    from fine-grained bet sizing.
    Supports both 6-action (Legacy Models D & E: Min, 50%, 100%)
    and 8-action (Model F: Min, 33%, 50%, 100%, 150%).
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        action_dim: int = 6,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        num_sizings = 5 if action_dim == 8 else 3

        # Core Action Stream: Fold (0), Call (1), Raise Base (2), All-In (3)
        self.core_stream = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

        # Bet Sizing Stream:
        # If action_dim == 6: Min-Raise (0), Half-Pot (1), Full-Pot (2)
        # If action_dim == 8: Min-Raise (0), Third-Pot (1), Half-Pot (2), Full-Pot (3), Overbet (4)
        self.sizing_stream = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_sizings),
        )

    def forward(self, latent_pi: torch.Tensor) -> torch.Tensor:
        """
        Forward pass composing 4 core logits and sizing logits into unified (B, action_dim) tensor.
        """
        core = self.core_stream(latent_pi)      # (B, 4)
        sizing = self.sizing_stream(latent_pi)  # (B, 3) or (B, 5)

        f_logit = core[:, 0:1]  # Fold (action 0)
        c_logit = core[:, 1:2]  # Check/Call (action 1)
        r_base = core[:, 2:3]   # Raise Base
        a_logit = core[:, 3:4]  # All-In

        if self.action_dim == 8:
            r_min = r_base + sizing[:, 0:1]    # Min-Raise (action 2)
            r_third = r_base + sizing[:, 1:2]  # 1/3-Pot (action 3)
            r_half = r_base + sizing[:, 2:3]   # Half-Pot (action 4)
            r_pot = r_base + sizing[:, 3:4]    # Full-Pot (action 5)
            r_over = r_base + sizing[:, 4:5]   # Overbet (action 6)
            return torch.cat([f_logit, c_logit, r_min, r_third, r_half, r_pot, r_over, a_logit], dim=-1)  # (B, 8)
        else:
            r_min = r_base + sizing[:, 0:1]   # Min-Raise (action 2)
            r_half = r_base + sizing[:, 1:2]  # Half-Pot (action 3)
            r_pot = r_base + sizing[:, 2:3]   # Full-Pot (action 4)
            return torch.cat([f_logit, c_logit, r_min, r_half, r_pot, a_logit], dim=-1)  # (B, 6)


class HoldemDecoupledActorCriticPolicy(MaskableActorCriticPolicy):
    """
    MaskableActorCriticPolicy with HoldemDecoupledActionNet for Texas Hold'em Models D, E, F.
    """

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        # Build decoupled hierarchical action head (respecting discrete action space dimensionality)
        action_dim = int(self.action_space.n) if hasattr(self.action_space, "n") else 6
        self.action_net = HoldemDecoupledActionNet(
            latent_dim=self.mlp_extractor.latent_dim_pi,
            hidden_dim=128,
            action_dim=action_dim,
        )
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        # Weight initialization
        if self.ortho_init:
            module_gains = {
                self.mlp_extractor: np.sqrt(2),
                self.action_net.core_stream: 0.01,
                self.action_net.sizing_stream: 0.01,
                self.value_net: 1.0,
            }
            for module, gain in module_gains.items():
                if module is not None:
                    for name, param in module.named_parameters():
                        if "bias" in name:
                            nn.init.constant_(param, 0.0)
                        elif "weight" in name:
                            if len(param.shape) >= 2:
                                nn.init.orthogonal_(param, gain=gain)

        # Init optimizer
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
