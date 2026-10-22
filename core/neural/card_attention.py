"""
core/neural/card_attention.py — Permutation-Invariant Multi-Head Attention Card Extractor.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CardAttentionExtractor(BaseFeaturesExtractor):
    """
    Permutation-invariant PyTorch feature extractor for card games.
    
    Extracts the private cards, tokenizes them, processes them via Multi-Head Self-Attention,
    and pools them into an invariant hand representation before fusing with table features.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        num_cards: int = 5,
        embed_dim: int = 32,
        num_heads: int = 4,
        pooling: str = "attention",
        card_features_dim: int = 32,
        game_features_dim: int = 64,
        slot_features_dim: int = 24,
    ):
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        assert pooling in ("mean", "max", "attention", "both"), f"Unknown pooling method: {pooling}"

        self.num_cards = num_cards
        self.card_in_dim = 2  # [rank, suit]
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.pooling = pooling
        self.card_features_dim = card_features_dim
        self.game_features_dim = game_features_dim
        self.slot_features_dim = slot_features_dim

        total_slot_dim = self.num_cards * slot_features_dim if slot_features_dim > 0 else 0
        features_dim = card_features_dim + total_slot_dim + game_features_dim
        super().__init__(observation_space, features_dim=features_dim)

        # 1. Token Projection: R^2 -> R^embed_dim
        self.token_proj = nn.Sequential(
            nn.Linear(self.card_in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 2. Multi-Head Self-Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # Feed-forward network on tokens
        self.token_ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.ff_norm = nn.LayerNorm(embed_dim)

        # 3. Pooling Mechanisms
        if pooling == "attention":
            self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.pool_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=1,
                batch_first=True,
            )
            self.pool_norm = nn.LayerNorm(embed_dim)
            pool_out_dim = embed_dim
        elif pooling == "both":
            pool_out_dim = embed_dim * 2
        else:
            pool_out_dim = embed_dim

        # Card projection: pool_out_dim -> card_features_dim
        self.card_proj = nn.Sequential(
            nn.Linear(pool_out_dim, card_features_dim),
            nn.LayerNorm(card_features_dim),
            nn.ReLU(),
        )

        # Slot-preserving projection
        if slot_features_dim > 0:
            self.slot_proj = nn.Sequential(
                nn.Linear(embed_dim, slot_features_dim),
                nn.LayerNorm(slot_features_dim),
                nn.ReLU(),
            )
        else:
            self.slot_proj = None

        # 4. Table Context Network
        raw_obs_dim = int(observation_space.shape[0])
        game_in_dim = raw_obs_dim - (self.num_cards * self.card_in_dim)
        assert game_in_dim > 0, f"Observation space ({raw_obs_dim}) must be larger than card dims (10)"

        self.game_net = nn.Sequential(
            nn.Linear(game_in_dim, game_features_dim),
            nn.LayerNorm(game_features_dim),
            nn.ReLU(),
            nn.Linear(game_features_dim, game_features_dim),
            nn.LayerNorm(game_features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        card_raw = observations[:, : self.num_cards * self.card_in_dim]
        game_raw = observations[:, self.num_cards * self.card_in_dim :]

        B = observations.shape[0]
        tokens = card_raw.view(B, self.num_cards, self.card_in_dim)
        tokens = self.token_proj(tokens)

        # Self-attention over card tokens
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.attn_norm(tokens + attn_out)

        # Token FFN
        ff_out = self.token_ff(tokens)
        tokens = self.ff_norm(tokens + ff_out)

        # Pooling
        if self.pooling == "mean":
            pooled = tokens.mean(dim=1)
        elif self.pooling == "max":
            pooled = tokens.max(dim=1).values
        elif self.pooling == "both":
            p_mean = tokens.mean(dim=1)
            p_max = tokens.max(dim=1).values
            pooled = torch.cat([p_mean, p_max], dim=-1)
        elif self.pooling == "attention":
            q = self.query.expand(B, -1, -1)
            pool_attn_out, _ = self.pool_attn(q, tokens, tokens)
            pooled = self.pool_norm(q + pool_attn_out).squeeze(1)

        card_feat = self.card_proj(pooled)

        parts = [card_feat]
        if self.slot_proj is not None:
            slot_feats = self.slot_proj(tokens).view(B, -1)
            parts.append(slot_feats)

        game_feat = self.game_net(game_raw)
        parts.append(game_feat)

        return torch.cat(parts, dim=-1)
