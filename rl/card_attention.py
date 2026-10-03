"""
card_attention.py — Self-Attention Permutation-Invariant Card Extractor for Model D.

Mathematical Foundation:
In 5-Card Draw Poker, the Hero hand is an unordered multiset of 5 cards:
    Hand = {c_1, c_2, c_3, c_4, c_5}, where c_i = [rank_norm, suit_norm] in [0, 1]^2.

Standard MLPs treat input dimensions as positional indices, forcing the network to waste
millions of gradient steps memorizing that permutations sigma(Hand) represent the exact same hand.

CardAttentionExtractor achieves strict permutation invariance:
1. Token Projection: Each card c_i is linearly projected to an embedding token of dim d_embed.
2. Self-Attention (Permutation Equivariant): Multi-Head Self-Attention without positional
   encoding computes intra-hand card interactions (pairs, flushes, kickers).
3. Permutation-Invariant Pooling: Aggregates the 5 card tokens via Global Mean, Max,
   or Learnable Query Attention Pooling:
       Pool(sigma(H)) = Pool(H) for all permutations sigma in S_5.
4. Game State Fusion: The invariant card embedding is concatenated with table state,
   opponent tracker statistics, and intra-hand sequence history (77 dims).
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
    Permutation-invariant PyTorch feature extractor for CardShark-RL Model D.
    
    Extracts the 5 private cards (first 10 dims of observation), tokenizes them,
    processes them via Multi-Head Self-Attention, and pools them into an invariant
    hand representation before fusing with table and opponent features.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        embed_dim: int = 32,
        num_heads: int = 2,
        pooling: str = "mean",
        card_features_dim: int = 32,
        game_features_dim: int = 64,
        slot_features_dim: int = 0,
    ):
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        assert pooling in ("mean", "max", "attention", "both"), f"Unknown pooling method: {pooling}"

        self.num_cards = 5
        self.card_in_dim = 2  # [rank, suit]
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.pooling = pooling
        self.card_features_dim = card_features_dim
        self.game_features_dim = game_features_dim
        self.slot_features_dim = slot_features_dim

        # Total feature dimensions:
        # 1. Global pooled card representation (for betting / hand strength)
        # 2. Slot-preserving representations (5 * slot_features_dim for discard decision indexing)
        # 3. Game / Table / Opponent representation
        total_slot_dim = self.num_cards * slot_features_dim if slot_features_dim > 0 else 0
        features_dim = card_features_dim + total_slot_dim + game_features_dim
        super().__init__(observation_space, features_dim=features_dim)

        # 1. Token Projection: R^2 -> R^embed_dim
        self.card_proj = nn.Sequential(
            nn.Linear(self.card_in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # 2. Multi-Head Self-Attention (permutation-equivariant across the 5 slots)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # 3. Feed-Forward Token Refinement
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # 4. Permutation-Invariant Global Pooling (Hand Context for Betting Actions)
        if pooling == "attention":
            self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        elif pooling == "both":
            self.pool_proj = nn.Linear(embed_dim * 2, embed_dim)

        self.card_out = nn.Sequential(
            nn.Linear(embed_dim, card_features_dim),
            nn.LayerNorm(card_features_dim),
            nn.GELU(),
        )

        # 5. Slot-Preserving Refinement (Per-Card Attention Context for 32-Discard Bitmask Decisions)
        if slot_features_dim > 0:
            self.slot_out = nn.Sequential(
                nn.Linear(embed_dim, slot_features_dim),
                nn.LayerNorm(slot_features_dim),
                nn.GELU(),
            )
        else:
            self.slot_out = None

        # 6. Non-Card Table & Opponent State Projection
        obs_dim = observation_space.shape[0]
        self.game_in_dim = obs_dim - (self.num_cards * self.card_in_dim)  # e.g., 87 - 10 = 77
        self.game_proj = nn.Sequential(
            nn.Linear(self.game_in_dim, game_features_dim),
            nn.LayerNorm(game_features_dim),
            nn.GELU(),
        )

    def extract_cards_and_game(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Separates the 5-card tokens from the table / opponent state vector."""
        card_raw = observations[:, : self.num_cards * self.card_in_dim]
        game_raw = observations[:, self.num_cards * self.card_in_dim :]
        return card_raw, game_raw

    def forward_card_tokens(self, card_raw: torch.Tensor) -> torch.Tensor:
        """Processes 5 raw card tokens through projection and multi-head self-attention."""
        batch_size = card_raw.size(0)
        cards = card_raw.view(batch_size, self.num_cards, self.card_in_dim)

        # 1. Project tokens
        h_cards = self.card_proj(cards)  # (B, 5, embed_dim)

        # 2. Self-Attention with residual connection & LayerNorm
        attn_out, _ = self.mha(h_cards, h_cards, h_cards)
        h_cards = self.norm1(h_cards + attn_out)

        # 3. Position-wise Feedforward Network
        ff_out = self.ffn(h_cards)
        h_cards = self.norm2(h_cards + ff_out)  # (B, 5, embed_dim)
        return h_cards

    def pool_hand(self, h_cards: torch.Tensor) -> torch.Tensor:
        """Computes permutation-invariant global hand representation."""
        batch_size = h_cards.size(0)
        if self.pooling == "mean":
            return h_cards.mean(dim=1)
        elif self.pooling == "max":
            pooled, _ = h_cards.max(dim=1)
            return pooled
        elif self.pooling == "attention":
            q = self.query.expand(batch_size, -1, -1)
            scores = torch.bmm(q, h_cards.transpose(1, 2)) / math.sqrt(self.embed_dim)
            weights = torch.softmax(scores, dim=-1)
            return torch.bmm(weights, h_cards).squeeze(1)
        elif self.pooling == "both":
            mean_p = h_cards.mean(dim=1)
            max_p, _ = h_cards.max(dim=1)
            return self.pool_proj(torch.cat([mean_p, max_p], dim=-1))
        return h_cards.mean(dim=1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        card_raw, game_raw = self.extract_cards_and_game(observations)
        batch_size = card_raw.size(0)

        # 1. Contextualized token representations via Multi-Head Self-Attention
        h_cards = self.forward_card_tokens(card_raw)  # (B, 5, embed_dim)

        # 2. Global invariant pooling for betting context
        pooled = self.pool_hand(h_cards)
        global_card_feats = self.card_out(pooled)  # (B, card_features_dim)

        # 3. Table and opponent context
        game_feats = self.game_proj(game_raw)  # (B, game_features_dim)

        # 4. Slot-preserving representations for discard decisions
        if self.slot_out is not None:
            h_slots = self.slot_out(h_cards)  # (B, 5, slot_features_dim)
            slot_feats = h_slots.reshape(batch_size, -1)  # (B, 5 * slot_features_dim)
            return torch.cat([global_card_feats, slot_feats, game_feats], dim=-1)
        else:
            return torch.cat([global_card_feats, game_feats], dim=-1)

