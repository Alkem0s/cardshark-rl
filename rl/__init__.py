"""
rl package — Reinforcement Learning & Neural Modules for CardShark-RL.

Contains Gymnasium wrappers, Bayesian opponent tracking, Self-Attention feature
extractors, and League fictitious-play sparring.
"""

from rl.gym_wrapper import DrawPokerGymEnv, make_env, mask_fn
from rl.multi_gym_wrapper import (
    MultiDrawPokerGymEnv,
    make_multi_env,
    make_superhuman_multi_env,
    multi_mask_fn,
    OBS_DIM,
    SUPERHUMAN_OBS_DIM,
    build_observation_vector,
)
from rl.opponent_tracker import (
    TableOpponentTracker,
    SeatProfile,
    BetaBinomialTracker,
    DecayingBetaBinomialTracker,
)
import sys
import rl.card_attention as _ca
sys.modules.setdefault("card_attention", _ca)

from rl.card_attention import CardAttentionExtractor
from rl.league import LeaguePool, LeagueOpponent, get_cached_model

__all__ = [
    "DrawPokerGymEnv",
    "make_env",
    "mask_fn",
    "MultiDrawPokerGymEnv",
    "make_multi_env",
    "make_superhuman_multi_env",
    "multi_mask_fn",
    "OBS_DIM",
    "SUPERHUMAN_OBS_DIM",
    "build_observation_vector",
    "TableOpponentTracker",
    "SeatProfile",
    "BetaBinomialTracker",
    "DecayingBetaBinomialTracker",
    "CardAttentionExtractor",
    "LeaguePool",
    "LeagueOpponent",
    "get_cached_model",
]
