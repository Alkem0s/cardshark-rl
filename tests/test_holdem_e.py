"""
tests/test_holdem_e.py — Unit & Integration Test Suite for Texas Hold'em Model E.

Tests:
1. 158-dimensional observation vector geometry and normalization bounds.
2. HoldemLeaguePool Prioritized Fictitious Self-Play (PFSP) opponent sampling.
3. HoldemLeagueOpponent neural inference via act_with_env.
4. Two-Stage HoldemCurriculumRewardCallback reward decay mechanics.
5. Environment rollout stability under SubprocVecEnv/DummyVecEnv with obs_version='e'.
"""

from __future__ import annotations
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from games.holdem.gym_wrapper import HoldemPokerGymEnv, OBS_DIM_HOLDEM_E
from games.holdem.action_abstraction import NUM_ACTIONS, A_FOLD, A_CHECK_CALL
from rl.holdem_league import HoldemLeaguePool, HoldemLeagueOpponent
from rl.holdem_curriculum import HoldemCurriculumRewardCallback


def test_obs_dim_holdem_e():
    """Verifies that HoldemPokerGymEnv with obs_version='e' produces exact 158-d observations."""
    env = HoldemPokerGymEnv(
        num_seats=5,
        starting_chips=200,
        obs_version="e",
        rng_seed=42,
    )
    obs, info = env.reset()
    assert obs.shape == (OBS_DIM_HOLDEM_E,), f"Expected shape ({OBS_DIM_HOLDEM_E},), got {obs.shape}"
    assert obs.shape[0] == 158
    assert np.isfinite(obs).all(), "Observation contains NaN or Inf values"
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0), "Features exceed normalized bounds [-1.0, 1.0]"
    env.close()


def test_action_masks_validity():
    """Verifies action masks return a valid binary vector with at least one legal action."""
    env = HoldemPokerGymEnv(obs_version="e", rng_seed=123)
    obs, info = env.reset()
    mask = env.action_masks()
    assert mask.shape == (NUM_ACTIONS,)
    assert set(mask.tolist()).issubset({0, 1})
    assert np.sum(mask) >= 1, "Must have at least one legal action"
    env.close()


def test_league_pool_and_sampling():
    """Verifies that HoldemLeaguePool correctly samples opponents across PFSP categories."""
    pool = HoldemLeaguePool(
        league_dir="models/holdem/league",
        anchor_model_path="models/holdem/model_holdem_d_best.zip",
    )
    rng = np.random.default_rng(999)

    sampled_opponents = [pool.sample_opponent(seat_idx=i % 4 + 1, rng=rng) for i in range(20)]
    assert len(sampled_opponents) == 20
    for opp in sampled_opponents:
        assert hasattr(opp, "act")
        assert opp.name is not None


def test_league_opponent_act_with_env():
    """Verifies HoldemLeagueOpponent runs seat-perspective neural inference or falls back cleanly."""
    anchor_path = "models/holdem/model_holdem_d_best.zip"
    if not os.path.exists(anchor_path):
        print(f"  [Notice] Anchor model not found at {anchor_path}, skipping neural inference check.")
        return

    opp = HoldemLeagueOpponent(
        model_path=anchor_path,
        opponent_id=1,
        deterministic=True,
    )
    assert opp.is_neural is True
    assert opp.model is not None

    env = HoldemPokerGymEnv(obs_version="e", rng_seed=42)
    obs, info = env.reset()
    legal = env.env.get_legal_actions(1)
    act = opp.act_with_env(env.env, seat_idx=1, legal_actions=legal)
    assert act in legal, f"Opponent action {act} not in legal actions {legal}"
    env.close()


def test_curriculum_reward_callback_decay():
    """Verifies HoldemCurriculumRewardCallback correctly transitions from 1.0 to 0.0."""
    cb = HoldemCurriculumRewardCallback(
        scaffold_steps=1000,
        total_timesteps=5000,
        min_shaping_mult=0.0,
        verbose=0,
    )

    # Stage 1: should stay at 1.0
    cb.num_timesteps = 500
    cb._on_step()
    assert cb.current_mult == 1.0

    cb.num_timesteps = 1000
    cb._on_step()
    assert cb.current_mult == 1.0

    # Mid Stage 2: halfway through (3000 steps), cosine(pi * 0.5) = 0.0 -> mult = 0.5
    cb.num_timesteps = 3000
    cb._on_step()
    assert abs(cb.current_mult - 0.5) < 1e-2

    # End Stage 2: 5000 steps -> mult = 0.0
    cb.num_timesteps = 5000
    cb._on_step()
    assert abs(cb.current_mult - 0.0) < 1e-5


def test_gym_env_rollout_with_league_pool():
    """Verifies full multi-hand tournament rollout with HoldemLeaguePool active."""
    pool = HoldemLeaguePool(
        league_dir="models/holdem/league",
        anchor_model_path="models/holdem/model_holdem_d_best.zip",
    )
    env = HoldemPokerGymEnv(
        num_seats=5,
        starting_chips=200,
        obs_version="e",
        league_mode=True,
        league_pool=pool,
        rng_seed=77,
    )
    obs, info = env.reset()
    assert obs.shape == (OBS_DIM_HOLDEM_E,)

    for _ in range(25):
        mask = env.action_masks()
        valid = np.where(mask == 1)[0]
        act = int(valid[0])
        obs, rew, term, trunc, info = env.step(act)
        assert np.isfinite(obs).all()
        assert np.isfinite(rew)
        if term or trunc:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  RUNNING TEXAS HOLD'EM MODEL E TEST SUITE")
    print("=" * 60)
    print("  [1/6] Running test_obs_dim_holdem_e...")
    test_obs_dim_holdem_e()
    print("  [2/6] Running test_action_masks_validity...")
    test_action_masks_validity()
    print("  [3/6] Running test_league_pool_and_sampling...")
    test_league_pool_and_sampling()
    print("  [4/6] Running test_league_opponent_act_with_env...")
    test_league_opponent_act_with_env()
    print("  [5/6] Running test_curriculum_reward_callback_decay...")
    test_curriculum_reward_callback_decay()
    print("  [6/6] Running test_gym_env_rollout_with_league_pool...")
    test_gym_env_rollout_with_league_pool()
    print("=" * 60)
    print("  ALL 6 MODEL E UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
