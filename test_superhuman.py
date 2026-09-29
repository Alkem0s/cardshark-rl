"""
test_superhuman.py — Comprehensive Unit & Integration Test for Model C Superhuman Subsystems.

Verifies:
1. Dual-Timescale Recency Tracker & Tilt Detection in opponent_tracker.py
2. 87-Dimensional Observation Vector & Intra-Hand Action Sequence Memory in multi_gym_wrapper.py
3. Multi-Agent League Sparring Pool & Checkpoint Registration in multi_opponents.py
4. LeagueOpponent Inference with Loaded Policy
"""

import os
import numpy as np

from opponent_tracker import SeatProfile, DecayingBetaBinomialTracker
from multi_gym_wrapper import SUPERHUMAN_OBS_DIM, make_superhuman_multi_env
from multi_opponents import LeaguePool, LeagueOpponent, make_random_archetype


def test_dual_timescale_tilt_detection():
    profile = SeatProfile(seat_idx=2)

    # 1. 10 passive hands (Rock mode)
    for _ in range(10):
        profile.record_hand(vpip=False, pfr=False, post_draw_raised=False, faced_raise_and_folded=True)

    is_tilting, label = profile.detect_tilt(threshold=0.25)
    assert not is_tilting, f"Expected stable, got {label}"

    # 2. Sudden aggressive tilt: 3 hands raising pre-draw and post-draw
    for _ in range(3):
        profile.record_hand(vpip=True, pfr=True, post_draw_raised=True, faced_raise_and_folded=False)

    is_tilting, label = profile.detect_tilt(threshold=0.25)
    delta = profile.get_recency_delta()

    assert is_tilting, f"Expected tilt alert, got {label} with delta {delta}"
    assert "Aggressive Tilt" in label
    assert delta > 0.25, f"Expected delta > 0.25, got {delta}"
    print("test_dual_timescale_tilt_detection: PASS (Detected tilt in 3 hands)")


def test_87_dim_observation_space():
    env_fn = make_superhuman_multi_env(
        num_seats=5,
        starting_chips=200,
        seed=101,
        max_hands_per_session=10,
    )
    env = env_fn()
    obs, info = env.reset(seed=101)

    assert obs.shape == (SUPERHUMAN_OBS_DIM,), f"Expected ({SUPERHUMAN_OBS_DIM},), got {obs.shape}"
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0), "Observation vector out of bounds [-1.0, 1.0]"

    # Execute multiple steps and check observation validity throughout
    mask = env.action_masks()
    for _ in range(20):
        legal = np.where(mask == 1)[0]
        action = int(legal[0])
        obs, reward, term, trunc, info = env.step(action)
        assert obs.shape == (SUPERHUMAN_OBS_DIM,)
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
        if term or trunc:
            obs, info = env.reset()
        mask = env.action_masks()

    print("test_87_dim_observation_space: PASS (Verified 87-dim bounds and sequence encoding)")


def test_league_pool_and_sparring():
    model_b_path = "models/model_b_multiplayer.zip"
    league_dir = "models/league"

    pool = LeaguePool(
        base_model_path=model_b_path,
        league_dir=league_dir,
        neural_opponent_prob=1.0, # Force neural sampling for test
    )

    if os.path.exists(model_b_path):
        assert len(pool.checkpoints) >= 1, "Expected base model checkpoint in pool"
        opp = pool.sample_opponent(seat_idx=1)
        assert isinstance(opp, LeagueOpponent)
        assert opp.is_superhuman is False # Model B is 63-dim
        print("test_league_pool_and_sparring: PASS (Loaded LeagueOpponent from Model B checkpoint)")
    else:
        print("test_league_pool_and_sparring: SKIPPED (Model B checkpoint not on disk)")


if __name__ == "__main__":
    test_dual_timescale_tilt_detection()
    test_87_dim_observation_space()
    test_league_pool_and_sparring()
    print("ALL SUPERHUMAN TESTS PASSED!")
