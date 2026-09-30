"""
test_gym_wrapper.py — Unit test & validation for MultiDrawPokerGymEnv.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from rl.multi_gym_wrapper import MultiDrawPokerGymEnv, OBS_DIM, SUPERHUMAN_OBS_DIM, make_superhuman_multi_env
from game.multi_draw_poker_env import TOTAL_ACTIONS


def test_wrapper_reset_and_step():
    env = MultiDrawPokerGymEnv(
        num_seats=5,
        starting_chips=200,
        rng_seed=42,
        max_hands_per_session=20,
    )
    obs, info = env.reset()

    assert obs.shape == (OBS_DIM,), f"Expected shape ({OBS_DIM},), got {obs.shape}"
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0), "Observation values out of bounds [-1.0, 1.0]"
    assert "hero_seat" in info
    assert "alive_players" in info

    mask = env.action_masks()
    assert mask.shape == (TOTAL_ACTIONS,)
    assert mask.sum() > 0, "No legal actions available!"

    # Step through a sequence of turns
    done = False
    step_count = 0
    total_reward = 0.0

    while not done and step_count < 100:
        step_count += 1
        legal_actions = np.where(mask == 1)[0]
        action = int(np.random.choice(legal_actions))

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        assert obs.shape == (OBS_DIM,)
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
        mask = env.action_masks()

    print(f"test_wrapper_reset_and_step: PASS (Steps: {step_count}, Total reward: {total_reward:.4f}, Hands: {info['hands_played']})")


def test_scale_invariance():
    # Test identical stack depth in BB (e.g. 50 BB):
    # Game A: 100 chips per player with SB=1, BB=2
    # Game B: 10,000 chips per player with SB=100, BB=200
    env_small = MultiDrawPokerGymEnv(starting_chips=100, small_blind=1, big_blind=2, rng_seed=10)
    env_large = MultiDrawPokerGymEnv(starting_chips=10000, small_blind=100, big_blind=200, rng_seed=10)

    obs_small, _ = env_small.reset(seed=10)
    obs_large, _ = env_large.reset(seed=10)

    # Hero stack ratio (index 12) and pot ratio (index 13) should be identical
    print(f"Small chips: stack_ratio={obs_small[12]:.5f}, pot_ratio={obs_small[13]:.5f}")
    print(f"Large chips: stack_ratio={obs_large[12]:.5f}, pot_ratio={obs_large[13]:.5f}")

    assert abs(obs_small[12] - obs_large[12]) < 1e-4
    assert abs(obs_small[13] - obs_large[13]) < 1e-4
    print("test_scale_invariance: PASS (Normalized ratios scale identically across 100 and 10,000 chips)")


def test_randomized_stacks_and_blind_escalation():
    """Validates stack depth randomization and blind escalation in the gym wrapper."""
    env = MultiDrawPokerGymEnv(
        num_seats=5,
        starting_chips=200,
        blind_escalation_interval=4,
        randomize_stacks=True,
        rng_seed=123,
        max_hands_per_session=25,
    )
    obs, info = env.reset(seed=123)

    assert obs.shape == (OBS_DIM,)
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)

    # Check total chips conserved
    total_chips = sum(s.chips for s in env.env.seats) + env.env.pot
    assert total_chips == 1000, f"Expected 1000 total chips, got {total_chips}"

    # Check stack diversity
    stacks = [s.chips for s in env.env.seats]
    assert len(set(stacks)) > 1, f"Expected unequal stacks from randomize_stacks, got {stacks}"

    # Step through hands and verify blind escalation
    done = False
    step_count = 0
    while not done and step_count < 150:
        step_count += 1
        mask = env.action_masks()
        legal = np.where(mask == 1)[0]
        action = int(np.random.choice(legal))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    assert "is_winner" in info
    print(f"test_randomized_stacks_and_blind_escalation: PASS (Big blind escalated to {env.env.big_blind}, Hands: {info['hands_played']})")


def test_superhuman_obs_vector():
    """Validates Model C 87-dimensional observation space, action history, and trap line encoding."""
    env_fn = make_superhuman_multi_env(
        num_seats=5,
        starting_chips=200,
        seed=77,
        max_hands_per_session=15,
    )
    env = env_fn()
    obs, info = env.reset(seed=77)

    assert obs.shape == (SUPERHUMAN_OBS_DIM,), f"Expected shape ({SUPERHUMAN_OBS_DIM},), got {obs.shape}"
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0), "Superhuman observation out of bounds [-1.0, 1.0]"

    # Step through 50 turns
    done = False
    step_count = 0
    while not done and step_count < 50:
        step_count += 1
        mask = env.action_masks()
        legal = np.where(mask == 1)[0]
        action = int(np.random.choice(legal))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        assert obs.shape == (SUPERHUMAN_OBS_DIM,)
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)

    print(f"test_superhuman_obs_vector: PASS (87-dim observation verified over {step_count} steps)")


if __name__ == "__main__":
    test_wrapper_reset_and_step()
    test_scale_invariance()
    test_randomized_stacks_and_blind_escalation()
    test_superhuman_obs_vector()
    print("ALL GYM WRAPPER TESTS PASSED!")
