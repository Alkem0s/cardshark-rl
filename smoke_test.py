"""smoke_test.py — Quick sanity check for the CardShark-RL codebase."""
import sys

print("=== CardShark-RL Smoke Test ===\n")

# 1. Imports
print("[1/5] Testing imports...", end=" ", flush=True)
from card_utils import evaluate_hand, Deck, hand_category_name
from opponents import CallingStation, Maniac, Rock, make_random_opponent
from draw_poker_env import DrawPokerEnv
from gym_wrapper import DrawPokerGymEnv, mask_fn
import numpy as np
print("OK")

# 2. Hand evaluator
print("[2/5] Testing hand evaluator...", end=" ", flush=True)
royal_flush = [12, 25, 38, 51, 8]   # A♣ Q♦ K♥ A♠ T♣  — actually let's use a known one
# Royal flush: Tc Jc Qc Kc Ac  = cards 8,9,10,11,12
royal = [8, 9, 10, 11, 12]
pair_aces = [0, 13, 1, 2, 3]   # Ac Ad 3c 4c 5c  (pair of 2s actually)
assert evaluate_hand(royal) > evaluate_hand(pair_aces), "Royal flush must beat pair"
assert hand_category_name(royal) == "Straight Flush", f"Got: {hand_category_name(royal)}"
print("OK")

# 3. Opponents
print("[3/5] Testing opponent archetypes...", end=" ", flush=True)
rng = np.random.default_rng(42)
deck = Deck(rng=rng)
hand = deck.deal(5)
for OpponentClass in [CallingStation, Maniac, Rock]:
    opp = OpponentClass(rng=rng)
    bet_act = opp.bet_action(hand, pot=4, bet_to_call=2, phase="pre_draw", can_raise=True)
    draw_act = opp.draw_action(hand)
    assert bet_act in (0, 1, 2), f"{opp.name}: invalid bet action {bet_act}"
    assert all(0 <= i <= 4 for i in draw_act), f"{opp.name}: invalid draw indices {draw_act}"
print("OK")

# 4. Environment reset + random rollout
print("[4/5] Testing env reset and random rollout (200 episodes)...", end=" ", flush=True)
env = DrawPokerGymEnv(use_implicit_modeling=False, rng_seed=0)
errors = 0
for ep in range(200):
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape, "Obs shape mismatch"
    done = False
    steps = 0
    while not done and steps < 50:
        mask = env.action_mask()
        valid = np.where(mask == 1)[0]
        assert len(valid) > 0, f"No valid actions at ep {ep} step {steps}"
        action = int(np.random.choice(valid))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    if not done:
        errors += 1
assert errors == 0, f"{errors}/200 episodes did not terminate"
print("OK")

# 5. Implicit modeling obs size
print("[5/5] Testing implicit vs explicit obs sizes...", end=" ", flush=True)
env_a = DrawPokerGymEnv(use_implicit_modeling=False, rng_seed=1)
env_b = DrawPokerGymEnv(use_implicit_modeling=True,  rng_seed=1)
obs_a, _ = env_a.reset()
obs_b, _ = env_b.reset()
assert obs_a.shape == (15,), f"Model A obs should be (15,), got {obs_a.shape}"
assert obs_b.shape == (18,), f"Model B obs should be (18,), got {obs_b.shape}"
print("OK")

print("\n=== ALL TESTS PASSED ===")
