"""
tests/test_five_card_draw/test_smoke.py — Quick sanity smoke test for 5-Card Draw pipeline.
"""
import sys
import os
import shutil
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("=== 5-Card Draw Smoke Test ===\n")

# 1. Imports
print("[1/5] Testing imports...", end=" ", flush=True)
try:
    from games.five_card_draw.rules import evaluate_hand, Deck, hand_category_name
    from games.five_card_draw.opponents import CallingStation, Maniac, Rock, make_random_opponent
    from games.five_card_draw.heads_up_env import DrawPokerEnv
except ImportError:
    from game.card_utils import evaluate_hand, Deck, hand_category_name
    from game.opponents import CallingStation, Maniac, Rock, make_random_opponent
    from game.draw_poker_env import DrawPokerEnv
from rl.gym_wrapper import DrawPokerGymEnv
print("OK")

# 2. Hand evaluator
print("[2/5] Testing hand evaluator...", end=" ", flush=True)
royal = [8, 9, 10, 11, 12]
pair_aces = [0, 13, 1, 2, 3]
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
print("[4/5] Testing env reset and rollout...", end=" ", flush=True)
env = DrawPokerGymEnv(use_implicit_modeling=False, rng_seed=0)
obs, info = env.reset()
assert obs.shape == env.observation_space.shape, "Obs shape mismatch"
print("OK")

# 5. Observation sizes
print("[5/5] Testing obs shapes...", end=" ", flush=True)
env_a = DrawPokerGymEnv(use_implicit_modeling=False, rng_seed=1)
env_b = DrawPokerGymEnv(use_implicit_modeling=True, rng_seed=1)
obs_a, _ = env_a.reset()
obs_b, _ = env_b.reset()
assert obs_a.shape == (26,), f"Expected (26,), got {obs_a.shape}"
assert obs_b.shape == (33,), f"Expected (33,), got {obs_b.shape}"
print("OK")

print("\n=== 5-CARD DRAW SMOKE TEST PASSED ===")
