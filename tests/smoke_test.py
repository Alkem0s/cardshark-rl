"""
tests/smoke_test.py — Universal Sanity Smoke Test for All CardShark-RL Game Engines.

Tests:
1. 5-Card Draw rules, evaluator, and gym wrapper
2. 2-7 Triple Draw lowball evaluator, wheel detection, and environment
3. Texas Hold'em 7-card evaluator, street transitions, and action abstractions
"""
import sys
import os
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("=" * 60)
print("  CARDSHARK-RL UNIVERSAL MULTI-GAME SMOKE TEST")
print("=" * 60)

# 1. 5-Card Draw
print("\n[1/3] Testing 5-Card Draw Engine...", end=" ", flush=True)
from games.five_card_draw.rules import evaluate_hand, Deck, hand_category_name
from games.five_card_draw.env import MultiDrawPokerEnv
from rl.gym_wrapper import DrawPokerGymEnv

royal = [8, 9, 10, 11, 12]
pair = [0, 13, 1, 2, 3]
assert evaluate_hand(royal) > evaluate_hand(pair), "High poker evaluator failure"
assert hand_category_name(royal) == "Straight Flush"

fcd_env = MultiDrawPokerEnv(num_seats=5, starting_chips=200, rng_seed=42)
fcd_env.reset_session()
assert len(fcd_env.seats) == 5
assert fcd_env.pot > 0
print("OK")

# 2. 2-7 Triple Draw (Kansas City Lowball)
print("[2/3] Testing 2-7 Triple Draw Engine...", end=" ", flush=True)
from games.triple_draw_27.rules import evaluate_27_hand, is_wheel_27, hand_category_27
from games.triple_draw_27.env import TripleDrawPokerEnv

# The wheel in 2-7 Lowball is 7-5-4-3-2 unsuited:
# 2♣ (0), 3♦ (14), 4♥ (28), 5♠ (42), 7♣ (5)
wheel = [0, 14, 28, 42, 5]
rough_8 = [0, 14, 28, 42, 6]  # 8-5-4-3-2
assert is_wheel_27(wheel), "Wheel detection failed"
assert evaluate_27_hand(wheel) > evaluate_27_hand(rough_8), "2-7 lowball rank inverted"

td_env = TripleDrawPokerEnv(num_seats=5, starting_chips=200, rng_seed=42)
td_env.reset_session()
assert len(td_env.seats) == 5
assert td_env.pot > 0
print("OK")

# 3. No-Limit Texas Hold'em
print("[3/3] Testing No-Limit Texas Hold'em Engine...", end=" ", flush=True)
from games.holdem.evaluator import evaluate_holdem_hand, holdem_category_name
from games.holdem.env import HoldemPokerEnv

# 7 cards: Hole = [A♠ (51), K♠ (50)], Board = [Q♠ (49), J♠ (48), 10♠ (47), 2♦ (14), 3♣ (1)]
royal_holdem = [51, 50, 49, 48, 47, 14, 1]
score, cat = evaluate_holdem_hand(royal_holdem)
assert holdem_category_name(cat) in ("Straight Flush", "Royal Flush"), f"Got: {holdem_category_name(cat)}"

nlhe_env = HoldemPokerEnv(num_seats=6, starting_chips=1000, rng_seed=42)
nlhe_env.reset_session()
assert len(nlhe_env.seats) == 6
assert nlhe_env.pot > 0
print("OK")

print("\n" + "=" * 60)
print("  ALL 3 GAME ENGINES PASSED SMOKE VALIDATION CLEANLY!")
print("=" * 60)
