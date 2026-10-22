import sys
import os
import numpy as np

PROJECT_ROOT = "/mnt/c/Users/efe20/Desktop/Inventory/Programming/Python/Uni CENG/AI & DS/cardshark-rl"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sb3_contrib import MaskablePPO
from games.holdem.gym_wrapper import HoldemPokerGymEnv
from games.holdem.action_abstraction import (
    A_FOLD, A_CHECK_CALL, A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN,
    STREET_PREFLOP, STREET_FLOP, STREET_TURN, STREET_RIVER
)

def test_inference(num_sessions=20):
    model = MaskablePPO.load("models/holdem/model_holdem_b.zip")
    env = HoldemPokerGymEnv(
        num_seats=5,
        starting_chips=200,
        hero_seat=0,
        max_hands_per_session=100,
        rng_seed=500_000,
        obs_version="b",
    )
    
    action_names = {0: "Fold", 1: "Check/Call", 2: "Min-Raise", 3: "Half-Pot", 4: "Pot", 5: "All-In"}
    action_counts = {a: 0 for a in range(6)}
    flop_counts = {a: 0 for a in range(6)}
    
    wins = 0
    rewards = []
    
    for sess_i in range(num_sessions):
        obs, info = env.reset(seed=500_000 + sess_i * 1000)
        done = False
        sess_r = 0.0
        
        while not done:
            mask = env.action_masks()
            # If on flop and pot is small relative to stack, test what happens if All-In is masked
            if env.env.street == STREET_FLOP and env.env.pot < 20 and env.env.seats[env.hero_seat].chips > 40:
                mask[A_ALL_IN] = 0  # Force model to choose a non-shove action
                
            act, _ = model.predict(obs, action_masks=mask, deterministic=True)
            act = int(act)
            action_counts[act] += 1
            if env.env.street == STREET_FLOP:
                flop_counts[act] += 1
                
            obs, r, term, trunc, info = env.step(act)
            sess_r += r
            done = term or trunc
            
        rewards.append(sess_r)
        if info.get("is_winner", False):
            wins += 1
            
    print(f"Masked Flop Shove Test ({num_sessions} sessions):")
    print(f"Win Rate: {wins/num_sessions*100:.1f}% | Mean Return: {np.mean(rewards):.2f} BBs")
    print("Overall Actions:", {action_names[k]: v for k, v in action_counts.items()})
    print("Flop Actions:", {action_names[k]: v for k, v in flop_counts.items()})

if __name__ == "__main__":
    test_inference(25)
