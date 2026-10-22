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

def test_pot_sizing(num_sessions=50):
    model = MaskablePPO.load("models/holdem/model_holdem_b.zip")
    env = HoldemPokerGymEnv(
        num_seats=5,
        starting_chips=200,
        hero_seat=0,
        max_hands_per_session=100,
        rng_seed=500_000,
        obs_version="b",
    )
    
    wins = 0
    eliminations = 0
    rewards = []
    action_counts = {a: 0 for a in range(6)}
    
    for sess_i in range(num_sessions):
        obs, info = env.reset(seed=500_000 + sess_i * 1000)
        done = False
        sess_r = 0.0
        
        while not done:
            p = env.env.seats[env.hero_seat]
            mask = env.action_masks().copy()
            
            # Mask over-bet open shoves when stack > 2.5x pot (unless short stack <= 12 BBs)
            bb_stack = p.chips / max(1, env.env.big_blind)
            if bb_stack > 12.0 and p.chips > env.env.pot * 2.0:
                mask[A_ALL_IN] = 0
                
            # If all-in was the only aggressive action masked, ensure legal actions exist
            if mask.sum() == 0 or (mask[A_CHECK_CALL] == 0 and mask[A_FOLD] == 0 and mask[A_MIN_RAISE] == 0 and mask[A_HALF_POT] == 0 and mask[A_POT] == 0):
                mask[A_CHECK_CALL] = 1
                
            act, _ = model.predict(obs, action_masks=mask, deterministic=True)
            act = int(act)
            action_counts[act] += 1
            
            obs, r, term, trunc, info = env.step(act)
            sess_r += r
            done = term or trunc
            
        rewards.append(sess_r)
        if info.get("is_winner", False):
            wins += 1
        if p.chips <= 0:
            eliminations += 1
            
    print(f"\n=== Over-Bet Shove Pruning Benchmark ({num_sessions} sessions) ===")
    print(f"Championship Win Rate: {wins/num_sessions*100:.1f}% ({wins}/{num_sessions})")
    print(f"Elimination Rate: {eliminations/num_sessions*100:.1f}% ({eliminations}/{num_sessions})")
    print(f"Mean Return: {np.mean(rewards):.2f} BBs")
    print("Action Counts:", action_counts)

if __name__ == "__main__":
    test_pot_sizing(50)
