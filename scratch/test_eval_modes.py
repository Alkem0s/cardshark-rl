import sys
import os
import numpy as np

PROJECT_ROOT = "/mnt/c/Users/efe20/Desktop/Inventory/Programming/Python/Uni CENG/AI & DS/cardshark-rl"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sb3_contrib import MaskablePPO
from games.holdem.gym_wrapper import HoldemPokerGymEnv

def test_modes(num_sessions=50):
    model = MaskablePPO.load("models/holdem/model_holdem_b.zip")
    
    for det in [True, False]:
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
                mask = env.action_masks()
                act, _ = model.predict(obs, action_masks=mask, deterministic=det)
                act = int(act)
                action_counts[act] += 1
                
                obs, r, term, trunc, info = env.step(act)
                sess_r += r
                done = term or trunc
                
            rewards.append(sess_r)
            if info.get("is_winner", False):
                wins += 1
            if env.env.seats[0].chips <= 0:
                eliminations += 1
                
        mode_str = "DETERMINISTIC" if det else "STOCHASTIC"
        print(f"\n=== Benchmark Mode: {mode_str} ({num_sessions} sessions) ===")
        print(f"Championship Win Rate: {wins/num_sessions*100:.1f}% ({wins}/{num_sessions})")
        print(f"Elimination Rate: {eliminations/num_sessions*100:.1f}% ({eliminations}/{num_sessions})")
        print(f"Mean Return: {np.mean(rewards):.2f} BBs")
        print(f"Action Counts: {action_counts}")

if __name__ == "__main__":
    test_modes(50)
