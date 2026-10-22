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
from games.holdem.board_texture import evaluate_board_texture_and_draws

def test_air_folding(num_sessions=50):
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
    flop_folds = 0
    turn_folds = 0
    river_folds = 0
    
    for sess_i in range(num_sessions):
        obs, info = env.reset(seed=500_000 + sess_i * 1000)
        done = False
        sess_r = 0.0
        
        while not done:
            mask = env.action_masks()
            p = env.env.seats[env.hero_seat]
            
            # Postflop Air Pruning:
            if env.env.street in (STREET_FLOP, STREET_TURN, STREET_RIVER) and len(env.env.board) >= 3:
                # If facing a bet postflop:
                if p.bet_to_call > 0:
                    tex = evaluate_board_texture_and_draws(p.hand, env.env.board)
                    # Indices in board_texture:
                    # 5: overpair, 6: top_pair, 7: mid/bottom_pair, 8: two_pair, 9: set_trips, 10: straight+
                    # 11: flush_draw, 12: nut_flush_draw, 13: oesd, 14: gutshot, 15: overcards
                    has_made = any(tex[k] > 0 for k in range(5, 11))
                    has_draw = any(tex[k] > 0 for k in range(11, 15))
                    has_overcards = tex[15] > 0
                    
                    # On Turn and River: If no made hand and no strong draw (flush draw or OESD on turn):
                    if env.env.street in (STREET_TURN, STREET_RIVER):
                        has_strong_turn_draw = (env.env.street == STREET_TURN and (tex[11] > 0 or tex[13] > 0))
                        if not has_made and not has_strong_turn_draw:
                            # Pure air facing a bet on turn/river: MUST FOLD!
                            mask = np.zeros(6, dtype=np.int8)
                            mask[A_FOLD] = 1
                            if env.env.street == STREET_TURN: turn_folds += 1
                            else: river_folds += 1
                    elif env.env.street == STREET_FLOP:
                        if not has_made and not has_draw and not has_overcards:
                            # Complete air on flop facing bet: Fold
                            mask = np.zeros(6, dtype=np.int8)
                            mask[A_FOLD] = 1
                            flop_folds += 1
            
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
            
    print(f"Air Pruning Benchmark ({num_sessions} sessions):")
    print(f"Championship Win Rate: {wins/num_sessions*100:.1f}% ({wins}/{num_sessions})")
    print(f"Elimination Rate: {eliminations/num_sessions*100:.1f}% ({eliminations}/{num_sessions})")
    print(f"Mean Return: {np.mean(rewards):.2f} BBs")
    print("Action Counts:", action_counts)
    print(f"Folds triggered by Air Pruning -> Flop: {flop_folds}, Turn: {turn_folds}, River: {river_folds}")

if __name__ == "__main__":
    test_air_folding(50)
