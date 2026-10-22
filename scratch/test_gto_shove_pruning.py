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
from games.holdem.evaluator import evaluate_preflop_hand_chen
from games.holdem.board_texture import evaluate_board_texture_and_draws

def run_test(num_sessions=50):
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
    elims = 0
    rewards = []
    action_counts = {a: 0 for a in range(6)}
    
    for sess_i in range(num_sessions):
        obs, info = env.reset(seed=500_000 + sess_i * 1000)
        done = False
        sess_r = 0.0
        
        while not done:
            p = env.env.seats[env.hero_seat]
            mask = env.action_masks().copy()
            
            # Prune suicidal shoves:
            if mask[A_ALL_IN] == 1:
                bb_stack = p.chips / env.env.big_blind
                pot = env.env.pot
                to_call = p.bet_to_call
                
                allow_shove = False
                if env.env.street == STREET_PREFLOP and len(p.hand) == 2:
                    chen = evaluate_preflop_hand_chen(p.hand[0], p.hand[1])
                    # Premium hands can always shove
                    if chen >= 0.55: # AA, KK, QQ, AKs
                        allow_shove = True
                    # If very short stack (<= 10 BB), can shove decent hands (Chen >= 0.35)
                    elif bb_stack <= 10.0 and chen >= 0.35:
                        allow_shove = True
                    # If facing an all-in and pot committed with decent hand
                    elif to_call >= p.chips * 0.5 and chen >= 0.38:
                        allow_shove = True
                else: # Flop, Turn, River
                    tex = evaluate_board_texture_and_draws(p.hand, env.env.board)
                    # Made hands: 5: overpair, 6: top_pair, 7: mid/bottom_pair, 8: two_pair, 9: set/trips, 10: straight+
                    has_monster = (tex[8] > 0 or tex[9] > 0 or tex[10] > 0) # Two pair or better
                    has_top_pair_good_kicker = (tex[6] > 0 and tex[16] >= 0.70)
                    has_monster_draw = (tex[11] > 0 and tex[13] > 0) # Flush + Straight draw
                    
                    if has_monster or has_monster_draw:
                        allow_shove = True
                    elif has_top_pair_good_kicker and (bb_stack <= 15.0 or to_call >= p.chips * 0.4):
                        allow_shove = True
                    elif bb_stack <= 6.0 and (tex[6] > 0 or tex[5] > 0 or tex[11] > 0 or tex[13] > 0):
                        allow_shove = True
                        
                if not allow_shove:
                    mask[A_ALL_IN] = 0
                    
            if mask.sum() == 0:
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
            elims += 1
            
    print(f"=== PRUNED SHOVE BENCHMARK ({num_sessions} SESSIONS) ===")
    print(f"Win Rate: {wins/num_sessions*100:.1f}% ({wins}/{num_sessions})")
    print(f"Elimination Rate: {elims/num_sessions*100:.1f}% ({elims}/{num_sessions})")
    print(f"Mean Return: {np.mean(rewards):.2f} BBs")
    print(f"Action Counts: {action_counts}")

if __name__ == "__main__":
    run_test(50)
