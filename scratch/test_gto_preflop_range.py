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

def test_gto_preflop(num_sessions=50):
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
    
    total_hands = 0
    vpip_hands = 0
    
    for sess_i in range(num_sessions):
        obs, info = env.reset(seed=500_000 + sess_i * 1000)
        done = False
        sess_r = 0.0
        
        last_hand = 0
        current_vpip = False
        
        while not done:
            p = env.env.seats[env.hero_seat]
            mask = env.action_masks().copy()
            
            # GTO PREFLOP DISCIPLINE:
            # If preflop and holding 2 cards:
            if env.env.street == STREET_PREFLOP and len(p.hand) == 2:
                chen = evaluate_preflop_hand_chen(p.hand[0], p.hand[1])
                
                # If facing big blind or raise with weak hand (Chen < 0.25):
                # Unless free check in Big Blind (bet_to_call == 0), FOLD!
                if chen < 0.25 and p.bet_to_call > 0:
                    mask = np.zeros(6, dtype=np.int8)
                    mask[A_FOLD] = 1
                    
                # If holding marginal hand (Chen < 0.35) facing a 3-bet / raise:
                # Cannot re-raise
                elif chen < 0.35 and p.bet_to_call >= env.env.big_blind * 2:
                    mask[A_MIN_RAISE] = 0
                    mask[A_HALF_POT] = 0
                    mask[A_POT] = 0
                    mask[A_ALL_IN] = 0
                    
            # POSTFLOP DISCIPLINE:
            elif env.env.street in (STREET_FLOP, STREET_TURN, STREET_RIVER) and len(env.env.board) >= 3 and len(p.hand) == 2:
                tex = evaluate_board_texture_and_draws(p.hand, env.env.board)
                has_made = any(tex[k] > 0 for k in range(5, 11))
                has_strong_draw = (tex[11] > 0 or tex[13] > 0)
                
                # Turn/River air: fold if bet into, check if checked to
                if env.env.street in (STREET_TURN, STREET_RIVER):
                    if not has_made and not (env.env.street == STREET_TURN and has_strong_draw):
                        if p.bet_to_call > 0:
                            mask = np.zeros(6, dtype=np.int8)
                            mask[A_FOLD] = 1
                        else:
                            mask = np.zeros(6, dtype=np.int8)
                            mask[A_CHECK_CALL] = 1
                
                # Mask deep-stack over-bet all-in shoves (stack > 2x pot)
                bb_stack = p.chips / max(1, env.env.big_blind)
                if bb_stack > 12.0 and p.chips > env.env.pot * 2.0:
                    mask[A_ALL_IN] = 0
                    
            if mask.sum() == 0:
                mask[A_CHECK_CALL] = 1
                
            act, _ = model.predict(obs, action_masks=mask, deterministic=False) # evaluate with policy distribution
            act = int(act)
            action_counts[act] += 1
            
            if env.env.street == STREET_PREFLOP:
                if act != A_FOLD and (p.bet_to_call > 0 or act != A_CHECK_CALL):
                    current_vpip = True
                    
            obs, r, term, trunc, info = env.step(act)
            sess_r += r
            done = term or trunc
            
            if env.env.hands_played > last_hand or done:
                total_hands += 1
                if current_vpip:
                    vpip_hands += 1
                current_vpip = False
                last_hand = env.env.hands_played
                
        rewards.append(sess_r)
        if info.get("is_winner", False):
            wins += 1
        if p.chips <= 0:
            eliminations += 1
            
    vpip_pct = (vpip_hands / max(1, total_hands)) * 100
    print(f"\n=== GTO Preflop & Postflop Benchmark ({num_sessions} sessions) ===")
    print(f"Championship Win Rate: {wins/num_sessions*100:.1f}% ({wins}/{num_sessions})")
    print(f"Elimination Rate: {eliminations/num_sessions*100:.1f}% ({eliminations}/{num_sessions})")
    print(f"Mean Return: {np.mean(rewards):.2f} BBs | VPIP: {vpip_pct:.1f}%")
    print("Action Counts:", action_counts)

if __name__ == "__main__":
    test_gto_preflop(50)
