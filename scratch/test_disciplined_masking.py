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

def test_disciplined(num_sessions=50):
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
            
            # 1. Preflop Discipline
            if env.env.street == STREET_PREFLOP and len(p.hand) == 2:
                chen = evaluate_preflop_hand_chen(p.hand[0], p.hand[1])
                # Trash hands facing raise: Fold
                if chen < 0.25 and p.bet_to_call > 0:
                    mask = np.zeros(6, dtype=np.int8)
                    mask[A_FOLD] = 1
                # Trash hands unopened: Cannot raise
                elif chen < 0.20 and p.bet_to_call == 0:
                    mask[A_MIN_RAISE] = 0
                    mask[A_HALF_POT] = 0
                    mask[A_POT] = 0
                    mask[A_ALL_IN] = 0
                    mask[A_CHECK_CALL] = 1
                # Marginal hands facing 3-bet: Cannot re-raise
                elif chen < 0.40 and p.bet_to_call >= env.env.big_blind * 2:
                    mask[A_MIN_RAISE] = 0
                    mask[A_HALF_POT] = 0
                    mask[A_POT] = 0
                    mask[A_ALL_IN] = 0
            
            # 2. Turn and River Air & Weak Shove Pruning
            elif env.env.street in (STREET_FLOP, STREET_TURN, STREET_RIVER) and len(env.env.board) >= 3 and len(p.hand) == 2:
                tex = evaluate_board_texture_and_draws(p.hand, env.env.board)
                # 5: overpair, 6: top_pair, 7: mid/bottom_pair, 8: two_pair, 9: set/trips, 10: straight+
                # 11: flush_draw, 12: nut_flush_draw, 13: oesd, 14: gutshot, 15: overcards
                has_made = any(tex[k] > 0 for k in range(5, 11))
                has_strong_made = (tex[5] > 0 or tex[6] > 0 or tex[8] > 0 or tex[9] > 0 or tex[10] > 0)
                has_strong_draw = (tex[11] > 0 or tex[13] > 0)
                is_micro_stack = (p.chips <= env.env.big_blind * 10)
                
                # Air folding on turn/river:
                if env.env.street in (STREET_TURN, STREET_RIVER):
                    if not has_made and not (env.env.street == STREET_TURN and has_strong_draw):
                        if p.bet_to_call > 0:
                            mask = np.zeros(6, dtype=np.int8)
                            mask[A_FOLD] = 1
                        else:
                            mask = np.zeros(6, dtype=np.int8)
                            mask[A_CHECK_CALL] = 1
                
                # Weak shove pruning: cannot shove all-in with weak pair / air unless micro-stack or strong hand
                if not is_micro_stack and not has_strong_made and not (has_strong_draw and tex[7] > 0):
                    mask[A_ALL_IN] = 0
                        
            # Ensure at least one legal action
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
            eliminations += 1
            
    print(f"\n=== Disciplined Masking Benchmark ({num_sessions} sessions) ===")
    print(f"Championship Win Rate: {wins/num_sessions*100:.1f}% ({wins}/{num_sessions})")
    print(f"Elimination Rate: {eliminations/num_sessions*100:.1f}% ({eliminations}/{num_sessions})")
    print(f"Mean Return: {np.mean(rewards):.2f} BBs")
    print("Action Counts:", action_counts)

if __name__ == "__main__":
    test_disciplined(50)
