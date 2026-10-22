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
from games.holdem.evaluator import evaluate_preflop_hand_chen, evaluate_holdem_hand, holdem_category_name
from games.holdem.board_texture import evaluate_board_texture_and_draws
from core.card import hand_str

model = MaskablePPO.load("models/holdem/model_holdem_b.zip")
env = HoldemPokerGymEnv(
    num_seats=5,
    starting_chips=200,
    hero_seat=0,
    max_hands_per_session=100,
    rng_seed=500_000,
    obs_version="b",
)

losses = 0
for sess_i in range(15):
    obs, info = env.reset(seed=500_000 + sess_i * 1000)
    done = False
    hand_logs = []
    
    while not done:
        p = env.env.seats[env.hero_seat]
        mask = env.action_masks().copy()
        
        if env.env.street == STREET_PREFLOP and len(p.hand) == 2:
            chen = evaluate_preflop_hand_chen(p.hand[0], p.hand[1])
            if chen < 0.25 and p.bet_to_call > 0:
                mask = np.zeros(6, dtype=np.int8)
                mask[A_FOLD] = 1
            elif chen < 0.20 and p.bet_to_call == 0:
                mask[A_MIN_RAISE] = 0
                mask[A_HALF_POT] = 0
                mask[A_POT] = 0
                mask[A_ALL_IN] = 0
                mask[A_CHECK_CALL] = 1
            elif chen < 0.40 and p.bet_to_call >= env.env.big_blind * 2:
                mask[A_MIN_RAISE] = 0
                mask[A_HALF_POT] = 0
                mask[A_POT] = 0
                mask[A_ALL_IN] = 0
        
        elif env.env.street in (STREET_TURN, STREET_RIVER) and len(env.env.board) >= 4 and len(p.hand) == 2:
            tex = evaluate_board_texture_and_draws(p.hand, env.env.board)
            has_made = any(tex[k] > 0 for k in range(5, 11))
            has_strong_draw = (env.env.street == STREET_TURN and (tex[11] > 0 or tex[13] > 0))
            if not has_made and not has_strong_draw:
                if p.bet_to_call > 0:
                    mask = np.zeros(6, dtype=np.int8)
                    mask[A_FOLD] = 1
                else:
                    mask = np.zeros(6, dtype=np.int8)
                    mask[A_CHECK_CALL] = 1
                    
        if mask.sum() == 0:
            mask[A_CHECK_CALL] = 1
            
        act, _ = model.predict(obs, action_masks=mask, deterministic=True)
        act = int(act)
        
        hero_hand = hand_str(p.hand, symbols=True) if p.hand else ""
        board = hand_str(env.env.board, symbols=True) if env.env.board else ""
        
        obs, r, term, trunc, info = env.step(act)
        done = term or trunc
        
        if p.chips <= 0:
            print(f"[BUST-OUT in Session #{sess_i+1}] on Street: {env.env.street}")
            print(f"  Hero hand: {hero_hand} | Board: {board} | Action taken: {act}")
            break
