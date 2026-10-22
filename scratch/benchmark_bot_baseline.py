import sys
import os
import numpy as np
from collections import Counter

PROJECT_ROOT = "/mnt/c/Users/efe20/Desktop/Inventory/Programming/Python/Uni CENG/AI & DS/cardshark-rl"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from games.holdem.env import HoldemPokerEnv
from games.holdem.opponents import make_random_holdem_opponent, TAGBot, LAGBot, CallingStationBot, RandomHoldemBot

def benchmark_bot_tournament(num_sessions=100):
    bot_wins = Counter()
    bot_types_at_seat0 = Counter()
    
    for sess_i in range(num_sessions):
        rng = np.random.default_rng(700_000 + sess_i * 1000)
        env = HoldemPokerEnv(num_seats=5, starting_chips=200, small_blind=1, big_blind=2, rng=rng)
        env.reset_session()
        
        # Create 5 bots
        bots = [make_random_holdem_opponent(opp_id=i, rng=rng) for i in range(5)]
        
        while not env.session_done:
            cur = env.current_seat
            p = env.seats[cur]
            legal = env.get_legal_actions(cur)
            if not legal:
                env._advance_betting()
                continue
            act = bots[cur].act(
                hole_cards=p.hand,
                board=env.board,
                pot=env.pot,
                bet_to_call=p.bet_to_call,
                street=env.street,
                stack=p.chips,
                legal_actions=legal
            )
            env.step_action(cur, act)
            
        # Determine winner
        chips = [s.chips for s in env.seats]
        winner = int(np.argmax(chips))
        winner_bot = bots[winner]
        bot_wins[type(winner_bot).__name__] += 1
        
    print(f"\n=== 5-Bot Tournament Baseline ({num_sessions} sessions) ===")
    total = sum(bot_wins.values())
    for b_name, count in bot_wins.most_common():
        print(f"  {b_name:20s}: {count:3d} wins ({count/total*100:5.1f}%)")

if __name__ == "__main__":
    benchmark_bot_tournament(100)
