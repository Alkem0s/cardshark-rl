import sys
import os

PROJECT_ROOT = "/mnt/c/Users/efe20/Desktop/Inventory/Programming/Python/Uni CENG/AI & DS/cardshark-rl"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sb3_contrib import MaskablePPO
from games.holdem.gym_wrapper import HoldemPokerGymEnv

model = MaskablePPO.load("models/holdem/model_holdem_b.zip")
env = HoldemPokerGymEnv(
    num_seats=5,
    starting_chips=200,
    hero_seat=0,
    max_hands_per_session=100,
    rng_seed=500_000,
    obs_version="b",
)

winners = {}
for sess_i in range(50):
    obs, info = env.reset(seed=500_000 + sess_i * 1000)
    done = False
    while not done:
        mask = env.action_masks()
        act, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, r, term, trunc, info = env.step(int(act))
        done = term or trunc
    
    # Check who has chips
    alive = [s for s in env.env.seats if s.chips > 0]
    if len(alive) == 1:
        winner_seat = alive[0].seat_idx
    else:
        winner_seat = max(env.env.seats, key=lambda s: s.chips).seat_idx

    if winner_seat == env.hero_seat:
        winner_name = "HERO (Model B)"
    else:
        opp = env.opponents.get(winner_seat)
        winner_name = f"{opp.name} (Seat {winner_seat})" if opp else f"Seat {winner_seat}"
    winners[winner_name] = winners.get(winner_name, 0) + 1

print("Tournament 50 Sessions Winner Breakdown:")
for name, wins in sorted(winners.items(), key=lambda x: -x[1]):
    print(f"  {name:15s}: {wins} wins ({wins/50*100:.1f}%)")
