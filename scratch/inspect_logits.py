import sys
import os
import torch
import numpy as np

PROJECT_ROOT = "/mnt/c/Users/efe20/Desktop/Inventory/Programming/Python/Uni CENG/AI & DS/cardshark-rl"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sb3_contrib import MaskablePPO
from games.holdem.gym_wrapper import HoldemPokerGymEnv
from games.holdem.action_abstraction import ACTION_NAMES

model = MaskablePPO.load("models/holdem/model_holdem_b.zip")
env = HoldemPokerGymEnv(num_seats=5, starting_chips=200, hero_seat=0, obs_version="b", rng_seed=42)

from core.card import parse_card

# Test 1: Pocket Aces preflop
policy = model.policy
env.env.seats[0].hand = [parse_card("As"), parse_card("Ah")]
env.env.seats[0].bet_to_call = 2
env.env.street = "preflop"
obs = env._get_obs() if hasattr(env, "_get_obs") else env.reset()[0]
env.env.seats[0].hand = [parse_card("As"), parse_card("Ah")]
env.env.seats[0].bet_to_call = 2
env.env.street = "preflop"

with torch.no_grad():
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)
    raw_logits = policy.action_net(policy.mlp_extractor.forward_actor(policy.features_extractor(obs_tensor)))[0].cpu().numpy()
    mask = env.action_masks()
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0).to(model.device)
    dist = policy.get_distribution(obs_tensor, action_masks=mask_tensor)
    probs = dist.distribution.probs.cpu().numpy()[0]

print("Pocket Aces Preflop (facing 2 BB call):")
for a in range(6):
    print(f"  Action {a} ({ACTION_NAMES[a]:12s}): Raw Logit = {raw_logits[a]:+6.3f} | Masked Prob = {probs[a]*100:5.2f}% | Legal = {bool(mask[a])}")

