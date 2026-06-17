# CardShark RL: Limit Heads-Up 5-Card Draw Poker Reinforcement Learning

This project implements a reinforcement learning framework for training and evaluating intelligent agents in Limit Heads-Up 5-Card Draw Poker. The training pipeline utilizes Proximal Policy Optimization with action masking via Stable-Baselines3.

---

## Core Concepts

### Game Environment
The custom game engine implements the complete logic for Limit Heads-Up 5-Card Draw Poker:
1. **Antes**: Both players contribute a fixed ante to the pot.
2. **Pre-Draw Betting Round**: Betting limits are set to the small bet size.
3. **Draw Phase**: Players can discard up to 5 cards and receive new ones.
4. **Post-Draw Betting Round**: Betting limits are increased to the big bet size.
5. **Showdown**: Hands are evaluated, and the pot is awarded to the best hand.

### Opponent Archetypes
To train a robust and adaptive agent, players face three hard-coded opponent strategies:
* **Calling Station**: A passive player who never raises, checks or calls everything, and draws mathematically.
* **Maniac**: An aggressive player who raises and bets frequently, bluffs, and chases flush or straight draws.
* **Rock**: A tight-predictable player who only plays high cards/pairs, folds weak hands pre-draw, and folds under pressure unless their hand improves.

### Modeling Paradigms
Two modeling techniques are implemented and compared:
* **Model A (Explicit Modeling)**: The agent receives the opponent's exact archetype identity as a one-hot vector in the observation space.
* **Model B (Implicit Modeling)**: The agent has no prior knowledge of the opponent's identity. Instead, it tracks a rolling window of opponent actions (e.g., pre/post-draw fold rates, VPIP, PFR, Aggression Factor) to implicitly model and adapt to their playing style.
