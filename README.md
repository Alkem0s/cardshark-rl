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

---

## Key Results & Findings

Based on evaluations over 10,000 hands against each opponent archetype (Calling Station, Maniac, Rock):

### Performance Comparison (BB/100)

| Opponent | Model A (Explicit) | Model B (Implicit) | Random |
| :--- | :---: | :---: | :---: |
| **Calling Station** | $+86.5 \pm 4.3$ | $+79.9 \pm 5.0$ | $-143.8 \pm 2.7$ |
| **Maniac** | $+408.0 \pm 15.7$ | $+390.8 \pm 11.5$ | $-222.1 \pm 0.1$ |
| **Rock** | $+47.5 \pm 2.8$ | $+28.9 \pm 6.5$ | $-151.3 \pm 2.0$ |
| **Average** | **$+180.65$** | **$+166.50$** | **$-172.44$** |

*Model A's direct archetype access gives it a modest but statistically significant edge of roughly ~14 BB/100 overall.*

### Non-Stationary Adaptation (1,500-Hand Tournament)
In a tournament where the opponent silently transitions archetype mid-session (Calling Station $\rightarrow$ Maniac $\rightarrow$ Rock), the implicit agent (Model B) dynamically adapts via its rolling buffer:
* **Model B (Implicit)**: **$+160.35$ BB/100**
* **Model A (Explicit - Oracle)**: $+156.28$ BB/100
* **Model A (Explicit - Blind)**: $+50.01$ BB/100

*Model B slightly outperforms the oracle-labeled explicit model by smoothing transition states instead of instantly swapping policy weights.*

### Behavioral Matrix Analysis
Model B learns specific, draw-count-driven behavioral heuristics without ever seeing archetype labels:
* **Against Rock**: Model B collapses call/raise probabilities to 0% (folds 100% of the time) when the Rock draws exactly **one** card (historically highly correlated with premium draws). Conversely, standing pat (draw 0) is treated as a cautious call/raise spot.
* **Against Maniac**: A single-card draw triggers Model B to raise 52% of the time, exploiting the Maniac's tendency to chase weak straight/flush draws.