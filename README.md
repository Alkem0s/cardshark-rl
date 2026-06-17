# CardShark-RL: Reinforcement Learning of Behavioral Tells in 5-Card Draw Poker

**Authors:** Alkım Gönenç Efe (220401050), Sarp Sünbül (230401114), Damla Parlakyıldız (220401067)  
**Affiliation:** Department of Computer Engineering, İzmir Katip Çelebi University, İzmir, Turkey

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-0.28.1-green.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/stable--baselines3-2.1.0-orange.svg)](https://stable-baselines3.readthedocs.io/)

CardShark-RL is a reinforcement learning system designed to study **opponent-adaptation and exploitation strategies** in limit heads-up 5-Card Draw Poker. 

By training agents using **MaskablePPO** within a custom PettingZoo/Gymnasium environment, this project compares two core modeling paradigms:
* **Model A (Explicit Modeling):** A baseline agent that is directly given the opponent's exact archetype identity as a one-hot vector in its observation space.
* **Model B (Implicit Modeling):** A more realistic agent that receives no identity labels. Instead, it tracks a rolling window of 10 behavioral statistics (e.g., VPIP, Aggression Factor, draw rates) to implicitly model and adapt to its opponent on the fly.

---

## Paper Abstract

> We present CardShark-RL, a reinforcement learning system for heads-up limit 5-Card Draw Poker that compares two opponent-adaptation strategies: explicit modeling, where a one-hot archetype identifier is fed directly into the agent's observation, and implicit modeling, where the agent works from a rolling window of 10 behavioral statistics and never sees an opponent label. Both agents are trained with MaskablePPO in a custom PettingZoo environment, with Optuna-based hyperparameter search and hand-crafted reward shaping. 
> 
> Evaluated over 10,000 hands each against three scripted archetypes—Calling Station, Maniac, and Rock—the explicit agent (Model A) averages $+180.65$ BB/100, while the implicit agent (Model B) averages $+166.50$ BB/100 ($p = 0.029$). The gap is real but modest. More interesting: in a 1,500-hand non-stationary tournament where the opponent switches archetype mid-session, Model B ($+160.35$ BB/100) edges out an oracle-labeled explicit agent ($+156.28$ BB/100). We analyze the behavioral mechanism through which Model B exploits opponent draw-count patterns without ever knowing who it is playing.

---

## System Architecture & Framework

The system is structured as a single-agent Gymnasium wrapper (`DrawPokerGymEnv`) around a custom PettingZoo AEC environment. The wrapper handles the active agent (`player_0`) while automatically stepping the selected opponent archetype (`player_1`).

![System Architecture](latest%20results/architecture.png)

### Action Space & Hard Action Masking
The action space is defined as `Discrete(35)`:
* **0:** Fold
* **1:** Call/Check
* **2:** Raise/Bet
* **3–34:** Draw combinations (encoded as a 5-bit bitmask corresponding to which of the 5 cards to discard).

To prevent the agent from selecting illegal actions (e.g., trying to draw during a betting phase, or raising when the betting limit is reached), we use **hard action masking**. The log-probabilities of invalid actions are set to $-\infty$ prior to the softmax layer of the policy network, which significantly improves sample efficiency and convergence.

### Observation Space
The base observation space consists of **23 features**:
* **Card Encoding (10):** Suit and normalized rank of the 5 cards in hand.
* **Hand Strength (2):** Broad category (e.g., pair, flush) and exact tiebreaker score.
* **Pot State (3):** Total pot size, cost-to-call, and pot odds.
* **Phase Indicators (3):** One-hot representation of the current phase (Pre-draw, Draw, Post-draw).
* **Opponent Draw Count (1):** Cards discarded/drawn by the opponent (set to $-1$ during pre-draw).
* **Strategic Indicators (4):** Relative table position (dealer/non-dealer), own raise count, opponent aggression flag, and opponent pre-draw raise flag.

#### Model-Specific Observations:
* **Model A (Explicit):** Appends a 3-dimensional one-hot opponent archetype vector (Calling Station, Maniac, or Rock). **Total: 26 dimensions.**
* **Model B (Implicit):** Appends a 10-dimensional rolling statistics buffer tracked over a window of $W$ hands. **Total: 33 dimensions.**

---

## Opponent Archetypes & Behavioral Telling

To train and evaluate the agents, we implemented three distinct scripted opponent archetypes:

1. **Calling Station (Passive):** Never raises and never folds. Evaluates hands mathematically to discard sub-optimal cards. 
2. **Maniac (Aggressive-Loose):** Bets and raises 70% of the time, folds only 5% of the time, and chases straight/flush draws 60% of the time by discarding 1 card.
3. **Rock (Tight-Passive):** Plays a highly conservative range. Folds unpaired hands pre-draw to any bet, calls only with weak/medium pairs (below Jacks), raises with Jacks-or-better, and stands pat (draws 0) on two-pair or stronger.

### The 10-Dimensional Rolling Buffer (Model B)
Model B calculates the following metrics over a sliding window of the last $W = 50$ hands:
* **VPIP (Voluntary Put Money in Pot):** Percentage of hands where the player voluntarily puts money in the pot pre-draw.
* **PFR (Pre-draw Raise Frequency):** Frequency of raising pre-draw.
* **Aggression Factor (AF):** Ratio of aggressive actions (bets + raises) to passive actions (calls) post-draw.
* **Average Cards Drawn:** Average number of cards drawn per hand.
* **Pre-draw Fold Rate:** Frequency of folding pre-draw.
* **Post-draw Fold Rate:** Frequency of folding post-draw.
* **Post-draw Raise Rate:** Frequency of raising post-draw.
* **Raise-then-Stand-Pat Correlation:** How often a pre-draw raise is followed by drawing 0 cards.
* **Fold-after-drawing-3:** Rate of folding after drawing exactly 3 cards.
* **Raise-after-standing-pat:** Rate of raising after drawing 0 cards.

---

## Methodology & Training

### Training Details
Both models utilize **MaskablePPO** with separate actor and critic MLP networks. Training is conducted over **1,000,000 timesteps** utilizing 4 parallel environments.

* **Model A Architecture:** 3 hidden layers of 256 neurons.
* **Model B Architecture:** 4 hidden layers of 256 neurons.

### Reward Shaping
The base reward is the net chip profit/loss (scaled by the big blind size). To accelerate training, we incorporate potential-based reward shaping:
1. **Fold Penalty ($-\alpha$):** Penalizes folding when checking is free (cost-to-call is 0).
2. **Steal Bonus ($+\beta$):** Rewards raising and successfully forcing the opponent to fold.

### Opponent Scheduling
* **Model A:** Uses uniform random scheduling (a new opponent archetype is selected randomly per hand).
* **Model B:** Employs a **hybrid opponent scheduling** routine. To solve the cold-start problem (where rolling stats are noisy at the start of training), it uses block scheduling (cycling archetypes every $B = 400$ hands) for the first $H = 972\text{k}$ timesteps. After $H$, it switches to uniform random scheduling.

### Hyperparameter Optimization (HPO)
We performed a 15-dimensional hyperparameter search using **Optuna** with Tree-structured Parzen Estimators (TPE) and median pruning. The objective function was weighted to emphasize performance against the defensive **Rock** archetype:
$$\text{Score} = 0.5 \cdot \text{BB/100}_{\text{Rock}} + 0.25 \cdot \text{BB/100}_{\text{CallingStation}} + 0.25 \cdot \text{BB/100}_{\text{Maniac}}$$

| Hyperparameter | Model A (Explicit) | Model B (Implicit) |
| :--- | :---: | :---: |
| **Learning Rate** | $1.63 \times 10^{-4}$ | $3.50 \times 10^{-4}$ |
| **n_steps / batch_size** | 4096 / 32 | 2048 / 64 |
| **n_epochs** | 22 | 11 |
| **Discount Factor ($\gamma$)** | 0.9658 | 0.9894 |
| **Entropy Coef ($ent\_coef$)** | 0.00103 | 0.00457 |
| **Clip Range** | 0.242 | 0.194 |
| **GAE Lambda ($\lambda$)** | 0.874 | 0.922 |
| **Value Function Coef ($vf\_coef$)** | 0.475 | 0.430 |
| **Max Gradient Norm** | 0.484 | 0.464 |
| **Fold Penalty ($\alpha$) / Steal Bonus ($\beta$)** | 0.379 / 0.492 | 0.328 / 0.212 |
| **Network Architecture** | $[128] \times 2$ | $[256] \times 4$ |
| **Rolling Window ($W$)** | N/A | 50 hands |
| **Block Size ($B$)** | N/A | 400 hands |
| **Hybrid Switch ($H$)** | N/A | 972,000 steps |

---

## Experimental Results

### 1. Cross-Evaluation Performance (BB/100)
Both models were evaluated over 10,000 hands against each opponent archetype across 3 seeds. Profit is normalized to **Big Blinds per 100 hands (BB/100)**:

| Opponent | Model A (Explicit) | Model B (Implicit) | Random Baseline |
| :--- | :---: | :---: | :---: |
| **Calling Station** | $+86.5 \pm 4.3$ | $+79.9 \pm 5.0$ | $-143.8 \pm 2.7$ |
| **Maniac** | $+408.0 \pm 15.7$ | $+390.8 \pm 11.5$ | $-222.1 \pm 0.1$ |
| **Rock** | $+47.5 \pm 2.8$ | $+28.9 \pm 6.5$ | $-151.3 \pm 2.0$ |
| **Average** | **$+180.65$** | **$+166.50$** | **$-172.44$** |

> [!NOTE]
> **Label Advantage:** Model A's direct access to opponent labels yields a statistically significant edge of roughly ~14 BB/100 overall ($p = 0.029$ in a paired t-test). The gap is largest against the Rock (+18.6 BB/100) because the Rock's sparse aggression signal is hardest to infer implicitly.

<p align="center">
  <img src="latest%20results/bb_per_100_comparison.png" width="48%" />
  <img src="latest%20results/cumulative_profit.png" width="48%" />
</p>

---

### 2. "The Tell": Behavioral Matrix Analysis
To understand *what* Model B actually learned to extract from opponent actions without explicit labels, we mapped its post-draw action distribution (Fold / Call / Raise) against the opponent's draw count (0, 1, or 3 cards).

![Behavioral Matrix](latest%20results/behavioral_matrix.png)

* **Against Calling Station:** The distribution is completely flat (0% fold rate at all draw counts). Since the Calling Station never raises, the agent learned that draw counts carry no threat, and it can always value-bet.
* **Against Maniac:** When the Maniac draws 1 card, Model B raises 52% of the time. The agent exploits the Maniac's tendency to chase weak straight/flush draws, applying high pressure.
* **Against Rock:** 
  * When the Rock stands pat (draw 0), Model B responds with 50% call and 43% raise (treating it as a standard strong-hand battle).
  * When the Rock draws exactly **1** card, Model B's call and raise frequencies **collapse to 0%** (folding 100% of the time). In training, a Rock drawing 1 card was highly correlated with a premium draw (e.g., drawing to a straight or flush that has hit). Without any explicit hand-range rules, Model B learned to fold immediately. **This is the "Tell."**

---

### 3. Non-Stationary Adaptation Tournament
We evaluated the models in a **1,500-hand tournament** where the opponent silently transitions archetypes mid-session:
$$\text{Calling Station (Hands 1–500)} \rightarrow \text{Maniac (501–1,000)} \rightarrow \text{Rock (1,001–1,500)}$$

We compared Model B against two Model A baselines:
* **Model A (Oracle):** Received the correct, updated one-hot archetype label instantly at each switch.
* **Model A (Blind):** Remained stuck with the initial "Calling Station" label throughout the tournament.

| Agent | BB/100 |
| :--- | :---: |
| **Model B (Implicit)** | **$+160.35$** |
| **Model A (Oracle)** | $+156.28$ |
| **Model A (Blind)** | $+50.01$ |

> [!TIP]
> **Why Model B outperforms the Oracle:**
> During transitions, the Oracle immediately swaps its policy weights. However, the policies are trained on pure static archetypes. Model B's rolling statistics update gradually ($\sim$50–100 hands, as seen in the telemetry figures below). This creates a smoothed, hybrid state representation during transition windows, which better aligns with the policy network's interpolation capabilities.

<p align="center">
  <img src="latest%20results/non_stationary_inference.png" width="48%" />
  <img src="latest%20results/non_stationary_performance.png" width="48%" />
</p>

---

## Discussion & Limitations

1. **Static Opponents:** The scripted archetypes play fixed, non-adaptive strategies. While Model B effectively exploits their Tells, it has not been tested against self-play models or human players who might adjust their own strategies dynamically.
2. **Cold-Start Phase:** Model B requires roughly 30–50 hands to populate its rolling buffer with representative statistics. During these initial hands, its performance is sub-optimal.
3. **Multiplayer Scalability:** Expanding this methodology to multiplayer games would require tracking separate rolling statistics buffers for every opponent, significantly increasing observation space dimensionality.

## Interactive P2P Game & Playing Against the RL Agent

CardShark-RL includes a complete, playable web-based game client (**[poker.html](file:///c:/Users/efe20/Desktop/Inventory/Programming/Python/Uni%20CENG/AI%20&%20DS/cardshark-rl/poker.html)**) that allows you to play 5-Card Draw Poker against other players over PeerJS, or against the trained **Model B** reinforcement learning agent itself.

### How it Works:
1. **The Web Client ([poker.html](file:///c:/Users/efe20/Desktop/Inventory/Programming/Python/Uni%20CENG/AI%20&%20DS/cardshark-rl/poker.html)):** An interactive HTML5/JavaScript UI built with vanilla CSS. It handles lobby creation, cards rendering (with click-to-select discard behaviors), actions (Fold, Check/Call, Raise, Discard), side pot distribution, and game log messages.
2. **P2P Connection:** PeerJS is used to establish direct WebRTC connections between browsers using generated Room IDs, meaning two human players can play against each other natively.
3. **The RL Agent Server ([npc_server.py](file:///c:/Users/efe20/Desktop/Inventory/Programming/Python/Uni%20CENG/AI%20&%20DS/cardshark-rl/npc_server.py)):** A Flask microservice that acts as an interface for the trained Stable-Baselines3 model.
   - It loads the trained policy weights (`models/model_b_implicit.zip`).
   - It maintains the rolling history of the opponent's behavior to calculate VPIP, Aggression Factor, and other tells.
   - The web client calls the `/npc_action` endpoint, providing the current hand and game state. The server normalizes the observation, runs the action mask, predicts the action via the model, and returns it to the client.

---

## How to Run & Play

### Play Against the RL Agent (Model B)
To play a live game against the trained reinforcement learning model:

1. **Install Dependencies:**
   Ensure Flask and Flask-CORS are installed along with the RL environment requirements:
   ```bash
   pip install Flask flask-cors sb3-contrib stable-baselines3 numpy
   ```

2. **Start the NPC Server:**
   Run the Flask server which serves the trained Model B agent:
   ```bash
   python npc_server.py
   ```
   *The server will run on `http://127.0.0.1:5000`.*

3. **Open the Game Client:**
   Open **[poker.html](file:///c:/Users/efe20/Desktop/Inventory/Programming/Python/Uni%20CENG/AI%20&%20DS/cardshark-rl/poker.html)** in any modern web browser.

4. **Setup the Match:**
   * Enter your name in the input box.
   * Click **Create Room**.
   * Click the orange **Add NPC** button. This spawns the "CardShark Model B" agent, which connects to your running Python server.
   * Click the red **Deal Hand** button to start the game!

---

## Retraining & Evaluation

If you want to run evaluations or retrain the models:
* Refer to the main training scripts: `train.py` and `hpo.py`.
* Evaluate policies using `evaluate.py` or run visual simulations with `visualizer.py`.
* Install requirements via `pip install -r requirements.txt`.