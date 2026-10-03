# CardShark-RL: Chipzen Competition Strategic Roadmap & Multi-Mode Architecture Plans

This document provides a comprehensive technical blueprint for adapting, training, and deploying competitive AI poker models across all **four Chipzen game modes** using **all four Chipzen platform integration pathways**.

---

## 1. Chipzen Platform Stack Integration Matrix

Chipzen offers four ways to deploy and compete. Below is how the **CardShark-RL** ecosystem maps to each deployment target across our core target modes:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               CHIPZEN COMPETITION STACK                                │
├─────────────────────────┬──────────────────────────┬───────────────────────────────────┤
│ Integration Pathway     │ CardShark-RL Architecture│ Best-Fit Game Modes               │
├─────────────────────────┼──────────────────────────┼───────────────────────────────────┤
│ 1. Upload (SDK)         │ MaskablePPO + PyTorch    │ • 2-7 Triple Draw (Sub-5ms)       │
│    (Python/Docker)      │ Optimized Standalone     │ • NLHE 6-Max / Heads-Up (Local)   │
├─────────────────────────┼──────────────────────────┼───────────────────────────────────┤
│ 2. Agent (MCP)          │ Neuro-Symbolic Agent     │ • NLHE Table Exploit / Meta-Game  │
│    (Model Context Proto)│ (LLM + CardShark Tools)  │ • Live Dynamic Range Adjustment   │
├─────────────────────────┼──────────────────────────┼───────────────────────────────────┤
│ 3. Remote (External API)│ Dedicated Workstation/GPU│ • NLHE Heavy Ensembles / Search   │
│    (Wire API / Token)   │ Hot-Reload Self-Play     │ • Live Fictitious Play Retraining │
├─────────────────────────┼──────────────────────────┼───────────────────────────────────┤
│ 4. Starter Bot          │ Heuristic Calibrator     │ • Quick baseline sparring         │
│    (No-Code Knob Tuning)│ Parameter Bounds Probe   │ • League seeding & test benches   │
└─────────────────────────┴──────────────────────────┴───────────────────────────────────┘
```

### Deep Dive: Deploying Across the 4 Stack Modes

#### Option 1: Upload (SDK — Python Docker Image)
- **Mechanism:** Package trained weights (`model_d.zip` or Hold'em policy), PyTorch runtime, and inference engine into a lightweight Docker image. Chipzen runs it in a sandboxed, low-latency container.
- **Advantages:** Sub-5ms decision latency, zero network dropouts, deterministic execution, zero ongoing compute costs on our side.
- **CardShark Fit:** Ideal for the final production champions of **2-7 Triple Draw**, **NLHE 6-Max**, and **NLHE Heads-Up**.

#### Option 2: Agent (MCP — Model Context Protocol)
- **Mechanism:** Chipzen exposes tables, seats, and live hands as **MCP Tools** (`get_table_state`, `get_legal_actions`, `submit_action`, `query_history`). An AI agent (Claude / Gemini / local agent) takes the seat directly over MCP.
- **Advantages:** Allows hybrid Neuro-Symbolic reasoning. The agent can invoke our specialized RL models or hand evaluators as sub-tools while doing high-level game-theoretic planning, psychological profiling, and meta-game bluff frequency modulation.
- **CardShark Fit:** Useful for dynamic table chatter and exploiting passive/calling-station bots in NLHE tournaments.

#### Option 3: Remote (External API — Wire Protocol with Long-Lived Token)
- **Mechanism:** Run an inference daemon on our local GPU workstation or cloud instance, connecting over WebSockets/REST with Chipzen's wire API.
- **Advantages:** Zero container size or RAM constraints. We can run large model ensembles, deep Monte Carlo Tree Search (MCTS) rollouts, turn/river subgame solvers, and even live-update weights or log opponent histories to our database without re-uploading containers.
- **CardShark Fit:** Ideal for **active training/evaluation phases**, running computationally heavy NLHE subgame resolvers, and rapid iterative testing.

#### Option 4: Starter Bot (No-Code Knob Tuning)
- **Mechanism:** Platform-provided rule engines with tunable aggression, tightness, and bluffing parameters.
- **CardShark Fit:** Used as baseline sparring benchmarks in our local league pool to calibrate our RL agents against standard human-like archetype distributions.

---

## 2. Game Mode Feasibility & Scope Decision

| Game Mode | Card Structure | Hidden Info | Core Dynamic | CardShark Code Reuse | Scope Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2-7 Triple Draw** | 5 Private Cards | Full (Draw counts visible) | Inverted Lowball, 3 Discard Phases | **95% (Near-Instant)** | **ACTIVE PRIORITY 1** |
| **NLHE (6-Max)** | 2 Hole + 5 Board | Partial (Community Cards) | Multi-Street Pot-Limit/No-Limit | **85% (High Overlap)** | **ACTIVE PRIORITY 2** |
| **NLHE (Heads-Up)** | 2 Hole + 5 Board | Partial (2-Player Zero-Sum) | Deep Nash Equilibrium, Wide Ranges | **85% (Same Engine)** | **ACTIVE PRIORITY 3** |
| **Pineapple OFC** | 13 Placed Cards | None (Open-Face Placement) | Spatial Point & Royalty Maximizer | **<20% (Disjoint)** | **OUT OF SCOPE (DROPPED)** |

> [!NOTE]
> **Why Pineapple OFC is Excluded:**
> Pineapple OFC is fundamentally **not poker in the game-theoretic sense**. There is no pot betting, no folding, no bluffing, and no hidden-information deception. It is an open-face spatial puzzle of optimizing 13 card placements to avoid foul conditions and maximize scoring royalties. 
> 
> Developing an OFC bot would require building an entirely separate solver stack with virtually zero architectural reuse. By dropping it, **100% of our compute, attention, and engineering time is channeled into the "Big Three" betting poker formats** where CardShark-RL possesses immediate, overwhelming competitive advantages.

---

## 3. Detailed Model Design Plans by Mode

---

### MODE 1: 2-7 Triple Draw (Deuce-to-Seven Lowball)
> **Strategic Value:** Highest Win-Rate Potential. Lowest Implementation Barrier.

#### 1. Game Dynamics
- Players are dealt 5 private cards.
- **4 betting rounds** separated by **3 draw rounds**.
- **Hand Ranking:** Inverted lowball. Straights and flushes *weaken* the hand. Aces are strictly high ($14$). The absolute best hand ("The Wheel") is:
  $$\mathbf{7\text{-}5\text{-}4\text{-}3\text{-}2} \quad (\text{unsuited})$$
- Draw actions: Discarding between $0$ (pat) and $5$ cards ($2^5 = 32$ discrete discard options).

#### 2. Model Approaches

##### Way A: Native Transformer PPO (CardShark Model D Extension) — *RECOMMENDED*
- **Architecture:** Keep the exact `CardAttentionExtractor` (Multi-Head Self-Attention over 5 card tokens).
- **Modifications:**
  1. **Lowball Hand Evaluator:** Invert evaluation order in `game/rules_27.py`.
  2. **Multi-Stage Draw Tracker:** Extend the observation vector to track draw round ($1, 2, \text{or } 3$) and opponent draw counts (pat, 1-card draw, 2-card draw, etc.).
  3. **Action Space:** 38 discrete actions (identical to CardShark: 6 betting sizes + 32 discard bitmasks).
- **Training Strategy:** Prioritized Fictitious Self-Play (PFSP) with cosine warm restarts over 2,000,000 steps.

##### Way B: Draw-Pattern Bayesian Counter-Strategy
- Track opponent discard behavior:
  - If an opponent stands pat on Draw 1, their hand distribution is conditioned on $\le 9$-high made hands.
  - If an opponent draws 1 card on Draw 3, calculate the exact probability of them completing an 8-or-better low.
- Feed Bayesian belief probabilities directly into the policy MLP input vector.

##### Way C: Fast Rollout Search (MCTS Hybrid for River/Draw 3)
- For the final betting round and final draw, run 200 Monte Carlo rollouts against estimated opponent ranges to choose the mathematically optimal discard or value bet.

---

### MODE 2: No-Limit Texas Hold'em (6-Max)
> **Strategic Value:** Flagship Competition Tier. High Prestige & Multi-Agent Complexity.

#### 1. Game Dynamics
- 2 private hole cards + 5 shared community cards (Flop: 3, Turn: 1, River: 1).
- 4 betting rounds: Pre-Flop, Flop, Turn, River.
- Multi-way pot dynamics with side pots and position advantage (Button, Cutoff, Small Blind, Big Blind, etc.).

#### 2. Model Approaches

##### Way A: Variable-Token Key-Padding Card-Attention PPO — *RECOMMENDED*
- **Token Input Representation:**
  - $7$ card tokens: 2 hole cards + up to 5 community cards.
  - Each token is a 5-dim embedding:
    $$\text{Token}_i = [\text{rank_norm}, \text{suit_one_hot (4 dims)}, \text{is_community_flag}]$$
  - Active tokens expand as streets progress:
    - Pre-Flop: 2 tokens (5 masked with PyTorch `key_padding_mask`)
    - Flop: 5 tokens (2 masked)
    - Turn: 6 tokens (1 masked)
    - River: 7 tokens (0 masked)
- **Multi-Head Attention:** Permutation-invariant attention across hole and board cards, extracting board texture (paired boards, flush draws, straight connectivity) automatically.
- **Bet Abstraction (Action Space):** 8 discrete strategic actions:
  - `0`: Fold
  - `1`: Check / Call
  - `2`: Min-Raise ($2\times$)
  - `3`: Small Bet ($33\%$ pot)
  - `4`: Standard Bet ($67\%$ pot)
  - `5`: Pot Bet ($100\%$ pot)
  - `6`: Overbet ($150\%$ pot)
  - `7`: All-In
- **Fast Evaluation:** Integrate `eval7` (Cython) or `treys` (pure Python bitwise 7-card evaluator, 0.5 microseconds per evaluation).

##### Way B: Deep CFR (Counterfactual Regret Minimization)
- Use Deep CFR or ReBeL-style policy-value networks.
- Advantages: Directly approximates Nash equilibrium in imperfect-information games.
- Drawback: Computationally heavier to train than Action-Masked PPO.

##### Way C: Neuro-Symbolic Hybrid (Pre-Flop GTO Matrix + Neural Post-Flop)
- Use pre-computed GTO opening/calling ranges for Pre-Flop decisions (169 canonical starting hand combinations).
- Switch to the CardShark Transformer PPO once the flop arrives.
- Eliminates wasteful pre-flop exploration during training.

---

### MODE 3: No-Limit Texas Hold'em (Heads-Up)
> **Strategic Value:** Game-Theoretic Purest Form. Rapid Nash Convergence.

#### 1. Game Dynamics
- 2 players: Button (Small Blind) vs Big Blind.
- Deep stacks, hyper-aggressive pre-flop play (Button VPIP typically $85\%\text{--}100\%$).
- Pure two-player zero-sum game: $\mathcal{R}_1 + \mathcal{R}_2 = 0$.

#### 2. Model Approaches

##### Way A: Heads-Up Specialization of Hold'em Transformer PPO — *RECOMMENDED*
- Configure the 6-Max engine with `num_seats=2`.
- **Symmetric Self-Play Training:** The agent plays directly against past checkpoints of itself.
- Nash equilibrium is mathematically guaranteed to be stable (unlike in 3+ player games where cyclic rock-paper-scissors dynamics can occur).
- Training convergence is approximately $3\times$ faster than 6-Max.

##### Way B: Real-Time Depth-Limited Subgame Solving
- For Turn and River decisions, solve the localized two-player zero-sum matrix game in real time using 20 iterations of CFR.
- Yields unexploitable bluff frequencies and optimal value-betting sizing.

---

### MODE 4: Pineapple OFC (Open-Face Chinese Poker) — [OUT OF SCOPE / DROPPED]
> **Status:** *De-scoped. Excluded from development pipeline.*

#### Rationale for Exclusion
- **Not a Betting Game:** Pineapple OFC contains no betting rounds, no pots, no folding, no bluffing, and no hidden card interactions.
- **Architectural Disconnect:** It is a spatial board placement problem (top 3 cards, middle 5 cards, bottom 5 cards) aiming to maximize royalties while avoiding foul states.
- **Resource Allocation:** Building a separate MCTS puzzle solver or board-placement DQN would offer $<20\%$ code reuse from CardShark. Dropping OFC enables **100% of GPU compute and engineering focus** to go into dominating **2-7 Triple Draw**, **NLHE 6-Max**, and **NLHE Heads-Up**.

---

## 4. Cross-Mode Architectural Comparison (Target Modes)

| Dimension | 2-7 Triple Draw | NLHE (6-Max) | NLHE (Heads-Up) |
| :--- | :--- | :--- | :--- |
| **Status** | **Priority 1 (Sprint 1)** | **Priority 2 (Sprint 2)** | **Priority 3 (Sprint 2)** |
| **Observation Dims** | 87 dims | 92 dims | 74 dims |
| **Card Tokens** | 5 cards (private) | 2 hole + 5 community | 2 hole + 5 community |
| **Attention Type** | Standard MHSA (5 tokens) | Masked MHSA (7 tokens with key padding) | Masked MHSA (7 tokens with key padding) |
| **Action Space** | 38 discrete (6 bet + 32 discard) | 8 discrete geometric bet sizes | 8 discrete geometric bet sizes |
| **Hand Evaluator** | Lowball 2-7 lookup | `eval7` / `treys` (7-card bitwise) | `eval7` / `treys` (7-card bitwise) |
| **Recommended Deployment**| **Upload SDK (Docker)** | **Upload SDK / Remote API** | **Upload SDK / Remote API** |

---

## 5. Deployment Implementation Blueprints

### Blueprint 1: Upload (SDK — Dockerfile & Wrapper)
```dockerfile
# Dockerfile for Chipzen Upload SDK
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ /app/models/
COPY game/ /app/game/
COPY rl/ /app/rl/
COPY bot_upload.py /app/bot.py

CMD ["python", "bot.py"]
```

```python
# bot_upload.py (Chipzen SDK Entry Point)
from chipzen_bot import Bot, Action
from sb3_contrib import MaskablePPO
import numpy as np

class CardSharkUploadBot(Bot):
    def __init__(self):
        # Load optimized frozen policy
        self.model = MaskablePPO.load("models/champion.zip", device="cpu")

    def decide(self, state) -> Action:
        obs = self.extract_features(state)
        mask = self.extract_action_mask(state)
        action_idx, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        return self.to_chipzen_action(action_idx, state)

if __name__ == "__main__":
    CardSharkUploadBot().run()
```

---

### Blueprint 2: Agent (MCP — Model Context Protocol Bridge)
```python
# mcp_agent_server.py
"""
Exposes CardShark-RL evaluation engines and neural policies as MCP tools
for Claude / GPT / Gemini agents competing in Chipzen's Agent Arena.
"""
from mcp.server.fastmcp import FastMCP
import numpy as np

mcp = FastMCP("CardShark-Chipzen-MCP")

@mcp.tool()
def evaluate_holdem_hand(hole_cards: list[str], board_cards: list[str]) -> dict:
    """Evaluates 7-card hand strength, percentile rank, and drawing outs."""
    # Symbolic Treys evaluator
    ...
    return {"rank": "Two Pair, Aces and Kings", "strength_percentile": 0.84}

@mcp.tool()
def get_optimal_rl_action(game_mode: str, table_state: dict) -> dict:
    """Queries CardShark-RL neural model for the game-theoretic optimal action."""
    ...
    return {"action": "RAISE_HALF_POT", "confidence": 0.91, "expected_value": +2.4}

@mcp.tool()
def get_opponent_profile(opponent_name: str) -> dict:
    """Retrieves Bayesian VPIP, PFR, and fold-to-3bet tendencies from tracker."""
    ...
    return {"archetype": "Loose-Passive (Calling Station)", "exploit_recommendation": "Value-bet thinly, never bluff"}

if __name__ == "__main__":
    mcp.run()
```

---

### Blueprint 3: Remote (External API — Wire Protocol Client)
```python
# remote_worker.py
"""
Connects from local GPU workstation to Chipzen platform using long-lived API token.
Allows hot-reloading weights, running heavy search rollouts, and live logging.
"""
import asyncio
import websockets
import json

CHIPZEN_WS_URL = "wss://api.chipzen.com/v1/arena/ws"
API_TOKEN = "your_long_lived_token_here"

async def run_remote_bot():
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    async with websockets.connect(CHIPZEN_WS_URL, extra_headers=headers) as ws:
        print("[Remote Worker] Connected to Chipzen Wire API.")
        async for message in ws:
            event = json.loads(message)
            if event["type"] == "ACTION_REQUEST":
                action = compute_cardshark_action(event["state"])
                await ws.send(json.dumps({"type": "SUBMIT_ACTION", "action": action}))

if __name__ == "__main__":
    asyncio.run(run_remote_bot())
```

---

## 6. Execution Roadmap & Timeline

```
Sprint 1 (Days 1–3)                  Sprint 2 (Days 4–7)                  Sprint 3 (Day 8+)
┌──────────────────────────────┐     ┌──────────────────────────────┐     ┌──────────────────────────────┐
│  Phase 1: 2-7 Triple Draw    │ ──► │  Phase 2: No-Limit Hold'em   │ ──► │  Phase 3: Chipzen Arena      │
│  • Lowball Evaluator Invert  │     │  • 7-Card Token Transformer  │     │  Integration & Deployment    │
│  • 3-Draw State Machine      │     │  • Multi-Street Betting      │     │  • Docker Upload Submission  │
│  • PFSP 2.0M Training        │     │  • 6-Max & Heads-Up Runs     │     │  • Live Arena Testing        │
└──────────────────────────────┘     └──────────────────────────────┘     └──────────────────────────────┘
```

### Milestone Checklist:
- [x] **Architecture Plan & Platform Strategy** (`plans.md` created)
- [ ] **Phase 1: 2-7 Triple Draw Engine** (`game/rules_27.py`, 3-draw environment)
- [ ] **Phase 1 Training:** Run 1.5M step PFSP lowball champion
- [ ] **Phase 2: Hold'em Engine:** Integrate `treys` / `eval7`, build 4-street state machine
- [ ] **Phase 2 Training:** Train 6-Max and Heads-Up CardAttention models
- [ ] **Phase 3: Deployment Bridges:** Build `Upload SDK` Dockerfile and `Agent MCP` server
- [ ] **Live Competition:** Enter Chipzen tournament arena

