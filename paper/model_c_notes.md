# CardShark-RL: Model C (Superhuman Autonomous Agent) Technical Brief

**Course:** CENG 454: Reinforcement Learning (Final Project)  
**Authors:** Alkım Gönenç Efe, Sarp Sünbül, Damla Parlakyıldız  
**Department of Computer Engineering, Izmir Katip Celebi University**

---

## 1. Executive Summary

| Attribute | Model A (Heads-Up) | Model B (Multiplayer) | Model C (Superhuman) |
| :--- | :---: | :---: | :---: |
| **Table Format** | 2 Seats (Heads-Up) | 5 Seats | 5 Seats |
| **Observation Dimensions** | 35 | 63 | **87** |
| **Action Space** | Discrete(38) | Discrete(38) | Discrete(38) |
| **Sparring Opponents** | 1 Static Bot | 5 Heuristic Bots | **Multi-Agent League (7 Checkpoints + 5 Bots)** |
| **Opponent Profiling** | None | Lifetime Beta-Binomial Prior | **Dual-Timescale Recency ($\lambda = 0.90$)** |
| **Action History Memory** | None | None | **Pre/Post-Draw Sequences & Trap Lines** |
| **1st Place Win Rate** | 50.0% (HU) | 26.0% | **42.0%** (Table Parity: 20.0%) |
| **Winrate (BB/100)** | +42.10 | +0.00 | **+77.97** |
| **Tournament Duration** | 31.0 hands | 67.4 hands | **49.3 hands** (Lethal Elimination) |

---

## 2. Core Architectural Pillars of Model C

### 2.1 Multi-Agent Fictitious Play League
- Table seats are populated 50% by neural league checkpoints and 50% by stochastic heuristics.
- Initialized with frozen `models/model_b_multiplayer.zip`.
- Checkpoints saved every 250,000 steps (`model_c_step_{N}.zip`) are automatically added to the pool, producing a 7-bot league pool by 1.5M steps.
- **Why it matters:** Closes bluffing loopholes, teaches 3-bet defense, and prevents policy collapse.

### 2.2 87-Dimensional Intra-Hand Sequence Vector
- $63 \text{ base features}$ + $8 \text{ pre-draw history}$ + $8 \text{ post-draw history}$ + $8 \text{ line combinations \& tilt delta}$.
- Directly classifies deceptive patterns:
  - **Check-Raise Traps:** Passive pre-draw, 1–2 card draw, then aggressive post-draw shove.
  - **Pat Value Bets:** Stand pat (0 discard) and large post-draw sizing.
  - **Multi-Card Draw Bluffs:** 3-card discard paired with aggressive pot bets.
- Scale-invariant: all chip commitments normalized by total table chips $C_{\text{table}}$.

### 2.3 Dual-Timescale Recency Tracker ($\lambda = 0.90$)
- **Lifetime Prior:** Beta-Binomial accumulation ($w_0 = 5.0$) defining the player's core identity.
- **Micro Recency Tracker:** Exponential discount forgetting ($\lambda = 0.90$, horizon $H = 10$ hands).
- **Tilt Detection:** Flags behavioral gear shifts when $\Delta_{\text{tilt}} > 0.25$ within 2–3 hands.

---

## 3. Key Findings for Paper / Presentation

1. **Massive Overperformance above Parity:** Model C's $42.0\%$ 1st place rate is $+110\%$ above natural 5-player table parity ($20.0\%$), and $+61.5\%$ above Model B ($26.0\%$).
2. **Clinical Elimination ("The Finisher Effect"):** Tournaments resolved 18.1 hands faster ($49.3$ vs $67.4$ hands) because Model C identifies opponent tilt and isolates weak ranges with value raises rather than letting games drag out.
3. **Dual Difficulty Modes Ready for Web Game:**
   - **Normal Mode:** Powered by Model B (`models/model_b_multiplayer.zip`).
   - **Hard / Boss Mode:** Powered by Model C (`models/model_c_superhuman.zip`).
