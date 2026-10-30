# CardShark-RL: Autonomous Reinforcement Learning for Card Games and Tournament Play

**Comprehensive System Architecture, Methodology Implementation, and Academic Research Plan**  
*Institution:* İzmir Kâtip Çelebi University, Department of Computer Engineering
*Project Team:* Alkım Gönenç Efe, Sarp Sünbül, Damla Parlakyıldız 

---

## Table of Contents
1. [Project Overview & Research Motivation](#1-project-overview--research-motivation)
2. [What Sets This Study Apart From Existing Poker AI](#2-what-sets-this-study-apart-from-existing-poker-ai)
3. [Core Methodology & Implementation Details](#3-core-methodology--implementation-details)
4. [Progressive Model Iterations (Texas Hold'em Models A through G)](#4-progressive-model-iterations-texas-holdem-models-a-through-g)
5. [Multi-Game Generalization: 5-Card Draw & 2-7 Triple Draw](#5-multi-game-generalization-5-card-draw--2-7-triple-draw)
6. [Academic Paper Focus & Planned Ablation Studies](#6-academic-paper-focus--planned-ablation-studies)

---

## 1. Project Overview & Research Motivation

CardShark-RL is an autonomous reinforcement learning framework engineered to solve multi-player, imperfect-information card games. The project has the primary goal of investigating whether model-free deep reinforcement learning algorithms can learn sound, adaptive tournament strategies on standard commodity hardware, without relying on heavy real-time tree solvers or massive computing clusters.

Card games represent a classic imperfect-information challenge in artificial intelligence. Unlike board games such as chess or Go, where both players have full visibility over the board state, a poker player only observes their private hand and the sequence of public actions. Furthermore, games are characterized by high statistical variance across deck deals, non-stationary multi-agent dynamics (where players adapt to one another), and shifting risk profiles as tournament blinds escalate. A successful agent cannot simply memorize hand strengths; it must learn to infer hidden information from betting patterns, exploit predictable opponent archetypes, and manage chip survival under tournament pressure.

This document serves two primary functions: first, as an accessible technical reference bringing all team members and co-authors up to speed on the system's architecture and design decisions; and second, the foundation for our upcoming paper.

---

## 2. What Sets This Study Apart From Existing Poker AI

Extensive academic literature exists on card game artificial intelligence, most notably landmark systems such as Libratus, DeepStack, and Pluribus. While these systems achieved milestone results against elite human professionals, CardShark-RL is situated in a distinct practical and scientific space:

- **Lightweight Model-Free Deep RL vs. Heavy Real-Time Tree Search:**  
  Systems like Pluribus and Libratus rely on Counterfactual Regret Minimization (CFR) paired with real-time subgame solving. During live play, those engines pause for several seconds per decision to compute fine-grained equilibrium approximations across deeply branched subgame trees. This demands massive distributed memory (hundreds of gigabytes of RAM) and multi-GPU server infrastructure. In contrast, CardShark-RL relies on pure model-free Deep Reinforcement Learning (Proximal Policy Optimization). Once trained, our neural network evaluates state vectors and selects actions in approximately one millisecond on a standard laptop CPU, demonstrating that competitive card play can be achieved in a lightweight, reactive footprint.
- **Multi-Player Tournament Play vs. Heads-Up Cash Games:**  
  The vast majority of academic card AI research focuses exclusively on two-player (heads-up) cash games with static chip stacks, where chip accumulation is linearly proportional to reward. CardShark-RL addresses full 5-player tournament play. Tournaments introduce non-linear survival incentives: losing one's remaining chips results in permanent elimination, while periodically escalating blinds force players to continuously adapt their strategies from deep-stack post-flop maneuvering to short-stack push-or-fold survival.
- **Architectural Solutions to Reinforcement Learning Failure Modes:**  
  Rather than mitigating deep RL instability through external solvers, we address classic failure modes (such as bet-sizing collapse and extreme over-folding) directly through architectural innovations: order-independent card attention, two-stage decoupled action policies, and self-play league sparring pools.
- **Extensible Multi-Game Engine:**  
  CardShark-RL is not an isolated Texas Hold'em bot. The repository implements a unified environment framework that supports both community card games (Texas Hold'em) and draw poker variants (5-Card Draw and 2-7 Triple Draw), highlighting the generalizability of our feature extraction and training pipeline.

---

## 3. Core Methodology & Implementation Details

The CardShark-RL framework addresses the game-theoretic and computational challenges of card play through modular, specialized mechanisms. Below is an overview of how each methodology was designed and implemented:

### 3.1 Order-Independent Hand Processing (Card Attention Feature Extractor)
**The Problem:** In card games, the physical order in which cards are dealt has no bearing on hand strength: holding Ace-King is mathematically identical to King-Ace. Standard multi-layer perceptrons treat card inputs as a fixed flat vector, which artificially forces the network to learn duplicate representations for every possible permutation of the same hand.

**How It Was Implemented:** Each card in the player's private hand and on the public board is treated as an individual entity token. The card's integer representation is mapped through two parallel embedding layers—one encoding rank (Two through Ace) and one encoding suit (Clubs, Diamonds, Hearts, Spades)—which are projected into a 64-dimensional dense vector. These card tokens are fed through a 4-head self-attention layer. The self-attention mechanism computes pairwise affinity weights between all cards simultaneously. This allows the network to naturally detect poker patterns—such as paired ranks, suited connectors, or straight draws—regardless of the slot each card occupied. The attended card vectors are averaged via mean-pooling into a single hand summary representation and concatenated with the table state vector.

### 3.2 Separating Intent from Bet Magnitude (Decoupled Hierarchical Action Policy)
**The Problem:** In No-Limit poker, players can choose from many different bet sizes. When a standard policy outputs a single flat list of actions containing both basic moves (Fold, Call) and all possible bet sizes (Half-Pot, Full-Pot, Shove), exploratory large bets early in training frequently lose chips on weak hands. The network receives strong negative feedback on those action indices and permanently suppresses their probabilities. The policy collapses into passive check-folding or repetitive minimum-raising.

**How It Was Implemented:** We restructured the policy network by splitting the output layers into two separate branches sharing a common latent torso. The first branch (the Core Intent head) outputs probability distributions over high-level strategic actions: Fold, Check/Call, Raise, and All-In. The second branch (the Sizing head) outputs probability distributions over fractional pot sizes (33%, 50%, 75%, 100%, and 150% of the pot). During action execution, if the core head chooses Fold, Call, or All-In, the action executes immediately without querying the sizing head. Only when the core head selects Raise is the sizing head evaluated to determine the exact bet magnitude. During backpropagation, sizing gradients are masked so that sizing parameters are only updated on steps where a raise occurred. This decouples the exploration of bet sizing from the decision to bet, eliminating bet-sizing collapse.

### 3.3 Opponent Profiling and Active Exploitation
**The Problem:** Fixed baseline strategies leave substantial value on the table when playing against flawed opponents. An effective agent must recognize opponent habits—such as an opponent who folds too often to aggression, or a passive player who refuses to fold weak pairs—and dynamically alter its play to exploit those weaknesses.

**How It Was Implemented:** The environment maintains running historical statistics for every seat at the table across consecutive hands. It tracks metrics including Voluntarily Put Money in Pot (VPIP, measuring loose vs. tight pre-flop play), Pre-Flop Raise frequency (PFR, measuring pre-flop aggression), and Fold-to-Bet frequencies across each street. To handle the low sample sizes inherent in short tournament sessions, Bayesian updates smooth observed counts toward standard prior distributions. The resulting statistical profiles are categorized into opponent archetypes (such as Tight-Aggressive, Loose-Aggressive, or Calling Station) and directly concatenated into the observation vector. Furthermore, the reward function includes calibrated shaping bonuses that reward the agent for executing mathematically sound exploitative lines, such as bluffing against players with high fold frequencies or extracting value bets against calling stations without bluffing.

### 3.4 Self-Play League Sparring & Dynamic Opponent Curriculum
**The Problem:** Training an agent exclusively against static, rule-based bots causes severe overfitting: the agent learns to exploit the hardcoded programming quirks of those bots, but completely falls apart when exposed to novel styles or aggressive neural counter-strategies. Conversely, exposing an untrained network immediately to unconstrained self-play often leads to policy destabilization or bizarre exploratory loops before basic card fundamentals are established.

**How It Was Implemented:** We implemented an autonomous self-play league pool governed by a dynamic curriculum schedule. Training begins with an emphasis on diverse heuristic bots (tight-aggressive, loose-aggressive, and passive calling stations) to ground the policy in solid poker discipline—such as folding unplayable holdings pre-flop, respecting aggressive betting, and extracting value from calling stations. As training progresses, the opponent distribution systematically shifts toward self-play league sparring. Checkpoints of the learning policy are saved periodically into the pool. By late-stage training, the vast majority of sparring occurs against past neural iterations and anchor champions sampled via Prioritized Fictitious Self-Play. This curriculum ensures that the agent first masters fundamental game discipline before refining high-level counter-strategies against adaptive neural adversaries.

### 3.5 Tournament Chip Valuation (Independent Chip Model / ICM Equity)
**The Problem:** In cash games, chips have constant linear value. In tournament poker, however, chips lost are worth more than chips gained: losing one's last chip knocks a player out of the tournament, whereas doubling one's chip stack does not double the probability of winning first place. Agents that evaluate chip counts linearly take reckless gambles on marginal hands and bust out prematurely.

**How It Was Implemented:** We integrated tournament equity calculations based on the Independent Chip Model (ICM). The environment translates raw chip stack distributions into actual tournament prize equities. The reward signal provided to the agent is scaled by changes in tournament equity rather than simple raw chip differences. This penalizes reckless stack erosion when playing with a short stack and naturally encourages disciplined survival strategies near elimination boundaries.

### 3.6 Auxiliary Board Reading Supervision (Auxiliary Belief Head)
**The Problem:** Reinforcement learning agents often require millions of exploratory hands simply to understand how community card textures relate to hand categories (such as recognizing when a board is dangerous due to completed flushes or straights).

**How It Was Implemented:** We attached an auxiliary classification head branching directly from the shared torso representation. During training, this auxiliary head is trained under supervised loss to predict board texture danger tiers (e.g., paired boards, monotone boards) and opponent hand strength categories. The resulting supervised auxiliary loss gradients accelerate representation learning in early layers, helping the network learn meaningful card representations significantly faster than relying on policy return gradients alone.

### 3.7 Dynamic Blind Escalation & Uncontested Pot Incentives
**The Problem:** In tournament simulations, if blind escalation freezes at a low level, deep chip stacks pass chips back and forth indefinitely, causing games to hit artificial turn limits and turn into arbitrary tie-breaks. Furthermore, agents trained purely on showdown chip results frequently become timid, checking down strong hands to avoid risk and suffering unnecessary luck variance.

**How It Was Implemented:** The table engine implements an uncapped blind schedule that doubles the blinds after fixed hand intervals, escalating through multiple blind tiers. This dynamically compresses stack depths as the tournament progresses, forcing the policy to smoothly transition from deep-stack play to short-stack push-or-fold endgame dynamics without hitting timeouts. Additionally, we introduced an uncontested pot bonus that awards a small incentive whenever the agent wins a pot by forcing all active opponents to fold before showdown. This teaches the agent to apply healthy pressure and accumulate uncontested chips when opponents display weakness.

### 3.8 Two-Phase Curriculum & Reward Annealing
**The Problem:** Heuristic reward shaping guides early exploration, but if left active indefinitely, the agent over-optimizes for intermediate bonuses (such as folding rewards or exploratory bluffing bonuses) rather than overall tournament success.

**How It Was Implemented:** We implemented a scheduled curriculum callback that manages reward shaping weights across two distinct phases. In Phase 1 (Scaffolding), auxiliary shaping rewards are fully active to accelerate policy stabilization and prevent blunders. In Phase 2 (Competition), a decay schedule systematically scales down these shaping multipliers toward zero over several hundred thousand steps. By the final stages of training, the agent optimizes purely for raw tournament outcomes and chip accumulation, eliminating the risk of reward hacking.

### 3.9 Observation Vector Anatomy (Model F Feature Decomposition)
To provide complete transparency into the agent's sensory input, the 168-dimensional observation vector for Model F is structured into six functional tiers:

| Tier | Dimensions | Feature Components |
| :--- | :---: | :--- |
| **1. Core Table & Card State** | 73 | Private hole cards (4), community board cards (10), evaluated hand strength & Chen preflop score (1), pot odds, pot size, hero chips, and bet-to-call ratios (4), street indicators (4 one-hot), position relative to dealer button (5 one-hot), per-seat chip and status tracking across 5 seats (25), and round action history across 4 streets (20). |
| **2. Opponent Profiling & Board Textures** | 52 | ICM chip geometry and shortest stack ratios (3), board texture engine evaluating pairs, monotone boards, and straight/flush draws (17), and dual-timescale Bayesian opponent tracking tracking VPIP, PFR, fold-to-cbet, and aggression across fast and slow windows for all 4 opponents (32). |
| **3. Tournament Dynamics & GTO Baselines** | 13 | Pre-flop GTO positional baseline guidance (6), post-flop stack-to-pot ratio (SPR), showdown value classification, in-position advantage flag, range aggressor indicator (4), and non-linear ICM tournament equity valuations (3). |
| **4. Targeted Opponent Exploitation** | 14 | Real-time exploitability vectors evaluating opponent folding vulnerability, calling station value extraction opportunities, steal feasibility, and dynamic opponent archetype classification. |
| **5. Self-Play & Multi-Agent Dynamics** | 6 | Pre-flop 3-bet squeeze intensity, check-raise volatility index, effective stack depth in big blinds, push/fold regime trigger, multiway pot indicator, and neural league sparring opponent flag. |
| **6. Tournament Closer & Endgame Regimes** | 10 | Hands remaining in current blind level, blind escalation urgency, hero chip leaderboard rank, Harrington M-factor, Harrington M-Zone (Red/Orange/Yellow/Green), heads-up regime flag, podium/bubble pressure flag, Sklansky-Chubukov push urgency, fold-to-cbet steal indicator, and chip dominance ratio over second stack. |
| **Total** | **168** | **Complete state vector fed to Card Attention and policy networks.** |

### 3.10 Tournament Structure & Benchmark Evaluation Protocol
To evaluate agents in an ecologically valid competitive setting rather than synthetic isolated hands, the framework implements an end-to-end Sit-and-Go tournament simulator:

- **Tournament Mechanics:**
  - **Table Format:** 5-player Sit-and-Go tournament with winner-take-all / podium incentives.
  - **Chip Stacks:** Each player starts with 1,000 chips (5,000 total table chips in play).
  - **Blind Schedule:** Starts at Small Blind = 5 / Big Blind = 10, doubling every 20 hands across seven uncapped tiers (10/20, 20/40, 40/80, 80/160, 160/320, 320/640, 640/1280) to force natural endgame transitions without artificial turn timeouts.
- **Benchmark Opponent Roster:**
  - `TAGBot` (Tight-Aggressive): Selective pre-flop opening ranges; applies strong post-flop pressure on high equity.
  - `LAGBot` (Loose-Aggressive): Wide pre-flop opening ranges; frequent continuation betting and aggressive float lines.
  - `CallingStationBot`: Highly passive calling station; rarely folds post-flop, calls bets with marginal showdown value.
  - `RandomHoldemBot`: Stochastic random player introducing unpredictable betting noise and extreme entropy.
- **Evaluation Metrics:**
  - **1st-Place Win Rate:** Percentage of completed tournament sessions won outright (first place).
  - **Podium Finish Rate:** Percentage of tournaments finishing in the top 2 (heads-up) or top 3.
  - **Average Net Chips:** Net chip accumulation relative to the 1,000-chip starting stack.
  - **Composite Evaluation Score:** Balanced metric combining titles, podiums, chip efficiency, and survival:  
    $$\text{Score} = (\text{Wins} \times 5.0) + (\text{Podiums} \times 2.0) + (\text{Net Chips} \times 0.05) - (\text{Eliminations} \times 1.0)$$

---

## 4. Progressive Model Iterations (Texas Hold'em Models A through G)

Texas Hold'em models were evolved across seven sequential generations (Model A through Model G). Each iteration was designed to diagnose and resolve specific behavioral and game-theoretic flaws observed in previous versions:

- **Model A (Baseline Architecture):** Implemented as the initial proof-of-concept. Used a 73-dimensional state vector and a flat 6-action discrete space, trained exclusively against static rule-based bots. While it learned basic hand rankings and fundamental rules, it exhibited severe passivity, folded excessively under any aggression, and suffered from bet-sizing collapse.
- **Model B (Opponent Profiling):** Expanded the observation space to 125 dimensions by incorporating Bayesian tracking of opponent VPIP, PFR, and fold tendencies. This enabled the agent to distinguish between tight and loose opponents, allowing it to open wider against timid players and fold against tight raisers.
- **Model C (Tournament Awareness):** Expanded the observation space to 138 dimensions by integrating tournament survival logic, blind escalation ratios, and stack-depth metrics. The model learned stack preservation, shifting to conservative push-or-fold tactics when short-stacked rather than bleeding chips through calling.
- **Model D (Active Exploitation):** Expanded state inputs to 152 dimensions with explicit opponent exploitability signals. The agent began actively altering its post-flop lines based on opponent archetypes, consistently value-betting against calling stations and executing targeted bluffs against frequent folders.
- **Model E (Self-Play & Decoupled Policy):** Expanded inputs to 158 dimensions and introduced two major architectural milestones: the Prioritized Fictitious Self-Play league pool and the two-stage decoupled action policy. Decoupling strategic intent from sizing eliminated bet-sizing collapse, while self-play sparring eliminated the agent's vulnerability to varied opponent counter-strategies.
- **Model F (Fine-Grained Actions, Uncapped Blinds & Boundary Screening):** Expanded observation inputs to 168 dimensions and the action space to 8 discrete actions (introducing a 33% micro-continuation bet and a 150% overbet). Uncapped blind escalation up to level 7 was introduced to eliminate tournament stalling, and hyperparameter boundaries were systematically screened across learning rates, entropy schedules, and reward shaping coefficients to ensure stable policy convergence.
- **Model G (1v1 Heads-Up Specialist):** A dedicated heads-up specialist trained exclusively in two-player environments under pure linear chip-EV, 100 big blind deep stacks, and aggressive button exploitation. By removing tournament ICM survival penalties and utilizing a dynamic curriculum that transitions from heuristic fundamentals to self-play dominance, Model G learns wide button opening ranges, persistent continuation betting, and aggressive endgame extraction without fear of tournament elimination.

### The Composite Model Architecture
To resolve the fundamental objective mismatch between multi-player tournament survival (where chips lost are worth more than chips won) and heads-up endgame play (where the game reverts to zero-sum linear chip extraction), CardShark-RL introduces a modular **Composite Model**. 

Rather than hardcoding specific networks, the Composite Model provides a flexible, plug-and-play architecture with two explicit policy slots:
1. **Tournament Model Slot:** Operates whenever three or more competitors remain at the table, executing disciplined stack preservation and ICM bubble survival (e.g., slotted with Model F).
2. **Heads-Up Model Slot:** Automatically engages the moment the tournament narrows to the final two competitors, unleashing uninhibited button aggression and knockout pressure to capture the title (e.g., slotted with Model G).

### Summary Table of Model Lineage

| Model | State Dims | Action Space | Key Methodologies Added | Primary Behavioral Capability / Issue Resolved |
| :--- | :---: | :---: | :--- | :--- |
| **Model A** | 73 | 6 (Flat) | Baseline PPO, Card Attention Extractor | Learned valid hand rules; suffered from severe passivity and sizing collapse |
| **Model B** | 125 | 6 (Flat) | Bayesian Opponent Tracking | Learned opponent profiling; differentiated tight from loose pre-flop play |
| **Model C** | 138 | 6 (Flat) | ICM Equity & Stack Depth Ratios | Learned chip survival and stack preservation under escalating blinds |
| **Model D** | 152 | 6 (Flat) | Active Opponent Exploiter & Shaping | Learned archetype exploitation; targeted bluffs and value bets |
| **Model E** | 158 | 6 (Decoupled) | Self-Play League & Decoupled Action Head | Resolved bet-sizing collapse; learned robust multi-agent self-play balance |
| **Model F** | 168 | 8 (Decoupled) | 8-Action Space, Uncapped Blinds, High LR | Added 33% micro-bets and 150% overbets; eliminated tournament stalling |
| **Model G** | **168** | **8 (Decoupled)** | **Linear Chip-EV, 100 BB Heads-Up Self-Play** | **Dedicated 1v1 specialist; eliminates endgame over-folding and maximizes title capture** |

---

## 5. Multi-Game Generalization: 5-Card Draw & 2-7 Triple Draw

To establish that the CardShark-RL architecture is genuinely generalizable across imperfect-information card games, the repository implements complete game engines and Gym environments for two draw poker variants alongside Texas Hold'em:

### 5.1 5-Card Draw
5-Card Draw represents a contrasting game-theoretic dynamic compared to Texas Hold'em: there are no public community cards on the table. Each player receives five private cards, engages in an opening betting round, selects zero to five cards to discard and replace from the deck, and finishes with a final betting round.

**How It Was Implemented:** In the absence of board cards, information inference shifts entirely to behavioral tells observed during the drawing phase. The environment records the exact number of cards each opponent discards and replaces from the deck. A discard-tell classifier translates these counts into probability distributions over opponent hand categories (for example, discarding 3 cards typically implies an original pair, discarding 1 card implies two-pair or a 4-card flush/straight draw, and drawing zero cards indicates a completed pat hand). These inferred distributions are concatenated into the observation vector, demonstrating that behavioral action tracking can effectively replace community board cards.

### 5.2 2-7 Triple Draw Lowball
2-7 Triple Draw is an inverted-hierarchy poker variant: the lowest traditional poker hand wins the pot (Aces are strictly high, while straights and flushes penalize the player; the nut hand is 2-3-4-5-7 offsuit). The game features three successive draw rounds and four betting rounds.

**How It Was Implemented & Architecture Transfer:** Because 2-7 Triple Draw was developed after the 5-Card Draw framework had matured, its agent (Model 27-A) was directly synthesized on top of the mastered Model F architecture from 5-Card Draw. Rather than starting from an unguided baseline, Model 27-A inherited the multi-head card attention feature extractor, the two-stage decoupled action policy, and dominated discard masking. The environment integrates a specialized lowball hand evaluator alongside a multi-round game loop tracking draw actions across all three discard phases. By transferring the architectural innovations of Model F to an inverted-hierarchy ruleset, the agent achieved immediate, overwhelming strategic dominance, verifying that our feature representations and policy designs generalize across disparate game mechanics with minimal adaptation.

---

## 6. Academic Paper Focus & Planned Ablation Studies

For our paper, we plan to center the core narrative on Texas Hold'em, using 5-Card Draw and 2-7 Triple Draw as cross-game generalization case studies. To ensure that our study makes a substantive scientific contribution beyond simply reporting that a neural network learned to play cards, we have designed six targeted ablation experiments that isolate and quantify the specific contribution of each architectural component:

### 6.1 Ablation 1: The Showdown Avoidance Curve
- **Hypothesis:** Encouraging an agent to win pots uncontested reduces exposure to showdown variance, but excessive bonuses induce destructive over-bluffing.
- **Methodology:** We will train three identical model instances under varying uncontested pot bonus coefficients ($0.0$, $0.015$, and $0.045$). We will evaluate the resulting policies across bluffing frequencies, fold-to-raise rates, and chip accumulation stability. The expected result is a characteristic U-shaped efficiency curve, illustrating the empirical tipping point between passive check-folding and reckless bluffing.

### 6.2 Ablation 2: Action Space Granularity (6 vs. 8 Actions)
- **Hypothesis:** Fine-grained sizing options (specifically 33% micro-cbets and 150% overbets) enable the agent to protect equities and extract maximum value on polarized boards without risking unnecessary chips.
- **Methodology:** We will train two parallel models: one restricted to the standard 6-action abstraction (Min-Raise, Half-Pot, Full-Pot, Shove), and one equipped with the full 8-action space. By analyzing bet-sizing distributions across flop and river board textures, we will quantify how granular action abstractions improve pot control and value extraction.

### 6.3 Ablation 3: Blind Escalation Structure & Tournament Completion
- **Hypothesis:** Capped blind escalation creates artificial tournament deadlocks that reduce late-stage play to arbitrary timeout coin-flips, whereas uncapped escalation forces sound short-stack decision-making.
- **Methodology:** We will simulate tournaments under capped blind structures (freezing at Level 4, $16/$32) versus uncapped structures (escalating through Level 7, $128/$256). We will measure average hand duration, chip stack entropy, and timeout frequencies to demonstrate how blind schedules govern the emergence of short-stack survival strategies.

### 6.4 Ablation 4: Decoupled Hierarchical Policy vs. Single-Stage Flat Softmax
- **Hypothesis:** Separating strategic intent from bet sizing eliminates bet-sizing collapse and preserves healthy exploration across intermediate bet magnitudes.
- **Methodology:** We will train an agent using our two-stage decoupled action policy against a control agent utilizing a standard flat 8-output softmax policy. We will track policy entropy and bet-sizing action selection frequencies across training timesteps, demonstrating that the flat policy collapses to boundary actions (fold or min-raise) while the decoupled architecture maintains diverse sizing utilization.

### 6.5 Ablation 5: Self-Play League Sparring vs. Static Bot Training
- **Hypothesis:** Training in a Prioritized Fictitious Self-Play league pool produces policies that generalize to diverse and unseen opponent styles, whereas training strictly against fixed heuristic bots results in severe style overfitting.
- **Methodology:** We will train one agent exclusively against static heuristic bots and a second agent inside our self-play league pool. Both models will then be evaluated against novel, previously unencountered opponent archetypes (such as erratic random callers, extreme hyper-maniacs, and unbluffable stations). We will measure behavioral exploitability to verify that self-play sparring confers genuine strategic robustness.

### 6.6 Ablation 6: Composite Regime-Switching vs. Monolithic Policies
- **Hypothesis:** Monolithic policies trained exclusively under multi-player tournament incentives suffer from an objective mismatch: they learn risk-averse chip preservation that causes them to play overly passively once the field narrows to heads-up play. Conversely, a modular Composite Model pairing a tournament specialist (Model F) with a dedicated heads-up specialist (Model G) will significantly elevate heads-up conversion and overall championship titles.
- **Methodology:** We will evaluate the Tournament Specialist alone (Model F), the Heads-Up Specialist alone (Model G), and the Composite Model (Model F + Model G) across both dedicated 1v1 duels and 5-player Sit-and-Go tournament benchmarks. We will measure button open frequencies, showdown avoidance, and heads-up conversion rates to quantify how modular regime-switching overcomes the limitations of monolithic deep RL.
