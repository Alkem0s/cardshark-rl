"""
Generate a professional DOCX report for CardShark-RL.
Explains system architecture, implementation details of each method,
the model progression, multi-game engines, and academic paper roadmap.
Focuses on clear engineering explanations without file path clutter or arbitrary stats.
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def set_cell_background(cell, fill_hex):
    """Set background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    """Set inner padding for a table cell in dxa (1 pt = 20 dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_table_borders(table, color="D0D7DE", sz="4", val="single"):
    """Set clean subtle horizontal borders for a table."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def build_docx_report(output_path):
    doc = Document()
    
    # 1-inch margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
        # Header & Footer
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hr = hp.add_run("CardShark-RL Technical Documentation | CENG 454")
        hr.font.name = "Calibri"
        hr.font.size = Pt(8.5)
        hr.font.color.rgb = RGBColor(0x71, 0x80, 0x96)
        
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fr = fp.add_run("Department of Computer Engineering - İzmir Kâtip Çelebi University")
        fr.font.name = "Calibri"
        fr.font.size = Pt(8.5)
        fr.font.color.rgb = RGBColor(0x71, 0x80, 0x96)

    # Color Palette
    COLOR_PRIMARY = RGBColor(0x1B, 0x36, 0x5D)    # Deep Navy
    COLOR_SECONDARY = RGBColor(0x2B, 0x6C, 0xB0)  # Slate Blue
    COLOR_TEXT = RGBColor(0x2D, 0x37, 0x48)       # Charcoal Body
    COLOR_MUTED = RGBColor(0x71, 0x80, 0x96)      # Gray Metadata

    def add_title(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(22)
        r.font.bold = True
        r.font.color.rgb = COLOR_PRIMARY
        return p

    def add_subtitle(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(12)
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(12.5)
        r.font.color.rgb = COLOR_SECONDARY
        return p

    def add_meta(lines):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(16)
        p.paragraph_format.line_spacing = 1.15
        for i, line in enumerate(lines):
            r = p.add_run(line + ("\n" if i < len(lines) - 1 else ""))
            r.font.name = "Calibri"
            r.font.size = Pt(9.5)
            r.font.color.rgb = COLOR_MUTED
        return p

    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(14)
        r.font.bold = True
        r.font.color.rgb = COLOR_PRIMARY
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(11)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(12)
        r.font.bold = True
        r.font.color.rgb = COLOR_SECONDARY
        return p

    def add_body(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(10.5)
        r.font.color.rgb = COLOR_TEXT
        return p

    def add_bullet(bold_prefix, text):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            rb = p.add_run(bold_prefix)
            rb.font.name = "Calibri"
            rb.font.size = Pt(10.5)
            rb.font.bold = True
            rb.font.color.rgb = COLOR_PRIMARY
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(10.5)
        r.font.color.rgb = COLOR_TEXT
        return p

    # --- Title & Metadata ---
    add_title("CardShark-RL: Autonomous Reinforcement Learning for Card Games")
    add_subtitle("System Architecture, Methodology Implementation, and Academic Research Plan")
    add_meta([
        "Course: CENG 454: Reinforcement Learning (Term Project)",
        "Department of Computer Engineering, İzmir Kâtip Çelebi University",
        "Project Team: Alkım Gönenç Efe, Sarp Sünbül, Damla Parlakyıldız",
        "Date: October 2026"
    ])

    # --- Section 1: Overview & Motivation ---
    add_h1("1. Project Overview & Research Motivation")
    add_body(
        "CardShark-RL is an autonomous reinforcement learning framework engineered to solve multi-player, imperfect-information card games. "
        "The project was initiated as our term project for CENG 454 (Reinforcement Learning) with the primary goal of investigating whether "
        "model-free deep reinforcement learning algorithms can learn sound, adaptive tournament strategies on standard commodity hardware, "
        "without relying on heavy real-time tree solvers or massive computing clusters."
    )
    add_body(
        "Card games represent a classic imperfect-information challenge in artificial intelligence. Unlike board games such as chess or Go, "
        "where both players have full visibility over the board state, a poker player only observes their private hand and the sequence of public actions. "
        "Furthermore, games are characterized by high statistical variance across deck deals, non-stationary multi-agent dynamics (where players adapt "
        "to one another), and shifting risk profiles as tournament blinds escalate. A successful agent cannot simply memorize hand strengths; "
        "it must learn to infer hidden information from betting patterns, exploit predictable opponent archetypes, and manage chip survival under tournament pressure."
    )
    add_body(
        "This document serves two primary functions: first, as an accessible technical reference bringing all team members and co-authors up to speed "
        "on the system's architecture and design decisions; and second, as a structured methodology overview for our faculty advisor and the foundation for our upcoming paper."
    )

    # --- Section 2: What Sets CardShark-RL Apart ---
    add_h1("2. What Sets This Study Apart From Existing Poker AI")
    add_body(
        "Extensive academic literature exists on card game artificial intelligence, most notably landmark systems such as Libratus, DeepStack, and Pluribus. "
        "While these systems achieved milestone results against elite human professionals, CardShark-RL is situated in a distinct practical and scientific space:"
    )
    add_bullet(
        "Lightweight Model-Free Deep RL vs. Heavy Real-Time Tree Search: ",
        "Systems like Pluribus and Libratus rely on Counterfactual Regret Minimization (CFR) paired with real-time subgame solving. During live play, "
        "those engines pause for several seconds per decision to compute fine-grained equilibrium approximations across deeply branched subgame trees. "
        "This demands massive distributed memory (hundreds of gigabytes of RAM) and multi-GPU server infrastructure. In contrast, CardShark-RL relies on pure "
        "model-free Deep Reinforcement Learning (Proximal Policy Optimization). Once trained, our neural network evaluates state vectors and selects actions "
        "in approximately one millisecond on a standard laptop CPU, demonstrating that competitive card play can be achieved in a lightweight, reactive footprint."
    )
    add_bullet(
        "Multi-Player Tournament Play vs. Heads-Up Cash Games: ",
        "The vast majority of academic card AI research focuses exclusively on two-player (heads-up) cash games with static chip stacks, where chip accumulation "
        "is linearly proportional to reward. CardShark-RL addresses full 5-player tournament play. Tournaments introduce non-linear survival incentives: "
        "losing one's remaining chips results in permanent elimination, while periodically escalating blinds force players to continuously adapt their strategies "
        "from deep-stack post-flop maneuvering to short-stack push-or-fold survival."
    )
    add_bullet(
        "Architectural Solutions to Reinforcement Learning Failure Modes: ",
        "Rather than mitigating deep RL instability through external solvers, we address classic failure modes (such as bet-sizing collapse and extreme over-folding) "
        "directly through architectural innovations: order-independent card attention, two-stage decoupled action policies, and self-play league sparring pools."
    )
    add_bullet(
        "Extensible Multi-Game Engine: ",
        "CardShark-RL is not an isolated Texas Hold'em bot. The repository implements a unified environment framework that supports both community card games "
        "(Texas Hold'em) and draw poker variants (5-Card Draw and 2-7 Triple Draw), highlighting the generalizability of our feature extraction and training pipeline."
    )

    # --- Section 3: Core Methodology & Implementation ---
    add_h1("3. Core Methodology & Implementation Details")
    add_body(
        "The CardShark-RL framework addresses the game-theoretic and computational challenges of card play through modular, specialized mechanisms. "
        "Below is an overview of how each methodology was designed and implemented:"
    )

    add_h2("3.1 Order-Independent Hand Processing (Card Attention Feature Extractor)")
    add_body(
        "The Problem: In card games, the physical order in which cards are dealt has no bearing on hand strength: holding Ace-King is mathematically identical to King-Ace. "
        "Standard multi-layer perceptrons treat card inputs as a fixed flat vector, which artificially forces the network to learn duplicate representations "
        "for every possible permutation of the same hand."
    )
    add_body(
        "How It Was Implemented: Each card in the player's private hand and on the public board is treated as an individual entity token. "
        "The card's integer representation is mapped through two parallel embedding layers—one encoding rank (Two through Ace) and one encoding suit (Clubs, Diamonds, Hearts, Spades)—which "
        "are projected into a 64-dimensional dense vector. These card tokens are fed through a 4-head self-attention layer. The self-attention mechanism computes pairwise "
        "affinity weights between all cards simultaneously. This allows the network to naturally detect poker patterns—such as paired ranks, suited connectors, or straight draws—regardless "
        "of the slot each card occupied. The attended card vectors are averaged via mean-pooling into a single hand summary representation and concatenated with the table state vector."
    )

    add_h2("3.2 Separating Intent from Bet Magnitude (Decoupled Hierarchical Action Policy)")
    add_body(
        "The Problem: In No-Limit poker, players can choose from many different bet sizes. When a standard policy outputs a single flat list of actions containing both basic moves "
        "(Fold, Call) and all possible bet sizes (Half-Pot, Full-Pot, Shove), exploratory large bets early in training frequently lose chips on weak hands. "
        "The network receives strong negative feedback on those action indices and permanently suppresses their probabilities. The policy collapses into passive check-folding or repetitive minimum-raising."
    )
    add_body(
        "How It Was Implemented: We restructured the policy network by splitting the output layers into two separate branches sharing a common latent torso. "
        "The first branch (the Core Intent head) outputs probability distributions over high-level strategic actions: Fold, Check/Call, Raise, and All-In. "
        "The second branch (the Sizing head) outputs probability distributions over fractional pot sizes (33%, 50%, 75%, 100%, and 150% of the pot). "
        "During action execution, if the core head chooses Fold, Call, or All-In, the action executes immediately without querying the sizing head. "
        "Only when the core head selects Raise is the sizing head evaluated to determine the exact bet magnitude. During backpropagation, sizing gradients are masked so that sizing parameters "
        "are only updated on steps where a raise occurred. This decouples the exploration of bet sizing from the decision to bet, eliminating bet-sizing collapse."
    )

    add_h2("3.3 Opponent Profiling and Active Exploitation")
    add_body(
        "The Problem: Fixed baseline strategies leave substantial value on the table when playing against flawed opponents. An effective agent must recognize opponent habits—such as "
        "an opponent who folds too often to aggression, or a passive player who refuses to fold weak pairs—and dynamically alter its play to exploit those weaknesses."
    )
    add_body(
        "How It Was Implemented: The environment maintains running historical statistics for every seat at the table across consecutive hands. It tracks metrics including "
        "Voluntarily Put Money in Pot (VPIP, measuring loose vs. tight pre-flop play), Pre-Flop Raise frequency (PFR, measuring pre-flop aggression), and Fold-to-Bet frequencies across each street. "
        "To handle the low sample sizes inherent in short tournament sessions, Bayesian updates smooth observed counts toward standard prior distributions. "
        "The resulting statistical profiles are categorized into opponent archetypes (such as Tight-Aggressive, Loose-Aggressive, or Calling Station) and directly concatenated into the observation vector. "
        "Furthermore, the reward function includes calibrated shaping bonuses that reward the agent for executing mathematically sound exploitative lines, such as bluffing against players with high fold frequencies "
        "or extracting value bets against calling stations without bluffing."
    )

    add_h2("3.4 Self-Play League Sparring (Prioritized Fictitious Self-Play)")
    add_body(
        "The Problem: Training an agent exclusively against static, rule-based bots causes severe overfitting: the agent learns to exploit the hardcoded programming quirks of those bots, "
        "but completely falls apart when exposed to novel styles or aggressive neural counter-strategies."
    )
    add_body(
        "How It Was Implemented: We implemented an autonomous self-play league pool. During training, the framework periodically saves frozen snapshots of the learning policy. "
        "When generating training rollout tables, seats are dynamically populated using a prioritized mixture: fixed heuristic bots are retained to maintain basic poker fundamentals, "
        "while historical neural checkpoints are sampled from the pool based on competitive difficulty. Past checkpoints that present a high challenge to the current policy are sampled more frequently. "
        "Sparring against diverse past iterations of itself prevents the policy from settling into predictable, easily exploitable habits."
    )

    add_h2("3.5 Tournament Chip Valuation (Independent Chip Model / ICM Equity)")
    add_body(
        "The Problem: In cash games, chips have constant linear value. In tournament poker, however, chips lost are worth more than chips gained: losing one's last chip knocks a player out of the tournament, "
        "whereas doubling one's chip stack does not double the probability of winning first place. Agents that evaluate chip counts linearly take reckless gambles on marginal hands and bust out prematurely."
    )
    add_body(
        "How It Was Implemented: We integrated tournament equity calculations based on the Independent Chip Model (ICM). The environment translates raw chip stack distributions into actual tournament prize equities. "
        "The reward signal provided to the agent is scaled by changes in tournament equity rather than simple raw chip differences. This penalizes reckless stack erosion when playing with a short stack "
        "and naturally encourages disciplined survival strategies near elimination boundaries."
    )

    add_h2("3.6 Auxiliary Board Reading Supervision (Auxiliary Belief Head)")
    add_body(
        "The Problem: Reinforcement learning agents often require millions of exploratory hands simply to understand how community card textures relate to hand categories "
        "(such as recognizing when a board is dangerous due to completed flushes or straights)."
    )
    add_body(
        "How It Was Implemented: We attached an auxiliary classification head branching directly from the shared torso representation. "
        "During training, this auxiliary head is trained under supervised loss to predict board texture danger tiers (e.g., paired boards, monotone boards) and opponent hand strength categories. "
        "The resulting supervised auxiliary loss gradients accelerate representation learning in early layers, helping the network learn meaningful card representations significantly faster than relying on policy return gradients alone."
    )

    add_h2("3.7 Dynamic Blind Escalation & Uncontested Pot Incentives")
    add_body(
        "The Problem: In tournament simulations, if blind escalation freezes at a low level, deep chip stacks pass chips back and forth indefinitely, causing games to hit artificial turn limits "
        "and turn into arbitrary tie-breaks. Furthermore, agents trained purely on showdown chip results frequently become timid, checking down strong hands to avoid risk and suffering unnecessary luck variance."
    )
    add_body(
        "How It Was Implemented: The table engine implements an uncapped blind schedule that doubles the blinds after fixed hand intervals, escalating through multiple blind tiers. "
        "This dynamically compresses stack depths as the tournament progresses, forcing the policy to smoothly transition from deep-stack play to short-stack push-or-fold endgame dynamics without hitting timeouts. "
        "Additionally, we introduced an uncontested pot bonus that awards a small incentive whenever the agent wins a pot by forcing all active opponents to fold before showdown. "
        "This teaches the agent to apply healthy pressure and accumulate uncontested chips when opponents display weakness."
    )

    add_h2("3.8 Two-Phase Curriculum & Reward Annealing")
    add_body(
        "The Problem: Heuristic reward shaping guides early exploration, but if left active indefinitely, the agent over-optimizes for intermediate bonuses "
        "(such as folding rewards or exploratory bluffing bonuses) rather than overall tournament success."
    )
    add_body(
        "How It Was Implemented: We implemented a scheduled curriculum callback that manages reward shaping weights across two distinct phases. "
        "In Phase 1 (Scaffolding), auxiliary shaping rewards are fully active to accelerate policy stabilization and prevent blunders. "
        "In Phase 2 (Competition), a decay schedule systematically scales down these shaping multipliers toward zero over several hundred thousand steps. "
        "By the final stages of training, the agent optimizes purely for raw tournament outcomes and chip accumulation, eliminating the risk of reward hacking."
    )

    add_h2("3.9 Observation Vector Anatomy (Model F Feature Decomposition)")
    add_body(
        "To provide complete transparency into the agent's sensory input, the 168-dimensional observation vector for Model F is structured into six functional tiers:"
    )

    obs_table = doc.add_table(rows=8, cols=3)
    obs_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    obs_table.autofit = False
    set_table_borders(obs_table)

    obs_headers = ["Tier", "Dimensions", "Feature Components"]
    obs_col_widths = [Inches(1.8), Inches(1.1), Inches(3.6)]

    hdr_cells = obs_table.rows[0].cells
    for i, title in enumerate(obs_headers):
        hdr_cells[i].width = obs_col_widths[i]
        set_cell_background(hdr_cells[i], "1B365D")
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=100, right=100)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i == 1 else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(title)
        r.font.name = "Calibri"
        r.font.size = Pt(9)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    obs_data = [
        ("1. Core Table & Card State", "73", "Private hole cards (4), community cards (10), evaluated hand strength & Chen score (1), pot odds, pot size, hero chips, and bet-to-call ratios (4), street indicators (4 one-hot), position relative to button (5 one-hot), per-seat chip and status tracking across 5 seats (25), and street action history (20)."),
        ("2. Opponent Profiling & Textures", "52", "ICM stack geometry and shortest stack ratios (3), board texture engine evaluating pairs, monotone boards, and straight/flush draws (17), and dual-timescale Bayesian opponent tracking tracking VPIP, PFR, fold-to-cbet, and aggression across fast and slow windows (32)."),
        ("3. Tournament Dynamics & GTO", "13", "Pre-flop GTO positional guidance (6), post-flop stack-to-pot ratio (SPR), showdown value classification, in-position advantage flag, range aggressor indicator (4), and non-linear ICM tournament equity valuations (3)."),
        ("4. Targeted Opponent Exploitation", "14", "Real-time exploitability vectors evaluating opponent folding vulnerability, calling station value extraction opportunities, steal feasibility, and dynamic opponent archetype classification."),
        ("5. Self-Play & Multi-Agent", "6", "Pre-flop 3-bet squeeze intensity, check-raise volatility index, effective stack depth in big blinds, push/fold regime trigger, multiway pot indicator, and neural league sparring opponent flag."),
        ("6. Tournament Closer & Endgame", "10", "Hands remaining in blind level, blind escalation urgency, hero chip rank, Harrington M-factor, Harrington M-Zone, heads-up regime flag, podium/bubble pressure flag, Sklansky-Chubukov push urgency, fold-to-cbet steal indicator, and chip dominance ratio."),
        ("Total", "168", "Complete state vector fed to Card Attention and policy networks."),
    ]

    for row_idx, row_data in enumerate(obs_data, start=1):
        row_cells = obs_table.rows[row_idx].cells
        is_total = (row_idx == len(obs_data))
        bg = "E2E8F0" if is_total else ("F8FAFC" if row_idx % 2 == 1 else "FFFFFF")
        for col_idx, text in enumerate(row_data):
            row_cells[col_idx].width = obs_col_widths[col_idx]
            set_cell_background(row_cells[col_idx], bg)
            set_cell_margins(row_cells[col_idx], top=80, bottom=80, left=100, right=100)
            p = row_cells[col_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx == 1 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(text)
            r.font.name = "Calibri"
            r.font.size = Pt(8.5)
            if is_total or col_idx == 0:
                r.font.bold = True
            r.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    add_h2("3.10 Tournament Structure & Benchmark Evaluation Protocol")
    add_body(
        "To evaluate agents in an ecologically valid competitive setting rather than synthetic isolated hands, the framework implements an end-to-end Sit-and-Go tournament simulator:"
    )
    add_bullet(
        "Tournament Mechanics: ",
        "5-player Sit-and-Go tournament with winner-take-all and podium incentives. Each player starts with 1,000 chips (5,000 total chips). "
        "Blinds start at 5/10 and double every 20 hands across seven uncapped tiers (10/20 up to 640/1280) to force natural endgame transitions without timeouts."
    )
    add_bullet(
        "Benchmark Opponent Roster: ",
        "Evaluations pit the agent against a balanced field consisting of TAGBot (tight-aggressive, selective ranges, high post-flop pressure), "
        "LAGBot (loose-aggressive, wide ranges, frequent float bets), CallingStationBot (passive calling station, rarely folds), and RandomHoldemBot (stochastic noise)."
    )
    add_bullet(
        "Evaluation Metrics: ",
        "Performance is measured across 1st-place win rate (% of tournaments won outright), podium finish rate (% top 2 or 3 finishes), average net chips accumulated, "
        "and a composite score: Score = (Wins * 5.0) + (Podiums * 2.0) + (Net Chips * 0.05) - (Eliminations * 1.0)."
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # --- Section 4: Progressive Model Iterations ---
    add_h1("4. Progressive Model Iterations (Texas Hold'em Models A through F)")
    add_body(
        "Rather than treating model development as a single monolithic training run, Texas Hold'em models were evolved across six sequential generations (Model A through Model F). "
        "Each iteration was designed to diagnose and resolve specific behavioral and game-theoretic flaws observed in previous versions:"
    )

    add_bullet(
        "Model A (Baseline Architecture): ",
        "Implemented as the initial proof-of-concept. Used a 73-dimensional state vector and a flat 6-action discrete space, trained exclusively against static rule-based bots. "
        "While it learned basic hand rankings and fundamental rules, it exhibited severe passivity, folded excessively under any aggression, and suffered from bet-sizing collapse."
    )
    add_bullet(
        "Model B (Opponent Profiling): ",
        "Expanded the observation space to 125 dimensions by incorporating Bayesian tracking of opponent VPIP, PFR, and fold tendencies. This enabled the agent to distinguish "
        "between tight and loose opponents, allowing it to open wider against timid players and fold against tight raisers."
    )
    add_bullet(
        "Model C (Tournament Awareness): ",
        "Expanded the observation space to 138 dimensions by integrating tournament survival logic, blind escalation ratios, and stack-depth metrics. The model learned stack "
        "preservation, shifting to conservative push-or-fold tactics when short-stacked rather than bleeding chips through calling."
    )
    add_bullet(
        "Model D (Active Exploitation): ",
        "Expanded state inputs to 152 dimensions with explicit opponent exploitability signals. The agent began actively altering its post-flop lines based on opponent archetypes, "
        "consistently value-betting against calling stations and executing targeted bluffs against frequent folders."
    )
    add_bullet(
        "Model E (Self-Play & Decoupled Policy): ",
        "Expanded inputs to 158 dimensions and introduced two major architectural milestones: the Prioritized Fictitious Self-Play league pool and the two-stage decoupled action policy. "
        "Decoupling strategic intent from sizing eliminated bet-sizing collapse, while self-play sparring eliminated the agent's vulnerability to varied opponent counter-strategies."
    )
    add_bullet(
        "Model F (Fine-Grained Actions, Uncapped Blinds & Boundary Screening): ",
        "Expanded observation inputs to 168 dimensions and the action space to 8 discrete actions (introducing a 33% micro-continuation bet and a 150% overbet). "
        "Uncapped blind escalation up to level 7 was introduced to eliminate tournament stalling, and hyperparameter boundaries were systematically screened across "
        "learning rates, entropy schedules, and reward shaping coefficients to ensure stable policy convergence."
    )

    # Summary Table of Model Lineage (NO WIN RATES, NO FILE PATHS)
    table = doc.add_table(rows=7, cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)

    headers = ["Model", "State Dims", "Action Space", "Key Methodologies Added", "Primary Behavioral Capability / Issue Resolved"]
    col_widths = [Inches(0.9), Inches(0.9), Inches(1.1), Inches(1.8), Inches(1.8)]

    # Header Row
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].width = col_widths[i]
        set_cell_background(hdr_cells[i], "1B365D")
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=100, right=100)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i in [1, 2] else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(title)
        r.font.name = "Calibri"
        r.font.size = Pt(9)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    # Data Rows
    data = [
        ("Model A", "73", "6 (Flat)", "Baseline PPO, Card Attention Extractor", "Learned valid hand rules; suffered from severe passivity and sizing collapse"),
        ("Model B", "125", "6 (Flat)", "Bayesian Opponent Tracking", "Learned opponent profiling; differentiated tight from loose pre-flop play"),
        ("Model C", "138", "6 (Flat)", "ICM Equity & Stack Depth Ratios", "Learned chip survival and stack preservation under escalating blinds"),
        ("Model D", "152", "6 (Flat)", "Active Opponent Exploiter & Shaping", "Learned archetype exploitation; targeted bluffs and value bets"),
        ("Model E", "158", "6 (Decoupled)", "Self-Play League & Decoupled Action Head", "Resolved bet-sizing collapse; learned robust multi-agent self-play balance"),
        ("Model F", "168", "8 (Decoupled)", "8-Action Space, Uncapped Blinds, High LR", "Added 33% micro-bets and 150% overbets; eliminated tournament stalling")
    ]

    for row_idx, row_data in enumerate(data, start=1):
        row_cells = table.rows[row_idx].cells
        bg_color = "F7FAFC" if row_idx % 2 == 0 else "FFFFFF"
        for col_idx, text in enumerate(row_data):
            row_cells[col_idx].width = col_widths[col_idx]
            set_cell_background(row_cells[col_idx], bg_color)
            set_cell_margins(row_cells[col_idx], top=80, bottom=80, left=100, right=100)
            p = row_cells[col_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx in [1, 2] else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(text)
            r.font.name = "Calibri"
            r.font.size = Pt(8.5)
            if row_idx == 6:
                r.font.bold = True
                r.font.color.rgb = COLOR_PRIMARY
            else:
                r.font.color.rgb = COLOR_TEXT

    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(4)
    p_spacer.paragraph_format.space_after = Pt(4)

    # --- Section 5: Multi-Game Generalization ---
    add_h1("5. Multi-Game Generalization: 5-Card Draw & 2-7 Triple Draw")
    add_body(
        "To establish that the CardShark-RL architecture is genuinely generalizable across imperfect-information card games, "
        "the repository implements complete game engines and Gym environments for two draw poker variants alongside Texas Hold'em:"
    )

    add_h2("5.1 5-Card Draw")
    add_body(
        "5-Card Draw represents a contrasting game-theoretic dynamic compared to Texas Hold'em: there are no public community cards on the table. "
        "Each player receives five private cards, engages in an opening betting round, selects zero to five cards to discard and replace from the deck, "
        "and finishes with a final betting round."
    )
    add_body(
        "How It Was Implemented: In the absence of board cards, information inference shifts entirely to behavioral tells observed during the drawing phase. "
        "The environment records the exact number of cards each opponent discards and replaces from the deck. A discard-tell classifier translates these counts "
        "into probability distributions over opponent hand categories (for example, discarding 3 cards typically implies an original pair, discarding 1 card implies "
        "two-pair or a 4-card flush/straight draw, and drawing zero cards indicates a completed pat hand). These inferred distributions are concatenated into the observation vector, "
        "demonstrating that behavioral action tracking can effectively replace community board cards."
    )

    add_h2("5.2 2-7 Triple Draw Lowball")
    add_body(
        "2-7 Triple Draw is an inverted-hierarchy poker variant: the lowest traditional poker hand wins the pot (Aces are strictly high, while straights "
        "and flushes penalize the player; the nut hand is 2-3-4-5-7 offsuit). The game features three successive draw rounds and four betting rounds."
    )
    add_body(
        "How It Was Implemented: The environment implements a specialized lowball hand evaluator alongside a multi-round game loop featuring three sequential draw phases "
        "and four betting rounds. The Gym observation wrapper accommodates multiple draw rounds and negative card value logic without requiring changes to the core "
        "reinforcement learning pipeline, verifying that our state representations and action abstractions are truly modular."
    )

    # --- Section 6: Paper Roadmap & Ablation Studies ---
    add_h1("6. Academic Paper Focus & Planned Ablation Studies")
    add_body(
        "For our term paper, we plan to center the core narrative on Texas Hold'em, using 5-Card Draw and 2-7 Triple Draw as cross-game generalization case studies. "
        "To ensure that our study makes a substantive scientific contribution beyond simply reporting that a neural network learned to play cards, "
        "we have designed five targeted ablation experiments that isolate and quantify the specific contribution of each architectural component:"
    )

    add_h2("6.1 Ablation 1: The Showdown Avoidance Curve")
    add_body(
        "Hypothesis: Encouraging an agent to win pots uncontested reduces exposure to showdown variance, but excessive bonuses induce destructive over-bluffing."
    )
    add_body(
        "Methodology: We will train three identical model instances under varying uncontested pot bonus coefficients (0.0, 0.015, and 0.045). "
        "We will evaluate the resulting policies across bluffing frequencies, fold-to-raise rates, and chip accumulation stability. "
        "The expected result is a characteristic U-shaped efficiency curve, illustrating the empirical tipping point between passive check-folding and reckless bluffing."
    )

    add_h2("6.2 Ablation 2: Action Space Granularity (6 vs. 8 Actions)")
    add_body(
        "Hypothesis: Fine-grained sizing options (specifically 33% micro-cbets and 150% overbets) enable the agent to protect equities and extract maximum value "
        "on polarized boards without risking unnecessary chips."
    )
    add_body(
        "Methodology: We will train two parallel models: one restricted to the standard 6-action abstraction (Min-Raise, Half-Pot, Full-Pot, Shove), and one equipped "
        "with the full 8-action space. By analyzing bet-sizing distributions across flop and river board textures, we will quantify how granular action abstractions "
        "improve pot control and value extraction."
    )

    add_h2("6.3 Ablation 3: Blind Escalation Structure & Tournament Completion")
    add_body(
        "Hypothesis: Capped blind escalation creates artificial tournament deadlocks that reduce late-stage play to arbitrary timeout coin-flips, "
        "whereas uncapped escalation forces sound short-stack decision-making."
    )
    add_body(
        "Methodology: We will simulate tournaments under capped blind structures (freezing at Level 4, $16/$32) versus uncapped structures (escalating through Level 7, $128/$256). "
        "We will measure average hand duration, chip stack entropy, and timeout frequencies to demonstrate how blind schedules govern the emergence of short-stack survival strategies."
    )

    add_h2("6.4 Ablation 4: Decoupled Hierarchical Policy vs. Single-Stage Flat Softmax")
    add_body(
        "Hypothesis: Separating strategic intent from bet sizing eliminates bet-sizing collapse and preserves healthy exploration across intermediate bet magnitudes."
    )
    add_body(
        "Methodology: We will train an agent using our two-stage decoupled action policy against a control agent utilizing a standard flat 8-output softmax policy. "
        "We will track policy entropy and bet-sizing action selection frequencies across training timesteps, demonstrating that the flat policy collapses to boundary actions "
        "(fold or min-raise) while the decoupled architecture maintains diverse sizing utilization."
    )

    add_h2("6.5 Ablation 5: Self-Play League Sparring vs. Static Bot Training")
    add_body(
        "Hypothesis: Training in a Prioritized Fictitious Self-Play league pool produces policies that generalize to diverse and unseen opponent styles, "
        "whereas training strictly against fixed heuristic bots results in severe style overfitting."
    )
    add_body(
        "Methodology: We will train one agent exclusively against static heuristic bots and a second agent inside our self-play league pool. Both models will then be evaluated "
        "against novel, previously unencountered opponent archetypes (such as erratic random callers, extreme hyper-maniacs, and unbluffable stations). "
        "We will measure behavioral exploitability to verify that self-play sparring confers genuine strategic robustness."
    )

    # Save document
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc.save(output_path)
    print(f"Successfully generated DOCX report at: {output_path}")

if __name__ == "__main__":
    out_file = os.path.join(os.getcwd(), "docs", "CardShark_RL_Project_Report.docx")
    build_docx_report(out_file)
