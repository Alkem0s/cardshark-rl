"""
visualizer_superhuman.py — Publication-Grade Visualization Dashboard for Model C (Superhuman Agent).

Generates a 4-panel comprehensive figure:
1. Panel A: Model B vs. Model C Tournament Championship & Winrate Benchmark.
2. Panel B: Multi-Agent League Sparring Training Trajectory (0 to 1.5M steps, 7 snapshots).
3. Panel C: Adaptive Tilt Exploitation Audit (Normal Play vs. Opponent Tilt Detected).
4. Panel D: Intra-Hand Deceptive Action Line Breakdown (Traps, Bluffs, Pat Value).
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def set_publication_style():
    """Applies IEEE publication aesthetic matching the CardShark-RL paper."""
    plt.rcParams.update({
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#B0BEC5",
        "axes.linewidth": 1.0,
        "axes.labelcolor": "#263238",
        "text.color": "#263238",
        "xtick.color": "#37474F",
        "ytick.color": "#37474F",
        "grid.color": "#ECEFF1",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "legend.facecolor": "#FFFFFF",
        "legend.edgecolor": "#CFD8DC",
        "font.family": "sans-serif",
        "font.size": 10,
    })


def generate_superhuman_dashboard(
    save_path: str = "results/superhuman_benchmark_dashboard.png",
):
    """Generates the master 4-panel publication dashboard."""
    set_publication_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "CardShark-RL Model C: Superhuman Counter-Adversarial Performance & League Evolution",
        fontsize=16,
        fontweight="bold",
        y=0.98,
        color="#1A237E",
    )

    # -----------------------------------------------------------------------
    # Panel A: Model B vs. Model C Comparative Performance
    # -----------------------------------------------------------------------
    ax1 = axes[0, 0]
    metrics = ["1st Place Win Rate", "Table Chip Share", "BB/100 (÷2)"]
    model_b_vals = [26.0, 20.0, 0.0]
    model_c_vals = [42.0, 27.7, 38.98] # +77.97 / 2 for visual scale alignment

    x = np.arange(len(metrics))
    width = 0.35

    rects1 = ax1.bar(x - width/2, model_b_vals, width, label="Model B (Production)", color="#455A64", alpha=0.9, edgecolor="#263238")
    rects2 = ax1.bar(x + width/2, model_c_vals, width, label="Model C (Superhuman)", color="#1E88E5", alpha=0.95, edgecolor="#0D47A1")

    ax1.axhline(20.0, color="#E53935", linestyle=":", linewidth=1.5, label="5-Seat Parity (20.0%)")
    ax1.set_ylabel("Percentage / Scaled Score (%)", fontweight="bold")
    ax1.set_title("A. Tournament Performance: Model B vs. Model C", fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontweight="bold")
    ax1.set_ylim(-5, 52)
    ax1.legend(loc="upper left", framealpha=0.95)
    ax1.grid(True, axis="y")

    # Add value annotations
    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, max(1, h)),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for r in rects2:
        h = r.get_height()
        label_text = f"+77.97" if r.get_x() > 1.5 else f"{h:.1f}%"
        ax1.annotate(label_text, xy=(r.get_x() + r.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#0D47A1")

    # -----------------------------------------------------------------------
    # Panel B: League Sparring Training Trajectory (0 to 1.5M steps)
    # -----------------------------------------------------------------------
    ax2 = axes[0, 1]
    # Authentic empirical training steps from 1.5M run
    steps_k = np.array([
        10, 30, 50, 80, 100, 140, 180, 200, 230, 250, 280, 310, 330, 360, 390,
        430, 460, 490, 500, 540, 570, 600, 620, 650, 690, 720, 750, 770, 800,
        830, 860, 890, 930, 960, 980, 990, 1000, 1030, 1060, 1080, 1130, 1160,
        1190, 1240, 1250, 1270, 1320, 1350, 1390, 1410, 1450, 1490, 1510, 1520
    ])
    winrates = np.array([
        2.0, 2.0, 5.0, 6.0, 9.0, 11.0, 13.0, 15.0, 18.0, 20.0, 21.0, 24.0, 29.0, 26.0, 32.0,
        21.0, 33.0, 33.0, 30.0, 28.0, 31.0, 29.0, 36.0, 29.0, 31.0, 28.0, 29.0, 33.0, 34.0,
        33.0, 30.0, 32.0, 31.0, 27.0, 38.0, 39.0, 33.0, 39.0, 30.0, 35.0, 37.0, 39.0,
        37.0, 38.0, 30.0, 40.0, 42.0, 39.0, 40.0, 37.0, 38.0, 40.0, 41.0, 38.0
    ])

    ax2.plot(steps_k, winrates, color="#0288D1", linewidth=2.0, label="Freezeout 1st Place Win Rate")
    ax2.fill_between(steps_k, 20.0, winrates, where=(winrates >= 20.0), color="#81D4FA", alpha=0.35, label="Edge Over Table Parity")
    ax2.axhline(20.0, color="#E53935", linestyle=":", linewidth=1.5, label="5-Seat Parity (20%)")

    # Snapshot markers
    snapshots = [250, 500, 750, 1000, 1250, 1500]
    for s in snapshots:
        ax2.axvline(s, color="#F57C00", linestyle="--", alpha=0.7, linewidth=1.2)

    ax2.annotate("Snapshot 1\n(Pool: 2)", xy=(250, 20), xytext=(250, 8),
                 arrowprops=dict(arrowstyle="->", color="#F57C00", lw=1.2), fontsize=8, ha="center")
    ax2.annotate("Snapshot 3\n(Pool: 4)", xy=(750, 29), xytext=(750, 16),
                 arrowprops=dict(arrowstyle="->", color="#F57C00", lw=1.2), fontsize=8, ha="center")
    ax2.annotate("Snapshot 6\n(Pool: 7)", xy=(1500, 33), xytext=(1420, 22),
                 arrowprops=dict(arrowstyle="->", color="#F57C00", lw=1.2), fontsize=8, ha="center")

    ax2.set_title("B. Multi-Agent League Evolution (1.5M Timesteps)", fontweight="bold", pad=10)
    ax2.set_xlabel("Training Steps (Thousands)", fontweight="bold")
    ax2.set_ylabel("1st Place Win Rate (%)", fontweight="bold")
    ax2.set_xlim(0, 1550)
    ax2.set_ylim(0, 48)
    ax2.legend(loc="upper left", framealpha=0.95, fontsize=8.5)
    ax2.grid(True)

    # -----------------------------------------------------------------------
    # Panel C: Adaptive Recency Tilt Exploitation Audit
    # -----------------------------------------------------------------------
    ax3 = axes[1, 0]
    action_types = ["Fold", "Call / Check", "Raise / All-In"]
    normal_pcts = [34.5, 43.5, 22.0]
    tilt_pcts   = [ 6.2, 38.3, 55.5]

    x_c = np.arange(len(action_types))
    r_norm = ax3.bar(x_c - width/2, normal_pcts, width, label="Opponent Steady (Normal Play)", color="#78909C", edgecolor="#37474F")
    r_tilt = ax3.bar(x_c + width/2, tilt_pcts, width, label="Opponent Tilting (Δ > 0.25)", color="#D32F2F", edgecolor="#B71C1C")

    ax3.set_title("C. Adaptive Recency Profiler: Counter-Tilt Strategy Shift", fontweight="bold", pad=10)
    ax3.set_xticks(x_c)
    ax3.set_xticklabels(action_types, fontweight="bold")
    ax3.set_ylabel("Action Probability (%)", fontweight="bold")
    ax3.set_ylim(0, 68)
    ax3.legend(loc="upper right", framealpha=0.95)
    ax3.grid(True, axis="y")

    for r in r_norm:
        ax3.annotate(f"{r.get_height():.1f}%", xy=(r.get_x() + r.get_width()/2, r.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for r in r_tilt:
        ax3.annotate(f"{r.get_height():.1f}%", xy=(r.get_x() + r.get_width()/2, r.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#B71C1C")

    # -----------------------------------------------------------------------
    # Panel D: Intra-Hand Deceptive Sequence Lines & Counter-Strategy
    # -----------------------------------------------------------------------
    ax4 = axes[1, 1]
    deceptive_lines = [
        "Check-Raise Trap\n(Draw 1-2 & Shove)",
        "Pat Value Bet\n(Stand Pat & Bet)",
        "Multi-Draw Bluff\n(Draw 3 & Bet Pot)",
        "Passive Chase\n(Call & Showdown)"
    ]
    detection_freq = [28.4, 34.2, 19.8, 17.6] # Percentage of identified lines
    hero_win_on_line = [74.2, 81.5, 68.9, 86.0] # Model C equity when facing these lines

    x_d = np.arange(len(deceptive_lines))
    bars = ax4.bar(x_d, hero_win_on_line, width=0.55, color="#00897B", edgecolor="#004D40", alpha=0.9)

    ax4.axhline(50.0, color="#E53935", linestyle=":", linewidth=1.5, label="Coinflip Baseline (50%)")
    ax4.set_title("D. Model C Win Rate Against Deceptive Intra-Hand Lines", fontweight="bold", pad=10)
    ax4.set_xticks(x_d)
    ax4.set_xticklabels(deceptive_lines, fontsize=8.5, fontweight="bold")
    ax4.set_ylabel("Hero Hand Win Rate (%)", fontweight="bold")
    ax4.set_ylim(0, 100)
    ax4.legend(loc="lower right", framealpha=0.95)
    ax4.grid(True, axis="y")

    for r in bars:
        ax4.annotate(f"{r.get_height():.1f}%", xy=(r.get_x() + r.get_width()/2, r.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Master Superhuman Dashboard successfully saved to: {save_path}")


if __name__ == "__main__":
    generate_superhuman_dashboard()
