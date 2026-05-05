"""
visualizer.py — Visualisation utilities for CardShark-RL.

Handles plotting learning curves, behavioral heatmaps, profit trajectories,
and neural network architecture diagrams. All plots use a white background.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

# ---------------------------------------------------------------------------
# Global Style & Colors
# ---------------------------------------------------------------------------

COLORS = {
    "model_a":  "#6C5CE7",   # Purple
    "model_b":  "#00B894",   # Teal
    "random":   "#636E72",   # Grey
    "station":  "#E17055",   # Coral
    "maniac":   "#FDCB6E",   # Gold
    "rock":     "#74B9FF",   # Sky blue
}

OPP_NAMES = ["CallingStation", "Maniac", "Rock"]
OPP_SHORT = ["Station", "Maniac", "Rock"]

def set_style():
    """Set the matplotlib style to white background and light theme."""
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "#2D3436",
        "axes.labelcolor":   "#2D3436",
        "text.color":        "#2D3436",
        "xtick.color":       "#2D3436",
        "ytick.color":       "#2D3436",
        "grid.color":        "#DFE6E9",
        "legend.facecolor":  "white",
        "legend.edgecolor":  "#2D3436",
        "font.family":       "sans-serif",
        "font.size":         11,
    })

# ---------------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------------

def plot_bb_comparison(results: dict, save_path: str = "results/bb_per_100_comparison.png"):
    """Bar chart comparing BB/100 across models and opponents."""
    set_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(OPP_SHORT))
    width = 0.25

    bb_a = [results["model_a"][n]["bb_per_100"] for n in OPP_NAMES]
    bb_b = [results["model_b"][n]["bb_per_100"] for n in OPP_NAMES]
    bb_r = [results["random"][n]["bb_per_100"] for n in OPP_NAMES]

    bars_a = ax.bar(x - width, bb_a, width, label="Model A (Explicit)",
                    color=COLORS["model_a"], edgecolor="#2D3436", linewidth=0.5, alpha=0.9)
    bars_b = ax.bar(x, bb_b, width, label="Model B (Implicit)",
                    color=COLORS["model_b"], edgecolor="#2D3436", linewidth=0.5, alpha=0.9)
    bars_r = ax.bar(x + width, bb_r, width, label="Random Baseline",
                    color=COLORS["random"], edgecolor="#2D3436", linewidth=0.5, alpha=0.9)

    # Value labels
    for bars in [bars_a, bars_b, bars_r]:
        for bar in bars:
            h = bar.get_height()
            va = "bottom" if h >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:+.1f}", ha="center", va=va, fontsize=9, fontweight="bold")

    ax.set_xlabel("Opponent Archetype", fontsize=13)
    ax.set_ylabel("BB / 100 Hands", fontsize=13)
    ax.set_title("Cross-Evaluation: Profitability by Model & Opponent",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(OPP_SHORT, fontsize=12)
    ax.legend(fontsize=11, loc="upper right")
    ax.axhline(y=0, color="#2D3436", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_learning_curves(
    callback_a,
    callback_b,
    window: int = 200,
    save_path: str = "results/learning_curves.png",
):
    """Plot smoothed reward curves for both models during training."""
    set_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    for label, cb, color in [
        ("Model A (Explicit)", callback_a, COLORS["model_a"]),
        ("Model B (Implicit)", callback_b, COLORS["model_b"]),
    ]:
        if not cb or not hasattr(cb, 'episode_rewards'):
            continue
            
        rewards = cb.episode_rewards
        if len(rewards) < window:
            window_eff = max(1, len(rewards) // 4)
        else:
            window_eff = window

        if len(rewards) > 0:
            # Smoothed rolling average
            smoothed = np.convolve(rewards, np.ones(window_eff) / window_eff, mode="valid")
            x_axis = np.arange(len(smoothed))
            ax.plot(x_axis, smoothed, label=label, color=color, linewidth=2, alpha=0.9)

            # Light raw data
            ax.plot(np.arange(len(rewards)), rewards,
                    color=color, alpha=0.15, linewidth=0.5)

    ax.set_xlabel("Episode", fontsize=13)
    ax.set_ylabel("Reward (smoothed)", fontsize=13)
    ax.set_title("Training Learning Curves - Model A vs Model B",
                 fontsize=15, fontweight="bold", pad=15)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="#2D3436", linestyle="--", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_behavioral_heatmap(
    matrix: dict,
    save_path: str = "results/behavioral_matrix.png",
):
    """Heatmap of Model B's post-draw action distribution by opponent draw count."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    draw_counts_display = [0, 1, 3]  # Most interesting draw counts
    action_names = ["Fold", "Call", "Raise"]
    cmap = plt.cm.YlOrRd

    for ax_idx, opp_name in enumerate(OPP_NAMES):
        ax = axes[ax_idx]
        data = matrix.get(opp_name, {})

        # Build heatmap data: rows = draw counts, cols = actions
        heat_data = np.zeros((len(draw_counts_display), 3))
        row_labels = []

        for row, dc in enumerate(draw_counts_display):
            row_labels.append(f"Draw {dc}")
            if dc in data:
                heat_data[row, 0] = data[dc]["fold"]
                heat_data[row, 1] = data[dc]["call"]
                heat_data[row, 2] = data[dc]["raise"]

        im = ax.imshow(heat_data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

        # Annotations
        for i in range(len(draw_counts_display)):
            for j in range(3):
                val = heat_data[i, j]
                text_color = "white" if val > 0.5 else "#2D3436"
                ax.text(j, i, f"{val*100:.0f}%",
                        ha="center", va="center", fontsize=13,
                        fontweight="bold", color=text_color)

        ax.set_xticks(range(3))
        ax.set_xticklabels(action_names, fontsize=11)
        ax.set_yticks(range(len(draw_counts_display)))
        ax.set_yticklabels(row_labels, fontsize=11)
        ax.set_title(f"vs {opp_name}", fontsize=13, fontweight="bold", pad=10)

    fig.suptitle(
        "\"The Tell\" - Model B Post-Draw Action Frequencies\n"
        "by Opponent Draw Count",
        fontsize=15, fontweight="bold", y=1.02,
    )

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Action Frequency", fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_cumulative_profit(
    results: dict,
    save_path: str = "results/cumulative_profit.png",
):
    """Plot cumulative profit curves for Model A and Model B against each opponent."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax_idx, opp_name in enumerate(OPP_NAMES):
        ax = axes[ax_idx]

        for model_key, label, color in [
            ("model_a", "Model A (Explicit)", COLORS["model_a"]),
            ("model_b", "Model B (Implicit)", COLORS["model_b"]),
            ("random", "Random", COLORS["random"]),
        ]:
            if opp_name not in results[model_key]:
                continue
                
            profits = results[model_key][opp_name]["hand_profits"]
            cum = np.cumsum(profits)
            ax.plot(cum, label=label, color=color, linewidth=1.5, alpha=0.9)

        ax.set_xlabel("Hand #", fontsize=11)
        ax.set_ylabel("Cumulative Profit", fontsize=11)
        ax.set_title(f"vs {opp_name}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="best")
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color="#2D3436", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.suptitle("Cumulative Profit Trajectories", fontsize=15, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_architecture(save_path: str = "results/architecture.png"):
    """Generate a high-level diagram of the Model B (Implicit) neural network architecture."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")

    # -----------------------------------------------------------------------
    # Helper to draw boxes
    # -----------------------------------------------------------------------
    def draw_box(x, y, w, h, text, color, alpha=0.8):
        rect = patches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="#2D3436", alpha=alpha, linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=11, fontweight="bold")

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#2D3436"))

    # -----------------------------------------------------------------------
    # Layers
    # -----------------------------------------------------------------------
    
    # Input groups
    y_in = 0.9
    draw_box(0.3, y_in, 0.35, 0.08, "Private Hand Info\n(Cards, Category, Score)", "#A8E6CF")
    draw_box(0.7, y_in, 0.35, 0.08, "Betting & Game State\n(Pot, Odds, Phase, Position)", "#DCEDC1")
    
    y_implicit = 0.78
    draw_box(0.5, y_implicit, 0.4, 0.08, "Implicit Opponent Modeling\n(Rolling Behavioral Stats)", "#FFD3B6")
    
    # Concatenation layer
    y_concat = 0.68
    draw_box(0.5, y_concat, 0.8, 0.06, "Observation Vector (N=33)", "#FFAAA5")
    
    draw_arrow(0.3, y_in-0.04, 0.5, y_concat+0.03)
    draw_arrow(0.7, y_in-0.04, 0.5, y_concat+0.03)
    draw_arrow(0.5, y_implicit-0.04, 0.5, y_concat+0.03)

    # Shared Hidden Layers
    y_h1 = 0.55
    y_h2 = 0.45
    y_h3 = 0.35
    draw_box(0.5, y_h1, 0.6, 0.06, "Shared MLP Layer 1 (256 ReLU)", "#D1D8E0")
    draw_box(0.5, y_h2, 0.6, 0.06, "Shared MLP Layer 2 (256 ReLU)", "#D1D8E0")
    draw_box(0.5, y_h3, 0.6, 0.06, "Shared MLP Layer 3 (256 ReLU)", "#D1D8E0")
    
    draw_arrow(0.5, y_concat-0.03, 0.5, y_h1+0.03)
    draw_arrow(0.5, y_h1-0.03, 0.5, y_h2+0.03)
    draw_arrow(0.5, y_h2-0.03, 0.5, y_h3+0.03)

    # Output Heads
    y_out = 0.22
    draw_box(0.3, y_out, 0.35, 0.08, "Policy Head\n(Action Masked Logits)", "#A29BFE")
    draw_box(0.7, y_out, 0.35, 0.08, "Value Head\n(Expected Return)", "#FAB1A0")
    
    draw_arrow(0.5, y_h3-0.03, 0.3, y_out+0.04)
    draw_arrow(0.5, y_h3-0.03, 0.7, y_out+0.04)
    
    # Action Masking
    y_mask = 0.1
    draw_box(0.3, y_mask, 0.35, 0.06, "Masking: Legal Actions Only", "#DFE6E9")
    draw_arrow(0.3, y_out-0.04, 0.3, y_mask+0.03)

    ax.set_title("Model B Architecture: Implicit Opponent Modeling",
                 fontsize=16, fontweight="bold", pad=20)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)
