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
    callbacks_a: list | object | None,
    callbacks_b: list | object | None,
    window: int = 200,
    save_path: str = "results/learning_curves.png",
):
    """Plot smoothed reward curves (with standard deviation shading if multiple seeds exist)."""
    set_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    for label, cb_list, color in [
        ("Model A (Explicit)", callbacks_a, COLORS["model_a"]),
        ("Model B (Implicit)", callbacks_b, COLORS["model_b"]),
    ]:
        if not cb_list:
            continue

        if not isinstance(cb_list, list):
            cb_list = [cb_list]

        all_rewards = []
        all_timesteps = []
        for cb in cb_list:
            if cb and hasattr(cb, 'episode_rewards') and len(cb.episode_rewards) > 0:
                all_rewards.append(np.array(cb.episode_rewards))
                all_timesteps.append(np.array(cb.timestamps))

        if not all_rewards:
            continue

        # Interpolate and smooth rewards on a common grid of timesteps
        max_ts = max(max(t) for t in all_timesteps)
        common_grid = np.linspace(0, max_ts, 1000)

        interpolated_rewards = []
        for rewards, ts in zip(all_rewards, all_timesteps):
            interp_y = np.interp(common_grid, ts, rewards)
            window_eff = min(window, len(interp_y) // 4)
            window_eff = max(1, window_eff)
            smoothed = np.convolve(interp_y, np.ones(window_eff) / window_eff, mode="same")
            interpolated_rewards.append(smoothed)

        interpolated_rewards = np.array(interpolated_rewards)
        mean_rewards = np.mean(interpolated_rewards, axis=0)

        # Plot mean curve
        ax.plot(common_grid, mean_rewards, label=label, color=color, linewidth=2, alpha=0.9)

        # Shaded standard deviation region for multi-seed runs
        if len(cb_list) > 1:
            std_rewards = np.std(interpolated_rewards, axis=0)
            ax.fill_between(
                common_grid,
                mean_rewards - std_rewards,
                mean_rewards + std_rewards,
                color=color,
                alpha=0.15,
                edgecolor="none"
            )
        else:
            # If a single seed, display the raw individual episodic points lightly
            ax.plot(all_timesteps[0], all_rewards[0], color=color, alpha=0.15, linewidth=0.5)

    ax.set_xlabel("Training Steps (Timesteps)", fontsize=13)
    ax.set_ylabel("Episodic Reward (smoothed)", fontsize=13)
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
        fontsize=15, fontweight="bold", y=0.98,
    )

    # Use a dedicated colorbar axis to prevent tight_layout warping
    plt.tight_layout(rect=[0, 0, 0.91, 0.93])
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Action Frequency", fontsize=12, fontweight="bold")

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


def plot_architecture(net_arch: list[int] | None = None, save_path: str = "results/architecture.png"):
    """Generate a high-level diagram of the Model B (Implicit) neural network architecture dynamically matching net_arch."""
    if net_arch is None:
        net_arch = [256, 256, 256]

    set_style()
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    # -----------------------------------------------------------------------
    # Helper to draw boxes
    # -----------------------------------------------------------------------
    def draw_box(x, y, w, h, text, color, alpha=0.9):
        rect = patches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.015",
            facecolor=color, edgecolor="#2D3436", alpha=alpha, linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=10, fontweight="bold")

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#2D3436", facecolor="#2D3436"))

    # Horizontal Inputs at y = 0.88
    y_in = 0.88
    draw_box(0.2, y_in, 0.26, 0.08, "Private Hand Info\n(Cards, Category, Score)", "#A8E6CF")
    draw_box(0.5, y_in, 0.28, 0.08, "Betting & Game State\n(Pot, Odds, Phase, Position)", "#DCEDC1")
    draw_box(0.8, y_in, 0.26, 0.08, "Implicit Opponent Modeling\n(Rolling Behavioral Stats)", "#FFD3B6")
    
    # Concatenation layer at y = 0.74
    y_concat = 0.74
    draw_box(0.5, y_concat, 0.86, 0.06, f"Observation Vector (N={23 + len(net_arch)} features)", "#FFAAA5")
    
    # Straight vertical connections into the concatenation vector
    draw_arrow(0.2, y_in-0.04, 0.2, y_concat+0.03)
    draw_arrow(0.5, y_in-0.04, 0.5, y_concat+0.03)
    draw_arrow(0.8, y_in-0.04, 0.8, y_concat+0.03)

    # Dynamic Hidden Layers
    n_layers = len(net_arch)
    y_start = 0.60
    y_end = 0.35
    y_positions = np.linspace(y_start, y_end, n_layers) if n_layers > 1 else [y_start]

    current_bottom_y = y_concat - 0.03
    for idx, (y_pos, width) in enumerate(zip(y_positions, net_arch)):
        draw_box(0.5, y_pos, 0.6, 0.045, f"Shared MLP Layer {idx+1} ({width} ReLU)", "#D1D8E0")
        draw_arrow(0.5, current_bottom_y, 0.5, y_pos + 0.0225)
        current_bottom_y = y_pos - 0.0225

    # Output Heads at y = 0.21
    y_out = 0.21
    draw_box(0.3, y_out, 0.35, 0.08, "Policy Head\n(Action Masked Logits)", "#A29BFE")
    draw_box(0.7, y_out, 0.35, 0.08, "Value Head\n(Expected Return)", "#FAB1A0")
    
    # Branching arrows from the last MLP layer to the output heads
    draw_arrow(0.5, current_bottom_y, 0.3, y_out + 0.04)
    draw_arrow(0.5, current_bottom_y, 0.7, y_out + 0.04)
    
    # Action Masking at y = 0.08
    y_mask = 0.08
    draw_box(0.3, y_mask, 0.35, 0.06, "Masking: Legal Actions Only", "#DFE6E9")
    draw_arrow(0.3, y_out-0.04, 0.3, y_mask+0.03)

    ax.set_title("Model Architecture & Inference Flow",
                 fontsize=16, fontweight="bold", pad=20)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_non_stationary_adaptation(
    results_implicit: dict,
    save_path: str = "results/non_stationary_adaptation.png"
):
    """Plot the real-time adaptation of Model B under non-stationary opponent shifts.

    Splits the plot into two separate, clean figures to avoid legend overlap:
    1. non_stationary_inference.png: Rolling behavioral inference tracking.
    2. non_stationary_performance.png: Win-rate and cumulative profit curves.
    """
    set_style()
    
    hand_rewards = results_implicit["hand_rewards"]
    rolling_stats = np.array(results_implicit["rolling_stats_history"])
    opponent_ids = results_implicit["true_opponent_ids"]
    n_hands = len(hand_rewards)
    run_dir = os.path.dirname(save_path)

    # Boundaries and transitions
    transitions = []
    for i in range(1, n_hands):
        if opponent_ids[i] != opponent_ids[i-1]:
            transitions.append(i)

    opp_names = ["Calling Station", "Maniac", "Rock"]
    boundaries = [0] + transitions + [n_hands]
    colors = ["#FFEAA7", "#FFD2D2", "#DFE6E9"]

    # =======================================================================
    # Figure 1: Implicit Opponent Modeling (Inference Tracking)
    # =======================================================================
    fig1, ax1 = plt.subplots(figsize=(13, 5))
    
    ax1.plot(rolling_stats[:, 7], label="Inferred VPIP %", color="#E17055", linewidth=2.0)
    ax1.plot(rolling_stats[:, 8], label="Inferred PFR %", color="#6C5CE7", linewidth=2.0)
    ax1.plot(rolling_stats[:, 9], label="Inferred Aggression Factor", color="#D63031", linewidth=2.0)
    ax1.plot(rolling_stats[:, 3], label="Inferred Avg Cards Drawn", color="#0984E3", linewidth=2.0, linestyle="--")

    # Era backgrounds and text labels using axes coordinate transforms
    for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        opp_id = opponent_ids[start]
        opp_name = opp_names[opp_id]
        ax1.axvspan(start, end, color=colors[opp_id % len(colors)], alpha=0.12)
        
        mid_frac = ((start + end) / 2) / n_hands
        ax1.text(
            mid_frac, 0.95, f"vs {opp_name}", transform=ax1.transAxes,
            ha="center", va="top", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#2D3436", alpha=0.9)
        )

    for t in transitions:
        ax1.axvline(x=t, color="#2D3436", linestyle=":", linewidth=1.5, alpha=0.8)

    ax1.set_ylabel("Inference Statistics (0-1)", fontsize=12)
    ax1.set_xlabel("Hand Number", fontsize=12)
    ax1.set_title("Implicit Opponent Modeling: Real-Time Behavioral Inference", fontsize=14, fontweight="bold", pad=15)
    
    # Legend outside on the right
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=10, framealpha=0.9)
    ax1.grid(alpha=0.15)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(0, n_hands)

    inference_path = os.path.join(run_dir, "non_stationary_inference.png")
    fig1.savefig(inference_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {inference_path}")
    plt.close(fig1)

    # =======================================================================
    # Figure 2: Online Exploitation (Performance Transition)
    # =======================================================================
    fig2, ax2 = plt.subplots(figsize=(13, 5))
    
    window = min(100, n_hands // 4)
    window = max(1, window)
    smoothed_profit = np.convolve(hand_rewards, np.ones(window)/window, mode="same")
    cum_profit = np.cumsum(hand_rewards)

    # Era backgrounds and text labels
    for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        opp_id = opponent_ids[start]
        opp_name = opp_names[opp_id]
        ax2.axvspan(start, end, color=colors[opp_id % len(colors)], alpha=0.12)
        
        mid_frac = ((start + end) / 2) / n_hands
        ax2.text(
            mid_frac, 0.95, f"vs {opp_name}", transform=ax2.transAxes,
            ha="center", va="top", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#2D3436", alpha=0.9)
        )

    for t in transitions:
        ax2.axvline(x=t, color="#2D3436", linestyle=":", linewidth=1.5, alpha=0.8)

    ax2.plot(smoothed_profit * 100, label=f"Smoothed Win-Rate (BB/100, window={window})", color="#00B894", linewidth=2.5)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(cum_profit, label="Cumulative Profit (Chips)", color="#2D3436", linewidth=1.5, linestyle="-.", alpha=0.7)
    
    ax2.set_ylabel("Win Rate (BB/100)", fontsize=12, color="#00B894")
    ax2_twin.set_ylabel("Cumulative Chips Won", fontsize=12, color="#2D3436")
    ax2.tick_params(axis="y", labelcolor="#00B894")
    ax2_twin.tick_params(axis="y", labelcolor="#2D3436")
    
    ax2.set_xlabel("Hand Number", fontsize=12)
    ax2.set_title("Online Exploitation: Performance Transition Curve", fontsize=14, fontweight="bold", pad=15)
    ax2.grid(alpha=0.15)
    ax2.axhline(y=0, color="#2D3436", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_xlim(0, n_hands)

    # Combine legends outside on the right
    lines, labels = ax2.get_legend_handles_labels()
    lines_twin, labels_twin = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines + lines_twin, labels + labels_twin, loc="upper left", bbox_to_anchor=(1.08, 1), borderaxespad=0, fontsize=10, framealpha=0.9)

    performance_path = os.path.join(run_dir, "non_stationary_performance.png")
    fig2.savefig(performance_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {performance_path}")
    plt.close(fig2)

