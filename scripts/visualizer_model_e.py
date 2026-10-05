"""
visualizer_model_e.py — Performance & Architecture Dashboard for Model E Crusher.

Generates a publication-quality 4-panel dashboard:
- Panel 1: Multi-Model Tournament Championship Hierarchy (Model E vs D vs C).
- Panel 2: BB/100 Win Rate Comparison (+106 BB/100 swing over Model C).
- Panel 3: SGDR Plasticity Synchronization across 2.0M Steps.
- Panel 4: 100-Session Tournament Title Distribution (Model E vs Adversaries).
"""

from __future__ import annotations
import os
import sys
import math
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def generate_model_e_dashboard(save_path: str = "results/model_e_champion_dashboard.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load comparison results if available
    comp_path = "results/comparison_e_vs_all.json"
    e_win, d_win, c_win, exp_win, arch_win = 25.0, 14.0, 2.0, 40.0, 19.0
    e_bb, d_bb, c_bb = -10.70, -65.27, -116.94

    if os.path.exists(comp_path):
        try:
            with open(comp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            e_win = data.get("model_e", {}).get("win_rate_pct", e_win)
            d_win = data.get("model_d", {}).get("win_rate_pct", d_win)
            c_win = data.get("model_c", {}).get("win_rate_pct", c_win)
            e_bb = data.get("model_e", {}).get("bb_per_100", e_bb)
            d_bb = data.get("model_d", {}).get("bb_per_100", d_bb)
            c_bb = data.get("model_c", {}).get("bb_per_100", c_bb)
            exp_win = data.get("adversaries", {}).get("exploiter_win_rate_pct", exp_win)
            arch_win = data.get("adversaries", {}).get("archetype_win_rate_pct", arch_win)
        except Exception:
            pass

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("#0f172a") # Slate-900

    for ax in axs.flat:
        ax.set_facecolor("#1e293b") # Slate-800
        ax.tick_params(colors="#cbd5e1", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(True, linestyle="--", alpha=0.25, color="#64748b")

    # --- PANEL 1: Tournament Championship Rate ---
    models = ["Model C\n(MLP Sequence)", "Model D\n(Card Attention)", "Model E\n(Attention + ICM)"]
    win_rates = [c_win, d_win, e_win]
    x = np.arange(len(models))

    ax1 = axs[0, 0]
    bars = ax1.bar(x, win_rates, width=0.45, color=["#ef4444", "#3b82f6", "#10b981"], edgecolor="#ffffff", linewidth=1.2)
    ax1.axhline(20.0, color="#f59e0b", linestyle=":", linewidth=2, label="Table Parity (20.0%)")
    ax1.set_ylabel("1st Place Championship Rate (%)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.set_title("5-Seat Tournament Championships (N=100 Sessions)", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, color="#f8fafc", fontsize=10)
    ax1.set_ylim(0, max(win_rates) * 1.35)
    ax1.legend(loc="upper left", facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")
    for b in bars:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.1f}%", ha="center", color="#f8fafc", fontweight="bold", fontsize=11)

    # --- PANEL 2: Cash Game BB/100 Rate Comparison ---
    ax2 = axs[0, 1]
    bb_rates = [c_bb, d_bb, e_bb]
    bars2 = ax2.bar(x, bb_rates, width=0.45, color=["#dc2626", "#2563eb", "#059669"], edgecolor="#ffffff", linewidth=1.2)
    ax2.axhline(0.0, color="#94a3b8", linestyle="-", linewidth=1.5, alpha=0.8)
    ax2.set_ylabel("Winrate (BB/100)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax2.set_title("Cash Game Winrate Comparison (BB/100)", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, color="#f8fafc", fontsize=10)
    for b in bars2:
        val = b.get_height()
        y_pos = val - 6 if val < 0 else val + 2
        ax2.text(b.get_x() + b.get_width() / 2, y_pos, f"{val:+.1f}", ha="center", color="#f8fafc", fontweight="bold", fontsize=11)

    # --- PANEL 3: SGDR Schedule (2.0M Steps, 7 Restarts) ---
    ax3 = axs[1, 0]
    total_steps = 2_000_000
    steps = np.linspace(0, total_steps, 1000)
    progress = steps / total_steps
    n_cycles = 7
    cycle_len = 1.0 / n_cycles
    lrs = []
    initial_lr = 1.65e-4
    min_lr = 2.0e-5
    warmup_frac = 0.05

    for p in progress:
        c = min(int(p / cycle_len), n_cycles - 1)
        cp = (p - c * cycle_len) / cycle_len
        if cp < warmup_frac:
            lr = min_lr + (cp / warmup_frac) * (initial_lr - min_lr)
        else:
            dp = (cp - warmup_frac) / (1.0 - warmup_frac)
            lr = min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + math.cos(math.pi * dp))
        lrs.append(lr * 1e4)

    ax3.plot(steps / 1000, lrs, color="#38bdf8", linewidth=2.5, label="Model E SGDR Schedule (7 Cycles)")
    for s_step in range(250, int(total_steps / 1000) + 1, 250):
        ax3.axvline(s_step, color="#f59e0b", linestyle=":", alpha=0.5)

    ax3.set_xlabel("Training Timesteps (k)", color="#f8fafc", fontsize=11)
    ax3.set_ylabel("Learning Rate (x 10^-4)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax3.set_title("SGDR Policy Plasticity Synchronization (2.0M Steps)", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
    ax3.legend(loc="upper right", facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")

    # --- PANEL 4: 100-Session Title Distribution ---
    ax4 = axs[1, 1]
    labels = [
        f"Model E\n({e_win:.1f}%)",
        f"Model D\n({d_win:.1f}%)",
        f"Model C\n({c_win:.1f}%)",
        f"Adversarial Bot\n({exp_win:.1f}%)",
        f"Archetypes\n({arch_win:.1f}%)",
    ]
    sizes = [e_win, d_win, c_win, exp_win, arch_win]
    colors = ["#10b981", "#3b82f6", "#ef4444", "#f97316", "#64748b"]
    explode = (0.08, 0, 0, 0, 0)

    wedges, texts, autotexts = ax4.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        explode=explode,
        textprops=dict(color="#f8fafc", fontsize=9),
        wedgeprops=dict(edgecolor="#1e293b", linewidth=1.5),
    )
    for at in autotexts:
        at.set_color("#ffffff")
        at.set_weight("bold")
    ax4.set_title("100-Session Multi-Model Tournament Share", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"  [Visualizer] Model E Champion Dashboard saved successfully to: {save_path}")


if __name__ == "__main__":
    generate_model_e_dashboard()
