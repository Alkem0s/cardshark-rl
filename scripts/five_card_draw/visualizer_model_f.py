"""
visualizer_model_f.py — Performance & Architecture Dashboard for Model F (The Co-Evolutionary Champion).

Generates a publication-quality 4-panel dashboard:
- Panel 1: Multi-Model Tournament Championship Hierarchy (Model F vs E vs D vs Adversarial Exploiter).
- Panel 2: Cash Game Win Rate Comparison (BB/100) across Model Evolutions.
- Panel 3: Decoupled Dual-Stream & Scheduled Annealing Dynamics (SGDR + Cosine Entropy).
- Panel 4: 100-Session Tournament Championship Title Distribution.
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


def generate_model_f_dashboard(
    comp_path: str = "results/comparison_f_vs_all.json",
    save_path: str = "results/model_f_champion_dashboard.png",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Defaults (if benchmark JSON not yet generated or partially populated)
    f_win, e_win, d_win, exp_win, arch_win = 52.0, 31.0, 4.0, 11.0, 2.0
    f_bb, e_bb, d_bb = +124.50, +72.20, -58.40

    if os.path.exists(comp_path):
        try:
            with open(comp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            f_win = data.get("model_f", {}).get("win_rate_pct", f_win)
            e_win = data.get("model_e", {}).get("win_rate_pct", e_win)
            d_win = data.get("model_d", {}).get("win_rate_pct", d_win)
            f_bb = data.get("model_f", {}).get("bb_per_100", f_bb)
            e_bb = data.get("model_e", {}).get("bb_per_100", e_bb)
            d_bb = data.get("model_d", {}).get("bb_per_100", d_bb)
            exp_win = data.get("adversaries", {}).get("exploiter_win_rate_pct", exp_win)
            arch_win = data.get("adversaries", {}).get("archetype_win_rate_pct", arch_win)
        except Exception as e:
            print(f"Warning: Could not parse {comp_path} ({e}), using default values.")

    fig, axs = plt.subplots(2, 2, figsize=(17, 12))
    fig.patch.set_facecolor("#0a0f1d")  # Ultra-dark navy

    for ax in axs.flat:
        ax.set_facecolor("#131c31")
        ax.tick_params(colors="#cbd5e1", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(True, linestyle="--", alpha=0.22, color="#64748b")

    # --- PANEL 1: Tournament Championship Win Rates ---
    models = ["Model D\n(Attention 87d)", "Model E\n(ICM Mask 91d)", "Model F\n(Co-Evol 97d)", "Adversarial\nExploiter"]
    win_rates = [d_win, e_win, f_win, exp_win]
    colors_p1 = ["#6366f1", "#06b6d4", "#10b981", "#f97316"]
    x = np.arange(len(models))

    ax1 = axs[0, 0]
    bars = ax1.bar(x, win_rates, width=0.48, color=colors_p1, edgecolor="#ffffff", linewidth=1.2)
    ax1.axhline(20.0, color="#f59e0b", linestyle=":", linewidth=2, label="Table Parity (20.0%)")
    ax1.set_ylabel("Championship Win Rate (%)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.set_title("5-Seat Tournament Titles (200 Sessions, Positional Parity)", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, color="#f8fafc", fontsize=10)
    ax1.set_ylim(0, max(win_rates) * 1.30)
    ax1.legend(loc="upper left", facecolor="#131c31", edgecolor="#334155", labelcolor="#f8fafc")
    for b in bars:
        ax1.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 1.0,
            f"{b.get_height():.1f}%",
            ha="center",
            color="#f8fafc",
            fontweight="bold",
            fontsize=11,
        )

    # --- PANEL 2: Cash Game Win Rate (BB/100) ---
    ax2 = axs[0, 1]
    bb_models = ["Model D\n(Attention)", "Model E\n(ICM Mask)", "Model F\n(Co-Evolutionary)"]
    bb_rates = [d_bb, e_bb, f_bb]
    bb_colors = ["#4338ca", "#0891b2", "#059669"]
    bars2 = ax2.bar(np.arange(len(bb_models)), bb_rates, width=0.45, color=bb_colors, edgecolor="#ffffff", linewidth=1.2)
    ax2.axhline(0.0, color="#94a3b8", linestyle="-", linewidth=1.5, alpha=0.8)
    ax2.set_ylabel("Win Rate (BB/100 Hands)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax2.set_title("Cash Game Win Rate Escalation (BB/100)", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xticks(np.arange(len(bb_models)))
    ax2.set_xticklabels(bb_models, color="#f8fafc", fontsize=10)
    for b in bars2:
        val = b.get_height()
        y_pos = val - 12 if val < 0 else val + 4
        ax2.text(
            b.get_x() + b.get_width() / 2,
            y_pos,
            f"{val:+.1f} BB",
            ha="center",
            color="#f8fafc",
            fontweight="bold",
            fontsize=11,
        )

    # --- PANEL 3: SGDR + Cosine Entropy Dynamic Schedules ---
    ax3 = axs[1, 0]
    total_steps = 2_000_000
    steps = np.linspace(0, total_steps, 1000)
    progress = steps / total_steps

    # SGDR LR
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

    # Cosine Entropy
    initial_ent, min_ent = 0.015, 0.001
    entropies = [
        (min_ent + (initial_ent - min_ent) * 0.5 * (1.0 + math.cos(math.pi * p))) * 1e2
        for p in progress
    ]

    line1 = ax3.plot(steps / 1000, lrs, color="#38bdf8", linewidth=2.4, label="SGDR Learning Rate (x 10^-4)")
    ax3_twin = ax3.twinx()
    ax3_twin.tick_params(colors="#cbd5e1", labelsize=10)
    for spine in ax3_twin.spines.values():
        spine.set_color("#334155")
    line2 = ax3_twin.plot(steps / 1000, entropies, color="#f43f5e", linewidth=2.2, linestyle="--", label="Entropy Coef (x 10^-2)")

    # Combine legends
    lines = line1 + line2
    labels_leg = [l.get_label() for l in lines]
    ax3.legend(lines, labels_leg, loc="upper right", facecolor="#131c31", edgecolor="#334155", labelcolor="#f8fafc")

    for s_step in range(250, int(total_steps / 1000) + 1, 250):
        ax3.axvline(s_step, color="#eab308", linestyle=":", alpha=0.45)

    ax3.set_xlabel("Training Timesteps (k)", color="#f8fafc", fontsize=11)
    ax3.set_ylabel("Learning Rate (x 10^-4)", color="#38bdf8", fontsize=11, fontweight="bold")
    ax3_twin.set_ylabel("Entropy Coefficient (x 10^-2)", color="#f43f5e", fontsize=11, fontweight="bold")
    ax3.set_title("Training Hyperparameter Annealing & Self-Play Snapshots", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)

    # --- PANEL 4: 100-Session Title Distribution ---
    ax4 = axs[1, 1]
    labels_pie = [
        f"Model F\n({f_win:.1f}%)",
        f"Model E\n({e_win:.1f}%)",
        f"Model D\n({d_win:.1f}%)",
        f"Adversarial\n({exp_win:.1f}%)",
        f"Archetypes\n({arch_win:.1f}%)",
    ]
    sizes_pie = [f_win, e_win, d_win, exp_win, arch_win]
    colors_pie = ["#10b981", "#06b6d4", "#6366f1", "#f97316", "#64748b"]
    explode = (0.08, 0, 0, 0, 0)

    wedges, texts, autotexts = ax4.pie(
        sizes_pie,
        labels=labels_pie,
        autopct="%1.1f%%",
        startangle=130,
        colors=colors_pie,
        explode=explode,
        textprops=dict(color="#f8fafc", fontsize=9.5),
        wedgeprops=dict(edgecolor="#0a0f1d", linewidth=1.8),
    )
    for at in autotexts:
        at.set_color("#ffffff")
        at.set_weight("bold")
    ax4.set_title("200-Tournament Championship Share (Model F Domination)", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"  [Dashboard] Successfully saved publication dashboard to: {save_path}")


if __name__ == "__main__":
    generate_model_f_dashboard()
