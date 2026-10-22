import json
import os
import sys

def analyze():
    history_path = "configs/holdem/hpo_trials_history_b.json"
    best_params_path = "configs/holdem/best_params_holdem_b.json"
    
    if not os.path.exists(history_path):
        print(f"Error: {history_path} does not exist yet.")
        return
        
    with open(history_path, "r", encoding="utf-8") as f:
        trials = json.load(f)
        
    print(f"Total trials recorded: {len(trials)}")
    
    # Sort by score descending
    sorted_trials = sorted(trials, key=lambda t: t.get("value", -9999), reverse=True)
    
    print("\n" + "="*80)
    print("TOP 5 TRIALS (by Composite Score = Return + 2.5 * WinRate)")
    print("="*80)
    for i, t in enumerate(sorted_trials[:5]):
        p = t.get("params", {})
        u = t.get("user_attrs", {})
        val = t.get("value", 0.0)
        wr = u.get("win_rate", 0.0)
        mr = u.get("mean_reward", 0.0)
        arch = p.get("net_arch", "N/A")
        attn = f"{p.get('embed_dim', '?')}d-{p.get('num_heads', '?')}h-{p.get('pooling', '?')}"
        lr = p.get("learning_rate", 0.0)
        ent = p.get("initial_ent_coef", 0.0)
        fp = p.get("fold_penalty", 0.0)
        print(f"#{i+1:02d} | Trial {t.get('trial_id', '?'):02d} | Score: {val:6.2f} | Win Rate: {wr:4.1f}% | Mean Return: {mr:+6.2f} BB | Attn: {attn:15s} | Arch: {arch:12s} | LR: {lr:.2e} | Ent: {ent:.3f} | FoldPen: {fp:.3f}")

    print("\n" + "="*80)
    print("CARD ATTENTION PARAMETER ANALYSIS")
    print("="*80)
    # Embed Dim
    dims = {}
    for t in trials:
        d = t.get("params", {}).get("embed_dim")
        val = t.get("value", 0.0)
        dims.setdefault(d, []).append(val)
    for d, vals in sorted(dims.items()):
        print(f"  embed_dim = {d:2d}: count={len(vals):2d}, avg_score={sum(vals)/len(vals):+6.2f}, max_score={max(vals):+6.2f}")

    # Num Heads
    heads = {}
    for t in trials:
        h = t.get("params", {}).get("num_heads")
        val = t.get("value", 0.0)
        heads.setdefault(h, []).append(val)
    for h, vals in sorted(heads.items()):
        print(f"  num_heads = {h:2d}: count={len(vals):2d}, avg_score={sum(vals)/len(vals):+6.2f}, max_score={max(vals):+6.2f}")

    # Pooling
    poolings = {}
    for t in trials:
        pol = t.get("params", {}).get("pooling")
        val = t.get("value", 0.0)
        poolings.setdefault(pol, []).append(val)
    for pol, vals in sorted(poolings.items()):
        print(f"  pooling = {pol:10s}: count={len(vals):2d}, avg_score={sum(vals)/len(vals):+6.2f}, max_score={max(vals):+6.2f}")

    # Net Arch
    archs = {}
    for t in trials:
        arc = t.get("params", {}).get("net_arch")
        val = t.get("value", 0.0)
        archs.setdefault(arc, []).append(val)
    print("\n" + "-"*80)
    print("NETWORK ARCHITECTURE ANALYSIS")
    for arc, vals in sorted(archs.items(), key=lambda x: str(x[0])):
        print(f"  net_arch = {str(arc):14s}: count={len(vals):2d}, avg_score={sum(vals)/len(vals):+6.2f}, max_score={max(vals):+6.2f}")

    if os.path.exists(best_params_path):
        print("\n" + "="*80)
        print("OPTIMAL PARAMS SAVED TO CONFIG:")
        print("="*80)
        with open(best_params_path, "r", encoding="utf-8") as f:
            bp = json.load(f)
        print(json.dumps(bp, indent=2))

if __name__ == "__main__":
    analyze()
