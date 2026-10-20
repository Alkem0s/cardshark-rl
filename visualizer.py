"""
visualizer.py — Unified Master Visualization Entrypoint for CardShark-RL.

Generates publication-quality performance dashboards:
- Model A: Heads-up win rates & learning curves
- Model B: 5-seat multi-player tournament metrics
- Model C: 4-panel superhuman dashboard & intra-hand sequence analysis
- Model D: 4-panel champion dashboard (Attention invariance & SGDR schedule)
- Model E: 4-panel superhuman crusher dashboard (ICM dynamics & dominated discard pruning)
- Model F: 4-panel co-evolutionary champion dashboard (Decoupled gradients & auxiliary belief)

Usage:
    python visualizer.py --model [a|b|c|d|e|f] [options]
    python visualizer.py --help
"""

import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

MODEL_SCRIPTS = {
    "a": "visualizer_model_a.py",
    "b": "visualizer_model_b.py",
    "c": "visualizer_model_c.py",
    "d": "visualizer_model_d.py",
    "e": "visualizer_model_e.py",
    "f": "visualizer_model_f.py",
}


def __getattr__(name: str):
    """Dynamic delegation to scripts/visualizer_model_a for main.py compatibility."""
    import scripts.visualizer_model_a as _vma
    if hasattr(_vma, name):
        return getattr(_vma, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")



def main():
    args = sys.argv[1:]
    if ("--help" in args or "-h" in args) and not any(k in args for k in ("--model", "-m")) and not (args and args[0].lower() in MODEL_SCRIPTS):
        print(__doc__)
        return

    model = "f"  # Default champion
    remaining_args = []

    i = 0
    while i < len(args):
        if args[i] in ("--model", "-m") and i + 1 < len(args):
            model = args[i + 1].lower()
            i += 2
        elif args[i].startswith("--model="):
            model = args[i].split("=")[1].lower()
            i += 1
        elif args[i].lower() in MODEL_SCRIPTS and i == 0:
            model = args[i].lower()
            i += 1
        else:
            remaining_args.append(args[i])
            i += 1

    if model not in MODEL_SCRIPTS:
        print(f"Error: Unknown model '{model}'. Valid options: {', '.join(MODEL_SCRIPTS.keys())}.")
        print("Usage: python visualizer.py --model [a|b|c|d|e|f] [options]")
        sys.exit(1)

    script_name = MODEL_SCRIPTS[model]
    script_path = os.path.join(SCRIPTS_DIR, script_name)

    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    cmd = [sys.executable, script_path] + remaining_args
    res = subprocess.run(cmd, env=env)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
