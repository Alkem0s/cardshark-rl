"""
hpo.py — Unified Master Hyperparameter Optimization Entrypoint for CardShark-RL.

Dispatches HPO to the specified model optimization pipeline:
- Model A: Heads-Up Optuna Study
- Model B: Multi-Player 5-Seat PPO HPO
- Model C: Superhuman 87-Dim Multi-Player Study
- Model D: Advanced Staged (Decoupled) & Joint Multivariate HPO
- Model E: Model E ICM & Feature Optimization
- Model F: Decoupled Actor-Critic & Auxiliary Belief Multi-Task HPO
- 2-7 Triple Draw: Deuce-to-Seven Lowball Optuna HPO (triple_draw_27)
- Texas Hold'em: No-Limit Texas Hold'em Optuna HPO (holdem)

Usage:
    python hpo.py --model [a|b|c|d|e|f|27|holdem] [options]
    python hpo.py --help
"""

import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

MODEL_SCRIPTS = {
    "a": "hpo_model_a.py",
    "b": "hpo_model_b.py",
    "c": "hpo_model_c.py",
    "d": "hpo_model_d.py",
    "e": "hpo_model_d.py",
    "f": "hpo_model_f.py",
    "27": "triple_draw_27/hpo_27.py",
    "triple_draw_27": "triple_draw_27/hpo_27.py",
    "holdem": "holdem/hpo_holdem.py",
    "texas_holdem": "holdem/hpo_holdem.py",
}


def run_hpo(*args, **kwargs):
    """Programmatic HPO compatibility for main.py."""
    from scripts.hpo_model_a import run_hpo as _rh
    return _rh(*args, **kwargs)



def main():
    args = sys.argv[1:]
    if ("--help" in args or "-h" in args) and not any(k in args for k in ("--model", "-m", "--game", "-g")) and not (args and args[0].lower() in MODEL_SCRIPTS):
        print(__doc__)
        return

    model = "f"  # Default 5-card champion
    remaining_args = []

    i = 0
    while i < len(args):
        if args[i] in ("--model", "-m", "--game", "-g") and i + 1 < len(args):
            model = args[i + 1].lower()
            i += 2
        elif args[i].startswith("--model="):
            model = args[i].split("=")[1].lower()
            i += 1
        elif args[i].startswith("--game="):
            model = args[i].split("=")[1].lower()
            i += 1
        elif args[i].lower() in MODEL_SCRIPTS and i == 0:
            model = args[i].lower()
            i += 1
        else:
            remaining_args.append(args[i])
            i += 1

    if model not in MODEL_SCRIPTS:
        print(f"Error: Unknown model or game '{model}'. Valid options: {', '.join(MODEL_SCRIPTS.keys())}.")
        print("Usage: python hpo.py --model [a|b|c|d|e|f|27|holdem] [options]")
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
