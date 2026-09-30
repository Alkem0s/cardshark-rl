"""
hpo.py — Unified Master Hyperparameter Optimization Entrypoint for CardShark-RL.

Dispatches HPO to the specified model optimization pipeline:
- Model A: Heads-Up Optuna Study
- Model B: Multi-Player 5-Seat PPO HPO
- Model C: Superhuman 87-Dim Multi-Player Study
- Model D: Advanced Staged (Decoupled) & Joint Multivariate HPO

Usage:
    python hpo.py --model a [options]
    python hpo.py --model b [options]
    python hpo.py --model c [options]
    python hpo.py --model d [options]
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
}


def run_hpo(*args, **kwargs):
    """Programmatic HPO compatibility for main.py."""
    from scripts.hpo_model_a import run_hpo as _rh
    return _rh(*args, **kwargs)



def main():
    args = sys.argv[1:]
    model = "d"  # Default champion
    remaining_args = []

    i = 0
    while i < len(args):
        if args[i] in ("--model", "-m") and i + 1 < len(args):
            model = args[i + 1].lower()
            i += 2
        elif args[i].startswith("--model="):
            model = args[i].split("=")[1].lower()
            i += 1
        elif args[i].lower() in ("a", "b", "c", "d") and i == 0:
            model = args[i].lower()
            i += 1
        else:
            remaining_args.append(args[i])
            i += 1

    if model not in MODEL_SCRIPTS:
        print(f"Error: Unknown model '{model}'. Valid options: a, b, c, d.")
        print("Usage: python hpo.py --model [a|b|c|d] [options]")
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
