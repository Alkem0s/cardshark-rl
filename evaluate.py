"""
evaluate.py — Unified Master Evaluation & Benchmarking Entrypoint for CardShark-RL.

Dispatches evaluation to the specified model benchmark:
- Model A: Heads-Up Benchmark vs 5 Behavioral Archetypes
- Model B: Multi-Player 5-Seat Tournament Benchmark & Tell Analysis
- Model C: Superhuman 87-Dim Tournament Sparring Benchmark & Tilt Exploitation Audit
- Model D: Next-Generation Champion Tournament Benchmark & Head-to-Head Titles

Usage:
    python evaluate.py --model a [options]
    python evaluate.py --model b [options]
    python evaluate.py --model c [options]
    python evaluate.py --model d [options]
"""

import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

MODEL_SCRIPTS = {
    "a": "evaluate_model_a.py",
    "b": "evaluate_model_b.py",
    "c": "evaluate_model_c.py",
    "d": "evaluate_model_d.py",
}


def __getattr__(name: str):
    """Dynamic delegation to scripts/evaluate_model_a for main.py compatibility."""
    import scripts.evaluate_model_a as _ema
    if hasattr(_ema, name):
        return getattr(_ema, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")



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
        print("Usage: python evaluate.py --model [a|b|c|d] [options]")
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
