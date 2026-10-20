"""
train.py — Unified Master Training Entrypoint for CardShark-RL.

Dispatches training to the specified model pipeline:
- Model A: Heads-Up Baseline (One-hot & rolling stats)
- Model B: Multi-Player 5-Seat Baseline (63-dim observation)
- Model C: Superhuman Autonomous Agent (87-dim sequence memory + League Sparring)
- Model D: Next-Generation Champion (Permutation-Invariant Card Attention + SGDR Restarts)
- Model E: Superhuman Crusher (91-dim ICM + Dominated Discard Masking)
- Model F: Co-Evolutionary Champion (97-dim Decoupled Actor + Auxiliary Belief Head)
- 2-7 Triple Draw: Deuce-to-Seven Lowball (triple_draw_27)
- Texas Hold'em: No-Limit Texas Hold'em (holdem)

Usage:
    python train.py --model [a|b|c|d|e|f|27|holdem] [options]
    python train.py --help
"""

import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

MODEL_SCRIPTS = {
    "a": "train_model_a.py",
    "b": "train_model_b.py",
    "c": "train_model_c.py",
    "d": "train_model_d.py",
    "e": "train_model_e.py",
    "f": "train_model_f.py",
    "27": "triple_draw_27/train_27.py",
    "triple_draw_27": "triple_draw_27/train_27.py",
    "holdem": "holdem/train_holdem.py",
    "texas_holdem": "holdem/train_holdem.py",
}


def train_model(*args, **kwargs):
    """Programmatic training compatibility for main.py."""
    from scripts.train_model_a import train_model as _tm
    return _tm(*args, **kwargs)


def linear_schedule(*args, **kwargs):
    """Programmatic schedule compatibility for main.py."""
    from scripts.train_model_a import linear_schedule as _ls
    return _ls(*args, **kwargs)



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
        print("Usage: python train.py --model [a|b|c|d|e|f|27|holdem] [options]")
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
