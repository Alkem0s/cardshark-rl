"""
evaluate.py — Unified Master Evaluation & Benchmarking Entrypoint for CardShark-RL.

Dispatches evaluation to the specified model benchmark:
- Model A: Heads-Up Benchmark vs 5 Behavioral Archetypes
- Model B: Multi-Player 5-Seat Tournament Benchmark & Tell Analysis
- Model C: Superhuman 87-Dim Tournament Sparring Benchmark & Tilt Exploitation Audit
- Model D: Next-Generation Champion Tournament Benchmark & Head-to-Head Titles
- Model E: Superhuman Crusher (91-dim ICM + Discard Masker) Comprehensive Tournament
- Model F: Co-Evolutionary Champion (97-dim Decoupled Actor) Comprehensive Tournament
- 2-7 Triple Draw: Deuce-to-Seven Lowball Benchmark (triple_draw_27)
- Texas Hold'em: No-Limit Texas Hold'em Benchmark (holdem)

Usage:
    python evaluate.py --model [a|b|c|d|e|f|27|holdem] [options]
    python evaluate.py --help
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
    "e": "compare_models_e_vs_all.py",
    "f": "compare_models_f_vs_all.py",
    "27": "triple_draw_27/compare_27.py",
    "triple_draw_27": "triple_draw_27/compare_27.py",
    "holdem": "holdem/compare_holdem.py",
    "texas_holdem": "holdem/compare_holdem.py",
}


def __getattr__(name: str):
    """Dynamic delegation to scripts/evaluate_model_a for main.py compatibility."""
    import scripts.evaluate_model_a as _ema
    if hasattr(_ema, name):
        return getattr(_ema, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")



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
        print("Usage: python evaluate.py --model [a|b|c|d|e|f|27|holdem] [options]")
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
