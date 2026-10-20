"""
visualizer.py — Unified Master Visualization Entrypoint for CardShark-RL.

Generates publication-quality performance dashboards:
- 5-Card Draw (scripts/five_card_draw):
    Models: a, b, c, d, e, f

Usage:
    python visualizer.py --game five_card_draw --model [a|b|c|d|e|f] [options]
    python visualizer.py --help
"""

import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

FCD_MODELS = {
    "a": "five_card_draw/visualizer_model_a.py",
    "b": "five_card_draw/visualizer_model_b.py",
    "c": "five_card_draw/visualizer_model_c.py",
    "d": "five_card_draw/visualizer_model_d.py",
    "e": "five_card_draw/visualizer_model_e.py",
    "f": "five_card_draw/visualizer_model_f.py",
}


def __getattr__(name: str):
    """Dynamic delegation to scripts/five_card_draw/visualizer_model_a for main.py compatibility."""
    import scripts.five_card_draw.visualizer_model_a as _vma
    if hasattr(_vma, name):
        return getattr(_vma, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def main():
    args = sys.argv[1:]
    if not args or "--help" in args or "-h" in args:
        print(__doc__)
        return

    game = None
    model = None
    remaining_args = []

    i = 0
    while i < len(args):
        if args[i] in ("--game", "-g") and i + 1 < len(args):
            game = args[i + 1].lower()
            i += 2
        elif args[i].startswith("--game="):
            game = args[i].split("=")[1].lower()
            i += 1
        elif args[i] in ("--model", "-m") and i + 1 < len(args):
            model = args[i + 1].lower()
            i += 2
        elif args[i].startswith("--model="):
            model = args[i].split("=")[1].lower()
            i += 1
        elif i == 0 and args[i].lower() in ("five_card_draw", "fcd"):
            game = args[i].lower()
            i += 1
        elif i == 0 and args[i].lower() in FCD_MODELS:
            model = args[i].lower()
            i += 1
        else:
            remaining_args.append(args[i])
            i += 1

    # Resolve target script
    script_rel_path = None
    if game in ("five_card_draw", "fcd"):
        m = model or "f"
        script_rel_path = FCD_MODELS.get(m, "five_card_draw/visualizer_model_f.py")
    elif model and model in FCD_MODELS:
        script_rel_path = FCD_MODELS[model]

    if not script_rel_path:
        print("Error: Please specify a valid --game or --model.")
        print(__doc__)
        sys.exit(1)

    script_path = os.path.join(SCRIPTS_DIR, script_rel_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    cmd = [sys.executable, script_path] + remaining_args
    res = subprocess.run(cmd, env=env)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
