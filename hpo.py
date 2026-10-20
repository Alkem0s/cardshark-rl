"""
hpo.py — Unified Master Hyperparameter Optimization Entrypoint for CardShark-RL.

Dispatches HPO to the specified game variant and optimization pipeline:
- 5-Card Draw (scripts/five_card_draw):
    Models: a, b, c, d, e, f
- 2-7 Triple Draw (scripts/triple_draw_27):
    Model: 27 (or 27_a)
- Texas Hold'em (scripts/holdem):
    Model: holdem (or holdem_a)

Usage:
    python hpo.py --game five_card_draw --model [a|b|c|d|e|f] [options]
    python hpo.py --game triple_draw_27 [options]
    python hpo.py --game holdem [options]
    python hpo.py --help
"""

import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

FCD_MODELS = {
    "a": "five_card_draw/hpo_model_a.py",
    "b": "five_card_draw/hpo_model_b.py",
    "c": "five_card_draw/hpo_model_c.py",
    "d": "five_card_draw/hpo_model_d.py",
    "e": "five_card_draw/hpo_model_d.py",
    "f": "five_card_draw/hpo_model_f.py",
}


def run_hpo(*args, **kwargs):
    """Programmatic HPO compatibility for main.py."""
    from scripts.five_card_draw.hpo_model_a import run_hpo as _rh
    return _rh(*args, **kwargs)


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
        elif i == 0 and args[i].lower() in ("five_card_draw", "fcd", "triple_draw_27", "27", "holdem", "texas_holdem"):
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
    if game in ("triple_draw_27", "27"):
        script_rel_path = "triple_draw_27/hpo_27.py"
    elif game in ("holdem", "texas_holdem"):
        script_rel_path = "holdem/hpo_holdem.py"
    elif game in ("five_card_draw", "fcd"):
        m = model or "f"
        script_rel_path = FCD_MODELS.get(m, "five_card_draw/hpo_model_f.py")
    elif model:
        if model in ("27", "27_a"):
            script_rel_path = "triple_draw_27/hpo_27.py"
        elif model in ("holdem", "holdem_a"):
            script_rel_path = "holdem/hpo_holdem.py"
        elif model in FCD_MODELS:
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
