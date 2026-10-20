"""
run_tests.py — Master test suite runner for CardShark-RL multi-game platform.

Supports:
- 5-Card Draw test suite (tests/test_five_card_draw/ or tests/)
- 2-7 Triple Draw test suite (tests/test_triple_draw_27/)
- Texas Hold'em test suite (tests/test_holdem/)
- Universal smoke test (tests/smoke_test.py)
"""
import os
import sys
import subprocess
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(PROJECT_ROOT, "tests")

FCD_TESTS = [
    "test_side_pot.py",
    "test_opponent_tracker.py",
    "test_multi_env.py",
    "test_gym_wrapper.py",
    "test_superhuman.py",
    "test_model_d.py",
    "test_model_e.py",
    "test_model_f.py",
    "test_npc_and_heuristics.py",
]


def run_test_script(test_script: str, env: dict) -> bool:
    print(f"\n[RUNNING] tests/{test_script}...")
    # Check in tests/test_five_card_draw first, then tests/
    candidates = [
        os.path.join(TEST_DIR, "test_five_card_draw", test_script),
        os.path.join(TEST_DIR, test_script),
    ]
    test_path = next((p for p in candidates if os.path.exists(p)), None)
    if not test_path:
        print(f"[ERROR] Test script {test_script} not found.")
        return False

    res = subprocess.run([sys.executable, test_path], capture_output=True, text=True, cwd=PROJECT_ROOT, env=env)
    if res.returncode == 0:
        print(f"[PASSED]  {test_script}")
        for line in res.stdout.strip().splitlines()[-4:]:
            print(f"   > {line}")
        return True
    else:
        print(f"[FAILED]  {test_script}")
        print(res.stdout)
        print(res.stderr)
        return False


def run_unittest_dir(test_dir_rel: str, env: dict) -> bool:
    print(f"\n[RUNNING] python -m unittest discover tests/{test_dir_rel}...")
    full_path = os.path.join(TEST_DIR, test_dir_rel)
    res = subprocess.run([sys.executable, "-m", "unittest", "discover", full_path], capture_output=True, text=True, cwd=PROJECT_ROOT, env=env)
    if res.returncode == 0:
        print(f"[PASSED]  tests/{test_dir_rel}")
        for line in res.stderr.strip().splitlines()[-2:]:
            print(f"   > {line}")
        return True
    else:
        print(f"[FAILED]  tests/{test_dir_rel}")
        print(res.stdout)
        print(res.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Master Test Runner for CardShark-RL")
    parser.add_argument("--game", "-g", type=str, default="all", choices=["all", "five_card_draw", "triple_draw_27", "holdem"], help="Game test suite to run (default: all)")
    args = parser.parse_args()

    print("=" * 65)
    print(f"  RUNNING CARDSHARK-RL TEST SUITE [GAME: {args.game.upper()}]")
    print("=" * 65)

    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    all_passed = True

    # 1. 5-Card Draw Tests
    if args.game in ("all", "five_card_draw"):
        print("\n--- 5-Card Draw Suites ---")
        for test in FCD_TESTS:
            if not run_test_script(test, env):
                all_passed = False

    # 2. 2-7 Triple Draw Tests
    if args.game in ("all", "triple_draw_27"):
        print("\n--- 2-7 Triple Draw Suites ---")
        if not run_unittest_dir("test_triple_draw_27", env):
            all_passed = False

    # 3. Texas Hold'em Tests
    if args.game in ("all", "holdem"):
        print("\n--- Texas Hold'em Suites ---")
        if not run_unittest_dir("test_holdem", env):
            all_passed = False

    print("\n" + "=" * 65)
    if all_passed:
        print("  ALL REQUESTED SUITES PASSED CLEANLY!")
    else:
        print("  SOME TESTS FAILED.")
    print("=" * 65)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
