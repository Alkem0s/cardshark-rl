"""
run_tests.py — Master test suite runner for CardShark-RL multi-player overhaul.
"""
import sys
import subprocess

TEST_SCRIPTS = [
    "test_side_pot.py",
    "test_opponent_tracker.py",
    "test_multi_env.py",
    "test_gym_wrapper.py",
]

def main():
    print("=" * 60)
    print("  RUNNING CARDSHARK-RL MULTI-PLAYER TEST SUITE")
    print("=" * 60)

    all_passed = True
    for test in TEST_SCRIPTS:
        print(f"\n[RUNNING] {test}...")
        res = subprocess.run([sys.executable, test], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"[PASSED]  {test}")
            for line in res.stdout.strip().splitlines():
                print(f"   > {line}")
        else:
            print(f"[FAILED]  {test}")
            print(res.stderr)
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("  ALL MULTI-PLAYER SUITES PASSED CLEANLY!")
    else:
        print("  SOME TESTS FAILED.")
    print("=" * 60)

    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    main()
