"""
play.py — Unified Master Play Launcher for CardShark Poker.

Launches the interactive game server and opens the poker table in your browser.
Usage:
    python play.py
    python play.py --port 5000
    python play.py --no-browser
"""

import os
import sys
import argparse
import webbrowser

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from game.server import run_server

def main():
    parser = argparse.ArgumentParser(description="Launch CardShark-RL Playable Game")
    parser.add_argument("--port", type=int, default=5000, help="Web server port (default: 5000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open web browser")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    run_server(port=args.port)

if __name__ == "__main__":
    main()
