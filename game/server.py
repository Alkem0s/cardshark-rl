"""
game/server.py — Zero-Dependency Standalone HTTP Server & Bot API for CardShark Poker.

Uses Python's built-in standard library (http.server, json, os) so that the game
runs on ANY computer with ZERO extra pip packages (no Flask or Flask-CORS required).
"""

from __future__ import annotations
import os
import sys
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any

GAME_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(GAME_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from game.npc import CardSharkNPC

_NPC_CACHE: Dict[str, CardSharkNPC] = {}

def get_npc(difficulty: str = "hard") -> CardSharkNPC:
    diff = difficulty.lower()
    if diff not in _NPC_CACHE:
        _NPC_CACHE[diff] = CardSharkNPC(difficulty=diff)
    return _NPC_CACHE[diff]


class PokerRequestHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP handler for poker web UI and NPC decision requests."""

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html", "/poker.html"):
            html_path = os.path.join(GAME_DIR, "poker.html")
            if not os.path.exists(html_path):
                html_path = os.path.join(PROJECT_ROOT, "poker.html")
            if os.path.exists(html_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self._send_cors_headers()
                self.end_headers()
                with open(html_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        elif path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            resp = json.dumps({"status": "ok", "cached_bots": list(_NPC_CACHE.keys())}).encode("utf-8")
            self.wfile.write(resp)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = self.path.split("?")[0]
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            data = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            data = {}

        if path == "/npc_action":
            difficulty = data.get("difficulty", "hard")
            npc = get_npc(difficulty)

            hand = data.get("hand", [])
            pot = int(data.get("pot", 0))
            bet_to_call = int(data.get("bet_to_call", 0))
            phase = data.get("phase", "pre_draw")

            parsed_hand = [{"rank": c.get("rank", "2"), "suit": c.get("suit", "♠")} for c in hand]

            num_seats = 5
            seat_chips = [1000] * num_seats
            seat_invested = [pot // num_seats] * num_seats
            seat_alive = [True] * num_seats
            seat_in_hand = [True] * num_seats
            seat_draw_counts = [-1] * num_seats

            try:
                action_res = npc.get_action(
                    npc_hand=parsed_hand,
                    pot=pot,
                    bet_to_call=bet_to_call,
                    phase=phase,
                    npc_chips=1000,
                    total_table_chips=5000,
                    button_seat=0,
                    seat_chips=seat_chips,
                    seat_invested=seat_invested,
                    seat_alive=seat_alive,
                    seat_in_hand=seat_in_hand,
                    seat_draw_counts=seat_draw_counts,
                )

                resp_payload = {
                    "action": action_res["action_type"],
                    "amount": action_res["raise_amount"],
                    "discard_indices": action_res["discard_indices"],
                    "raw_action": action_res["raw_action_id"],
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(resp_payload).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e), "action": "call", "amount": 0, "discard_indices": []}).encode("utf-8"))

        elif path == "/end_hand":
            for npc in _NPC_CACHE.values():
                npc.record_hand_outcome(
                    seat_idx=0,
                    vpip=bool(data.get("vpip", False)),
                    pfr=bool(data.get("pfr", False)),
                    af_bet=bool(data.get("post_draw_action", 0) == 2),
                    af_call=bool(data.get("post_draw_action", 0) == 1),
                    folded=bool(data.get("folded", False)),
                    draw_count=int(data.get("draw_count", 0)),
                )

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress routine log spew to keep terminal clean
        pass


def run_server(port: int = 5000, host: str = "0.0.0.0"):
    """Starts the zero-dependency poker HTTP server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, PokerRequestHandler)
    print("=" * 60)
    print(f"  CardShark-RL Poker Server Running")
    print(f"  Open in browser: http://127.0.0.1:{port}")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    run_server(port=port)
