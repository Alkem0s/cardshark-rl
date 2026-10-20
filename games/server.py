"""
games/server.py — Universal Poker HTTP Server & Bot Decision Engine for CardShark-RL.

Zero-dependency standard library HTTP server supporting:
1. 5-Card Draw (High Poker) — Models A-F + Heuristic Bots
2. 2-7 Triple Draw (Kansas City Lowball) — Model 27-A + Lowball Bots
3. No-Limit Texas Hold'em — Model Holdem-A + Hold'em Bots
"""

from __future__ import annotations
import os
import sys
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List, Optional
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from games.five_card_draw.npc import CardSharkNPC
from games.triple_draw_27.opponents import (
    PatStandBot, SmoothDrawBot, LowballExploiter, make_random_lowball_opponent
)
from games.holdem.opponents import (
    TAGBot, LAGBot, CallingStationBot, RandomHoldemBot, make_random_holdem_opponent
)

_FCD_NPC_CACHE: Dict[str, CardSharkNPC] = {}
_TD_BOT_CACHE: Dict[str, Any] = {}
_HOLDEM_BOT_CACHE: Dict[str, Any] = {}
_NEURAL_27_MODEL = None
_NEURAL_HOLDEM_MODEL = None


def get_fcd_npc(difficulty: str = "hard") -> CardSharkNPC:
    diff = difficulty.lower()
    if diff not in _FCD_NPC_CACHE:
        _FCD_NPC_CACHE[diff] = CardSharkNPC(difficulty=diff)
    return _FCD_NPC_CACHE[diff]


def get_td_bot(bot_type: str = "smooth"):
    b = bot_type.lower()
    if b not in _TD_BOT_CACHE:
        if "pat" in b:
            _TD_BOT_CACHE[b] = PatStandBot()
        elif "exploiter" in b:
            _TD_BOT_CACHE[b] = LowballExploiter()
        else:
            _TD_BOT_CACHE[b] = SmoothDrawBot()
    return _TD_BOT_CACHE[b]


def get_holdem_bot(bot_type: str = "tag"):
    b = bot_type.lower()
    if b not in _HOLDEM_BOT_CACHE:
        if "lag" in b:
            _HOLDEM_BOT_CACHE[b] = LAGBot()
        elif "station" in b:
            _HOLDEM_BOT_CACHE[b] = CallingStationBot()
        elif "random" in b:
            _HOLDEM_BOT_CACHE[b] = RandomHoldemBot()
        else:
            _HOLDEM_BOT_CACHE[b] = TAGBot()
    return _HOLDEM_BOT_CACHE[b]


class UniversalPokerRequestHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP handler for universal poker web UI and NPC decision requests."""

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
        if path in ("/", "/index.html", "/poker.html", "/lobby"):
            # Check games/poker.html first, then fallbacks
            candidates = [
                os.path.join(PROJECT_ROOT, "games", "poker.html"),
                os.path.join(PROJECT_ROOT, "game", "poker.html"),
                os.path.join(PROJECT_ROOT, "poker.html"),
            ]
            html_path = next((p for p in candidates if os.path.exists(p)), None)
            if html_path:
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
            resp = json.dumps({
                "status": "ok",
                "game_modes": ["five_card_draw", "triple_draw_27", "holdem"],
                "cached_fcd_bots": list(_FCD_NPC_CACHE.keys()),
            }).encode("utf-8")
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
            game_mode = data.get("game_mode", "five_card_draw").lower()

            if game_mode in ("triple_draw_27", "27", "lowball"):
                self._handle_27_action(data)
            elif game_mode in ("holdem", "texas_holdem", "nlhe"):
                self._handle_holdem_action(data)
            else:
                self._handle_five_card_draw_action(data)

        elif path == "/end_hand":
            for npc in _FCD_NPC_CACHE.values():
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

    def _handle_five_card_draw_action(self, data: dict):
        difficulty = data.get("difficulty", "hard")
        npc = get_fcd_npc(difficulty)

        hand = data.get("hand", [])
        pot = int(data.get("pot", 0))
        bet_to_call = int(data.get("bet_to_call", 0))
        phase = data.get("phase", "pre_draw")

        parsed_hand = [{"rank": c.get("rank", "2"), "suit": c.get("suit", "♠")} for c in hand]
        npc_chips = int(data.get("npc_chips", 1000))
        npc_seat = int(data.get("npc_seat", 0))
        button_seat = int(data.get("button_seat", 0))
        players_data = data.get("players", [])

        num_seats = 5
        seat_chips = [0] * num_seats
        seat_invested = [0] * num_seats
        seat_alive = [False] * num_seats
        seat_in_hand = [False] * num_seats
        seat_draw_counts = [-1] * num_seats

        if players_data:
            for i, p in enumerate(players_data[:num_seats]):
                seat_chips[i] = int(p.get("chips", 0))
                seat_invested[i] = int(p.get("invested", 0))
                seat_alive[i] = bool(p.get("is_alive", True))
                seat_in_hand[i] = bool(not p.get("folded", False) and seat_alive[i])
                seat_draw_counts[i] = int(p.get("draw_count", -1))
            total_table_chips = max(1, sum(seat_chips) + pot)
        else:
            seat_chips = [1000] * num_seats
            seat_invested = [pot // num_seats] * num_seats
            seat_alive = [True] * num_seats
            seat_in_hand = [True] * num_seats
            seat_draw_counts = [-1] * num_seats
            total_table_chips = 5000

        try:
            action_res = npc.get_action(
                npc_hand=parsed_hand,
                pot=pot,
                bet_to_call=bet_to_call,
                phase=phase,
                npc_chips=npc_chips,
                total_table_chips=total_table_chips,
                button_seat=button_seat,
                seat_chips=seat_chips,
                seat_invested=seat_invested,
                seat_alive=seat_alive,
                seat_in_hand=seat_in_hand,
                seat_draw_counts=seat_draw_counts,
                npc_seat=npc_seat,
            )

            resp_payload = {
                "action": action_res["action_type"],
                "amount": action_res["raise_amount"],
                "discard_indices": action_res["discard_indices"],
                "raw_action": action_res["raw_action_id"],
            }
            self._send_json(resp_payload)
        except Exception as e:
            self._send_json({"error": str(e), "action": "call", "amount": 0, "discard_indices": []}, status=500)

    def _handle_27_action(self, data: dict):
        bot_type = data.get("difficulty", "smooth")
        bot = get_td_bot(bot_type)

        hand = data.get("hand", [])
        from core.card import card_from_str
        # Cards as integer indices 0..51
        int_hand = []
        for c in hand:
            if isinstance(c, dict):
                r_str = str(c.get("rank", "2")).strip()
                s_str = str(c.get("suit", "♠")).strip()
                # Mapping
                r_map = {"2":0,"3":1,"4":2,"5":3,"6":4,"7":5,"8":6,"9":7,"10":8,"T":8,"J":9,"Q":10,"K":11,"A":12}
                s_map = {"♣":0,"c":0,"♦":1,"d":1,"♥":2,"h":2,"♠":3,"s":3}
                int_hand.append(s_map.get(s_str, 0) * 13 + r_map.get(r_str, 0))
            elif isinstance(c, int):
                int_hand.append(c)

        pot = int(data.get("pot", 0))
        bet_to_call = int(data.get("bet_to_call", 0))
        phase = data.get("phase", "bet_1")
        npc_chips = int(data.get("npc_chips", 1000))
        draw_round = int(data.get("draw_round", 0))

        if "draw" in phase:
            discards = bot.draw_action(int_hand, draw_round=draw_round)
            self._send_json({"action": "draw", "amount": 0, "discard_indices": discards})
        else:
            legal = [0, 1]  # Fold, Call
            if npc_chips > bet_to_call:
                legal.extend([2, 3, 4, 5])  # Min, Half-Pot, Pot, All-in
            act_idx = bot.bet_action(
                hand=int_hand,
                pot=pot,
                bet_to_call=bet_to_call,
                phase=phase,
                stack=npc_chips,
                legal_actions=legal,
            )
            act_map = {0: "fold", 1: "call", 2: "raise", 3: "raise", 4: "raise", 5: "all_in"}
            act_str = act_map.get(act_idx, "call")
            amount = 0
            if act_str == "raise":
                if act_idx == 2:
                    amount = min(npc_chips, bet_to_call + 20)
                elif act_idx == 3:
                    amount = min(npc_chips, bet_to_call + max(20, pot // 2))
                else:
                    amount = min(npc_chips, bet_to_call + max(20, pot))
            elif act_str == "all_in":
                amount = npc_chips

            self._send_json({"action": act_str, "amount": amount, "discard_indices": []})

    def _handle_holdem_action(self, data: dict):
        bot_type = data.get("difficulty", "tag")
        bot = get_holdem_bot(bot_type)

        hole_cards = data.get("hand", [])
        board_cards = data.get("community_cards", [])

        def _to_int(cards):
            res = []
            r_map = {"2":0,"3":1,"4":2,"5":3,"6":4,"7":5,"8":6,"9":7,"10":8,"T":8,"J":9,"Q":10,"K":11,"A":12}
            s_map = {"♣":0,"c":0,"♦":1,"d":1,"♥":2,"h":2,"♠":3,"s":3}
            for c in cards:
                if isinstance(c, dict):
                    res.append(s_map.get(str(c.get("suit","♠")), 0) * 13 + r_map.get(str(c.get("rank","2")), 0))
                elif isinstance(c, int):
                    res.append(c)
            return res

        int_hole = _to_int(hole_cards)
        int_board = _to_int(board_cards)

        pot = int(data.get("pot", 0))
        bet_to_call = int(data.get("bet_to_call", 0))
        street = data.get("phase", "preflop").lower()
        npc_chips = int(data.get("npc_chips", 1000))

        legal = [0, 1]
        if npc_chips > bet_to_call:
            legal.extend([2, 3, 4, 5])

        act_idx = bot.act(
            hole_cards=int_hole,
            board=int_board,
            pot=pot,
            bet_to_call=bet_to_call,
            street=street,
            stack=npc_chips,
            legal_actions=legal,
        )

        act_map = {0: "fold", 1: "call", 2: "raise", 3: "raise", 4: "raise", 5: "all_in"}
        act_str = act_map.get(act_idx, "call")
        amount = 0
        if act_str == "raise":
            if act_idx == 2:
                amount = min(npc_chips, bet_to_call + 20)
            elif act_idx == 3:
                amount = min(npc_chips, bet_to_call + max(20, pot // 2))
            else:
                amount = min(npc_chips, bet_to_call + max(20, pot))
        elif act_str == "all_in":
            amount = npc_chips

        self._send_json({"action": act_str, "amount": amount, "discard_indices": []})

    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def log_message(self, format, *args):
        pass


def run_server(port: int = 5000, host: str = "0.0.0.0"):
    """Starts the universal poker HTTP server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, UniversalPokerRequestHandler)
    print("=" * 65)
    print("  CardShark-RL Universal Multi-Game Poker Server Running")
    print(f"  Web Lobby: http://127.0.0.1:{port}")
    print("  Supported Modes: 5-Card Draw, 2-7 Triple Draw, Texas Hold'em")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    run_server(port=port)
