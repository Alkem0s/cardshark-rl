"""
game/server.py — Backward-compatibility shim.
Delegates to games.server.
"""
from games.server import *

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    run_server(port=port)
