"""
game/card_utils.py — Backward-compatibility shim.
Delegates to games.five_card_draw.card_utils and rules.
"""
from games.five_card_draw.rules import *
from games.five_card_draw.card_utils import *
