"""
games package — Multi-Game Poker Engine Collection.

Supported Variants:
1. five_card_draw: Standard High Poker (5-Card Draw)
2. triple_draw_27: Deuce-to-Seven Lowball (2-7 Triple Draw)
3. holdem: No-Limit Texas Hold'em (Community Card Poker)
"""

try:
    from games import five_card_draw
except ImportError:
    pass

try:
    from games import triple_draw_27
except ImportError:
    pass

try:
    from games import holdem
except ImportError:
    pass

__all__ = [
    "five_card_draw",
    "triple_draw_27",
    "holdem",
]
