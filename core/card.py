"""
core/card.py — Universal 52-Card Deck and Card Encodings.

Universal across all poker game variants (5-Card Draw, 2-7 Lowball, Hold'em, Omaha):
Card encoding:
    card  = 0..51
    rank  = card % 13   (0=2, 1=3, ..., 8=T, 9=J, 10=Q, 11=K, 12=A)
    suit  = card // 13  (0=Clubs, 1=Diamonds, 2=Hearts, 3=Spades)
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

RANK_NAMES = "23456789TJQKA"
SUIT_NAMES = "cdhs"
SUIT_SYMBOLS = "♣♦♥♠"

def rank_of(card: int) -> int:
    """Return rank 0..12 (2..A)."""
    return card % 13

def suit_of(card: int) -> int:
    """Return suit 0..3 (0=c, 1=d, 2=h, 3=s)."""
    return card // 13

def card_str(card: int, symbols: bool = False) -> str:
    """Human-readable card string, e.g. 'As' or 'A♠'."""
    suits = SUIT_SYMBOLS if symbols else SUIT_NAMES
    return RANK_NAMES[rank_of(card)] + suits[suit_of(card)]

def hand_str(cards: List[int], symbols: bool = False) -> str:
    """Pretty-print a collection of cards."""
    return " ".join(card_str(c, symbols=symbols) for c in sorted(cards, key=rank_of, reverse=True))

def parse_card(s: str) -> int:
    """Parse string like 'As', '2c', 'Td' into integer ID 0..51."""
    s = s.strip()
    if len(s) < 2:
        raise ValueError(f"Invalid card string: {s}")
    r_char, s_char = s[0].upper(), s[1].lower()
    r = RANK_NAMES.index(r_char)
    if s_char in SUIT_NAMES:
        su = SUIT_NAMES.index(s_char)
    elif s_char in SUIT_SYMBOLS:
        su = SUIT_SYMBOLS.index(s_char)
    else:
        raise ValueError(f"Invalid suit in card string: {s}")
    return su * 13 + r

def normalize_rank(rank: int) -> float:
    """Normalize rank 0..12 to [-1.0, 1.0]."""
    return (rank / 6.0) - 1.0

def normalize_suit(suit: int) -> float:
    """Normalize suit 0..3 to [-1.0, 1.0]."""
    return (suit / 1.5) - 1.0


class Deck:
    """Fast integer deck with NumPy shuffling."""

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self, exclude: Optional[List[int]] = None):
        if exclude:
            exclude_set = set(exclude)
            cards = [c for c in range(52) if c not in exclude_set]
            self._cards = np.array(cards, dtype=np.int32)
        else:
            self._cards = np.arange(52, dtype=np.int32)
        self.rng.shuffle(self._cards)
        self._idx = 0

    def deal(self, n: int, in_use: Optional[List[int]] = None) -> List[int]:
        """Deal *n* cards from the top, avoiding cards in use if reshuffle needed."""
        total_len = len(self._cards)
        if self._idx + n > total_len:
            rem = max(0, total_len - self._idx)
            dealt = self._cards[self._idx : total_len].tolist() if rem > 0 else []
            used = list(in_use or []) + dealt
            self.reset(exclude=used)
            needed = n - rem
            if needed > 0 and len(self._cards) > 0:
                dealt.extend(self.deal(needed, in_use=used))
            return dealt
        cards = self._cards[self._idx : self._idx + n].tolist()
        self._idx += n
        return cards

    def draw(self, n: int, in_use: Optional[List[int]] = None) -> List[int]:
        """Draw *n* replacement cards (alias for deal)."""
        return self.deal(n, in_use=in_use)

    @property
    def remaining(self) -> int:
        return max(0, len(self._cards) - self._idx)
