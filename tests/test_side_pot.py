"""
test_side_pot.py — Unit test for side_pot.py.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game.side_pot import calculate_side_pots, resolve_showdown_payouts, PotTier


def test_simple_heads_up():
    investments = {0: 10, 1: 10}
    folded = set()
    pots = calculate_side_pots(investments, folded)
    assert len(pots) == 1
    assert pots[0].amount == 20
    assert pots[0].eligible == {0, 1}

    # Player 0 wins
    scores = {0: 100, 1: 50}
    payouts, details = resolve_showdown_payouts(pots, scores)
    assert payouts[0] == 20
    assert payouts[1] == 0
    print("test_simple_heads_up: PASS")


def test_three_way_unequal_allin():
    # Player 0 all-in for 20
    # Player 1 all-in for 50
    # Player 2 calls 50
    investments = {0: 20, 1: 50, 2: 50}
    folded = set()
    pots = calculate_side_pots(investments, folded)
    assert len(pots) == 2
    # Tier 1 (up to 20): 20 * 3 = 60, eligible {0, 1, 2}
    assert pots[0].amount == 60
    assert pots[0].eligible == {0, 1, 2}
    # Tier 2 (20 to 50): 30 * 2 = 60, eligible {1, 2}
    assert pots[1].amount == 60
    assert pots[1].eligible == {1, 2}

    # Case A: Short stack (0) has best hand (score 900), Player 1 has 500, Player 2 has 200
    scores = {0: 900, 1: 500, 2: 200}
    payouts, _ = resolve_showdown_payouts(pots, scores)
    # Player 0 wins main pot (60)
    # Player 1 wins side pot (60)
    assert payouts[0] == 60
    assert payouts[1] == 60
    assert payouts[2] == 0
    print("test_three_way_unequal_allin (Short stack best): PASS")

    # Case B: Player 1 has best hand (score 900)
    scores = {0: 200, 1: 900, 2: 500}
    payouts, _ = resolve_showdown_payouts(pots, scores)
    # Player 1 wins both main and side pots (120)
    assert payouts[0] == 0
    assert payouts[1] == 120
    assert payouts[2] == 0
    print("test_three_way_unequal_allin (Deep stack best): PASS")


def test_folded_chips_in_pot():
    # Player 0 bets 30 and folds
    # Player 1 all-in for 50
    # Player 2 calls 50
    investments = {0: 30, 1: 50, 2: 50}
    folded = {0}
    pots = calculate_side_pots(investments, folded)
    # Total pot is 30 + 50 + 50 = 130
    total_pot_chips = sum(p.amount for p in pots)
    assert total_pot_chips == 130
    for p in pots:
        assert 0 not in p.eligible
    scores = {1: 400, 2: 300}
    payouts, _ = resolve_showdown_payouts(pots, scores)
    assert payouts[1] == 130
    assert payouts.get(0, 0) == 0
    print("test_folded_chips_in_pot: PASS")


def test_split_pot_with_odd_chip():
    investments = {0: 25, 1: 25}
    folded = set()
    # Total pot: 51 with an ante of 1 added
    investments = {0: 26, 1: 25} # player 0 uncalled or ante
    # Let's say 3 players split 25 chips
    pots = calculate_side_pots({0: 10, 1: 10, 2: 10}, set())
    # Total pot = 30, split evenly = 10 each
    scores = {0: 500, 1: 500, 2: 500}
    payouts, _ = resolve_showdown_payouts(pots, scores, seat_order_from_button=[0, 1, 2])
    assert payouts[0] == 10
    assert payouts[1] == 10
    assert payouts[2] == 10

    pots = [PotTier(amount=31, eligible={0, 1, 2})]
    payouts, _ = resolve_showdown_payouts(pots, scores, seat_order_from_button=[1, 2, 0])
    # Button is before 1, so seat 1 gets the odd chip
    assert payouts[1] == 11
    assert payouts[2] == 10
    assert payouts[0] == 10
    print("test_split_pot_with_odd_chip: PASS")


if __name__ == "__main__":
    test_simple_heads_up()
    test_three_way_unequal_allin()
    test_folded_chips_in_pot()
    test_split_pot_with_odd_chip()
    print("ALL SIDE POT TESTS PASSED!")
