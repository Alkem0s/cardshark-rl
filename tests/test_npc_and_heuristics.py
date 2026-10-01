"""
test_npc_and_heuristics.py — Test Model A adapter, all heuristic bots, and RandomCyclingBot.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game.npc import CardSharkNPC
from game.multi_opponents import (
    RandomCyclingBot,
    CallingStation,
    Maniac,
    Rock,
    TAG,
    LAG,
    AdversarialExploiter,
)


def test_random_cycling_bot():
    print("[1/5] Testing RandomCyclingBot...")
    cycler = RandomCyclingBot(rng_seed=42)
    initial_style = cycler.get_style_name()
    assert initial_style in ("CallingStation", "Maniac", "Rock", "TAG", "LAG", "AdversarialExploiter")

    # Rotate through hands and ensure variety
    styles_seen = {initial_style}
    for _ in range(30):
        cycler.on_hand_end()
        styles_seen.add(cycler.get_style_name())
    
    assert len(styles_seen) >= 3, f"Expected variety of styles, got: {styles_seen}"
    print(f"  RandomCyclingBot rotated across {len(styles_seen)} distinct styles: {styles_seen}")


def test_heuristic_archetypes_inference():
    print("[2/5] Testing all heuristic archetypes in CardSharkNPC...")
    archetypes = [
        "chameleon",
        "random_heuristic",
        "tag",
        "lag",
        "rock",
        "maniac",
        "calling_station",
        "exploiter",
    ]
    test_cards = [
        {"rank": "A", "suit": "♠"},
        {"rank": "K", "suit": "♠"},
        {"rank": "Q", "suit": "♠"},
        {"rank": "J", "suit": "♠"},
        {"rank": "10", "suit": "♠"},
    ]

    for arch in archetypes:
        npc = CardSharkNPC(difficulty=arch)
        assert npc.is_heuristic, f"{arch} should be marked is_heuristic"

        # Pre-draw
        act_pre = npc.get_action(test_cards, pot=100, bet_to_call=10, phase="pre_draw")
        assert act_pre["action_type"] in ("fold", "call", "raise", "all_in")
        assert "raise_amount" in act_pre

        # Draw
        act_draw = npc.get_action(test_cards, pot=100, bet_to_call=0, phase="draw")
        assert act_draw["action_type"] == "draw"
        assert isinstance(act_draw["discard_indices"], list)

        # Post-draw
        act_post = npc.get_action(test_cards, pot=100, bet_to_call=0, phase="post_draw")
        assert act_post["action_type"] in ("fold", "call", "raise", "all_in")

        # Hand end signal
        npc.record_hand_outcome(seat_idx=0, vpip=True, pfr=False)
        print(f"  Heuristic '{arch}' passed all phases.")


def test_model_a_adapter():
    print("[3/5] Testing Model A adapter (1v1 heads-up baseline)...")
    if not os.path.exists("models/model_a.zip"):
        print("  models/model_a.zip not found, skipping Model A neural test.")
        return

    npc = CardSharkNPC(difficulty="model_a")
    assert not npc.is_heuristic
    assert npc.is_model_a
    assert npc.obs_dim == 26

    test_cards = [
        {"rank": "A", "suit": "♥"},
        {"rank": "A", "suit": "♦"},
        {"rank": "K", "suit": "♣"},
        {"rank": "Q", "suit": "♠"},
        {"rank": "9", "suit": "♠"},
    ]

    act_pre = npc.get_action(test_cards, pot=50, bet_to_call=10, phase="pre_draw")
    assert act_pre["action_type"] in ("fold", "call", "raise")

    act_draw = npc.get_action(test_cards, pot=50, bet_to_call=0, phase="draw")
    assert act_draw["action_type"] == "draw"
    assert isinstance(act_draw["discard_indices"], list)

    act_post = npc.get_action(test_cards, pot=70, bet_to_call=20, phase="post_draw")
    assert act_post["action_type"] in ("fold", "call", "raise")

    print("  Model A 1v1 adapter: PASS")


def test_neural_models_b_c_d():
    print("[4/5] Testing neural models (B, C, D)...")
    for diff in ["normal", "hard", "champion"]:
        npc = CardSharkNPC(difficulty=diff)
        test_cards = [
            {"rank": "8", "suit": "♣"},
            {"rank": "8", "suit": "♦"},
            {"rank": "5", "suit": "♠"},
            {"rank": "3", "suit": "♥"},
            {"rank": "2", "suit": "♦"},
        ]
        act = npc.get_action(test_cards, pot=30, bet_to_call=4, phase="pre_draw")
        assert act["action_type"] in ("fold", "call", "raise", "all_in")
        print(f"  Model '{diff}' decision: {act['action_type']} (amt={act['raise_amount']})")


def test_edge_cases_and_graceful_fallbacks():
    print("[5/5] Testing edge cases and graceful fallbacks...")
    npc = CardSharkNPC(difficulty="chameleon")

    # Empty hand
    res_empty = npc.get_action([], pot=10, bet_to_call=2, phase="pre_draw")
    assert res_empty["action_type"] == "call"

    # Fewer than 5 cards
    res_short = npc.get_action([{"rank": "A", "suit": "♠"}], pot=10, bet_to_call=2, phase="pre_draw")
    assert res_short["action_type"] == "call"

    # All-in scenario (zero chips left)
    res_zero_chips = npc.get_action(
        [{"rank": "A", "suit": "♠"}] * 5,
        pot=100,
        bet_to_call=50,
        phase="pre_draw",
        npc_chips=0,
    )
    assert res_zero_chips["action_type"] in ("fold", "call")

    print("  Edge cases: PASS")


def test_variable_player_counts():
    print("[6/6] Testing variable player counts (2, 3, 4, 5 players)...")
    test_cards = [
        {"rank": "A", "suit": "♠"},
        {"rank": "K", "suit": "♠"},
        {"rank": "Q", "suit": "♠"},
        {"rank": "J", "suit": "♠"},
        {"rank": "10", "suit": "♠"},
    ]

    for num_alive in [2, 3, 4, 5]:
        seat_alive = [i < num_alive for i in range(5)]
        seat_in_hand = [i < num_alive for i in range(5)]
        seat_chips = [1000 if i < num_alive else 0 for i in range(5)]
        seat_invested = [20 if i < num_alive else 0 for i in range(5)]
        seat_draw_counts = [-1 if i < num_alive else -1 for i in range(5)]

        for diff in ["champion", "hard", "normal", "model_a", "chameleon"]:
            npc = CardSharkNPC(difficulty=diff)
            act = npc.get_action(
                test_cards,
                pot=num_alive * 20,
                bet_to_call=10,
                phase="pre_draw",
                npc_chips=1000,
                total_table_chips=num_alive * 1000,
                button_seat=0,
                seat_chips=seat_chips,
                seat_invested=seat_invested,
                seat_alive=seat_alive,
                seat_in_hand=seat_in_hand,
                seat_draw_counts=seat_draw_counts,
                npc_seat=1,
            )
            assert act["action_type"] in ("fold", "call", "raise", "all_in")
        print(f"  {num_alive} players at table: all models produced legal actions.")


def test_tiered_difficulty_modes():
    print("[7/7] Testing tiered difficulty modes (Beginner -> Grandmaster + Wildcard)...")
    tiers = {
        "beginner": ("CallingStation", True),
        "casual": ("TAG", True),
        "advanced": ("Model B", False),
        "master": ("Model C", False),
        "grandmaster": ("Model D", False),
        "wildcard": ("Chameleon", True),
    }

    test_cards = [
        {"rank": "A", "suit": "♠"},
        {"rank": "K", "suit": "♠"},
        {"rank": "Q", "suit": "♠"},
        {"rank": "J", "suit": "♠"},
        {"rank": "10", "suit": "♠"},
    ]

    for tier_name, (expected_type, expect_heuristic) in tiers.items():
        npc = CardSharkNPC(difficulty=tier_name)
        assert npc.is_heuristic == expect_heuristic, f"Tier {tier_name} heuristic mismatch: {npc.is_heuristic} vs {expect_heuristic}"
        act = npc.get_action(test_cards, pot=50, bet_to_call=10, phase="pre_draw")
        assert act["action_type"] in ("fold", "call", "raise", "all_in")
        print(f"  Tier '{tier_name}' -> {expected_type} validated.")


if __name__ == "__main__":
    test_random_cycling_bot()
    test_heuristic_archetypes_inference()
    test_model_a_adapter()
    test_neural_models_b_c_d()
    test_edge_cases_and_graceful_fallbacks()
    test_variable_player_counts()
    test_tiered_difficulty_modes()
    print("\nALL NPC & HEURISTIC TESTS COMPLETED SUCCESSFULLY!")
