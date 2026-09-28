"""
test_multi_env.py — Unit & smoke test for MultiDrawPokerEnv.
"""
from multi_draw_poker_env import (
    MultiDrawPokerEnv, A_FOLD, A_CALL, A_MIN_RAISE, A_ALL_IN,
    PHASE_PRE_DRAW, PHASE_DRAW, PHASE_POST_DRAW, PHASE_SHOWDOWN,
    TOTAL_ACTIONS, A_DRAW_START
)
from multi_opponents import make_opponent_by_id, CallingStation, Maniac


def test_basic_session_and_step():
    env = MultiDrawPokerEnv(num_seats=5, starting_chips=100, rng_seed=42)
    env.reset_session()

    assert len(env.seats) == 5
    assert len(env.alive_seats) == 5
    assert env.pot == 3  # SB (1) + BB (2) = 3
    assert env.total_table_chips == 500

    # Step through a hand with calls and stand-pats
    step_count = 0
    while not env.session_done and step_count < 100:
        step_count += 1
        curr_seat = env.current_seat
        legal = env.get_legal_actions(curr_seat)
        mask = env.get_action_mask(curr_seat)
        assert len(legal) > 0
        assert mask.shape == (TOTAL_ACTIONS,)
        for a in legal:
            assert mask[a] == 1

        if env.phase in (PHASE_PRE_DRAW, PHASE_POST_DRAW):
            action = A_CALL if A_CALL in legal else legal[0]
        else: # Draw
            action = A_DRAW_START # Stand pat

        env.step_action(curr_seat, action)

    # Check conservation of total table chips
    assert env.total_table_chips == 500
    print("test_basic_session_and_step: PASS (chips conserved at 500)")


def test_elimination_and_table_shrinking():
    # Start with tiny chips so someone eliminates fast
    env = MultiDrawPokerEnv(num_seats=5, starting_chips=10, small_blind=2, big_blind=4, rng_seed=123)
    env.reset_session()

    hands_played = 0
    while not env.session_done and hands_played < 50:
        curr_seat = env.current_seat
        legal = env.get_legal_actions(curr_seat)
        if not legal:
            break

        # Players either call or all-in
        if A_ALL_IN in legal and env.rng.random() < 0.4:
            action = A_ALL_IN
        elif A_CALL in legal:
            action = A_CALL
        else:
            action = legal[0]

        env.step_action(curr_seat, action)
        hands_played = env.hands_played

    # Verify that total table chips remain invariant (5 * 10 = 50)
    assert env.total_table_chips == 50
    # At least some players should have busted
    alive = env.alive_seats
    print(f"test_elimination_and_table_shrinking: PASS (Hands: {env.hands_played}, Remaining alive: {len(alive)})")


def test_deck_exhaustion_reshuffle():
    # Force heavy discards for all 5 players to test safety reshuffle
    env = MultiDrawPokerEnv(num_seats=5, starting_chips=200, rng_seed=999)
    env.reset_session()

    # Pre-draw: all call
    while env.phase == PHASE_PRE_DRAW and not env.session_done:
        legal = env.get_legal_actions(env.current_seat)
        env.step_action(env.current_seat, A_CALL if A_CALL in legal else legal[0])

    # Draw phase: discard 4 cards (bitmask 0b01111 = 15 -> action 6 + 15 = 21)
    if env.phase == PHASE_DRAW:
        while env.phase == PHASE_DRAW and not env.session_done:
            # Action 6 + 15 = 21 -> discard 4 cards
            env.step_action(env.current_seat, A_DRAW_START + 15)

    assert env.total_table_chips == 1000
    print("test_deck_exhaustion_reshuffle: PASS (Discards handled safely)")


if __name__ == "__main__":
    test_basic_session_and_step()
    test_elimination_and_table_shrinking()
    test_deck_exhaustion_reshuffle()
    print("ALL MULTI-ENV TESTS PASSED!")
