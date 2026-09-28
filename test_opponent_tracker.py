"""
test_opponent_tracker.py — Unit test for opponent_tracker.py.
"""
from opponent_tracker import TableOpponentTracker, SeatProfile


def test_prior_stability():
    profile = SeatProfile(seat_idx=0)
    feats = profile.get_feature_vector()
    # Check prior values
    # VPIP ~0.28, PFR ~0.15, AF ~0.30, Fold ~0.50, Draw ~0.40
    assert abs(feats[0] - 0.28) < 1e-4
    assert abs(feats[1] - 0.15) < 1e-4
    assert abs(feats[2] - 0.30) < 1e-4
    assert abs(feats[3] - 0.50) < 1e-4
    assert abs(feats[4] - 0.40) < 1e-4
    print("test_prior_stability: PASS")


def test_bayesian_updates():
    profile = SeatProfile(seat_idx=1)
    # A Maniac plays 3 hands: VPIP True all 3 times, PFR True all 3 times
    for _ in range(3):
        profile.record_hand(vpip=True, pfr=True, post_draw_raised=True, faced_raise_and_folded=False, cards_drawn=1)

    feats = profile.get_feature_vector()
    # Prior had weight 5 (1.4 successes). Added 3 successes in 3 trials.
    # Posterior VPIP = (1.4 + 3) / (5 + 3) = 4.4 / 8 = 0.55
    assert abs(feats[0] - 0.55) < 1e-4
    # PFR prior was 0.75 / 5. Posterior PFR = (0.75 + 3) / 8 = 3.75 / 8 = 0.46875
    assert abs(feats[1] - 0.46875) < 1e-4
    # Smooth progression: didn't jump instantly to 1.0!
    assert 0.30 < feats[0] < 1.0
    print("test_bayesian_updates: PASS")


def test_relative_vector():
    tracker = TableOpponentTracker(num_seats=5)
    # Hero is at seat 2
    # Seats 0, 1, 2, 3, 4
    # Relative offsets:
    # Slot +1 -> seat (2+1)%5 = 3
    # Slot +2 -> seat (2+2)%5 = 4
    # Slot +3 -> seat (2+3)%5 = 0
    # Slot +4 -> seat (2+4)%5 = 1

    is_alive = [True, True, True, True, False] # Seat 4 is eliminated
    in_hand = [True, False, True, True, False]
    stacks = [100, 200, 150, 250, 0]
    investments = [10, 0, 10, 20, 0]
    pot = 40
    draw_counts = [2, -1, 3, 1, -1]

    # Give Seat 3 some distinct actions
    tracker.record_seat_hand(3, vpip=True, pfr=True, post_draw_raised=True, faced_raise_and_folded=False, cards_drawn=1)

    vec = tracker.get_relative_opponent_vector(
        hero_seat=2,
        is_alive=is_alive,
        in_hand=in_hand,
        stacks=stacks,
        investments=investments,
        pot=pot,
        draw_counts_this_hand=draw_counts,
    )

    # 4 opponent slots * 10 features = 40 features
    assert len(vec) == 40

    # Slot +1 corresponds to Seat 3
    # Seat 3: alive=1.0, in_hand=1.0
    assert vec[0] == 1.0
    assert vec[1] == 1.0
    # Slot +2 corresponds to Seat 4: eliminated! All features 0.0
    assert vec[10] == 0.0 # alive
    assert vec[11] == 0.0 # in_hand
    assert all(v == 0.0 for v in vec[10:20])

    print("test_relative_vector: PASS")


if __name__ == "__main__":
    test_prior_stability()
    test_bayesian_updates()
    test_relative_vector()
    print("ALL OPPONENT TRACKER TESTS PASSED!")
