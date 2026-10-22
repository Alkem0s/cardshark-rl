import unittest
import numpy as np

from core.card import parse_card
from games.holdem.action_abstraction import (
    A_FOLD, A_CHECK_CALL, A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN,
    STREET_PREFLOP, STREET_FLOP
)
from games.holdem.evaluator import evaluate_preflop_hand_chen
from games.holdem.env import HoldemPokerEnv
from games.holdem.gym_wrapper import HoldemPokerGymEnv


class TestDominatedMaskingHoldem(unittest.TestCase):
    def test_chen_formula_accuracy(self):
        # Pocket Aces (monster)
        aa = evaluate_preflop_hand_chen(parse_card("As"), parse_card("Ah"))
        self.assertAlmostEqual(aa, 1.0, places=2)

        # 7-2 offsuit (worst hand in poker)
        c_72o = evaluate_preflop_hand_chen(parse_card("7s"), parse_card("2c"))
        self.assertAlmostEqual(c_72o, 0.0, places=2)

        # K-2 offsuit (trash hand: should be <= 0.20, NOT 0.73)
        k2o = evaluate_preflop_hand_chen(parse_card("Ks"), parse_card("2c"))
        self.assertLessEqual(k2o, 0.20)

        # AK suited (strong hand)
        aks = evaluate_preflop_hand_chen(parse_card("As"), parse_card("Ks"))
        self.assertGreaterEqual(aks, 0.60)

    def test_deep_stack_preflop_allin_masked(self):
        env = HoldemPokerEnv(num_seats=5, starting_chips=200, small_blind=1, big_blind=2, dominated_masking=True)
        env.reset_session()
        
        # At start of hand, all players have 200 chips (100 BBs)
        current_seat = env.current_seat
        env.seats[current_seat].hand = [parse_card("As"), parse_card("Ks")]
        legal = env.get_legal_actions(current_seat)
        
        # Preflop deep-stack: A_ALL_IN MUST be masked
        self.assertNotIn(A_ALL_IN, legal, "Deep-stack 100BB open shove should be masked as dominated!")
        
        # Standard raises should be available
        self.assertIn(A_MIN_RAISE, legal)
        self.assertIn(A_HALF_POT, legal)
        self.assertIn(A_POT, legal)
        self.assertIn(A_CHECK_CALL, legal)

    def test_preflop_raise_sizes_are_distinct(self):
        env = HoldemPokerEnv(num_seats=5, starting_chips=200, small_blind=1, big_blind=2, dominated_masking=True)
        env.reset_session()
        
        amt_min = env._calculate_raise_amount(env.current_seat, A_MIN_RAISE)
        amt_half = env._calculate_raise_amount(env.current_seat, A_HALF_POT)
        amt_pot = env._calculate_raise_amount(env.current_seat, A_POT)
        
        self.assertLess(amt_min, amt_half, "Half-pot open must exceed min-raise (no duplicates)")
        self.assertLess(amt_half, amt_pot, "Pot open must exceed half-pot")

    def test_short_stack_permits_shove(self):
        env = HoldemPokerEnv(num_seats=5, starting_chips=200, small_blind=1, big_blind=2, dominated_masking=True)
        env.reset_session()
        
        # Give current player a short stack (15 BBs = 30 chips)
        cur = env.current_seat
        env.seats[cur].chips = 30
        legal = env.get_legal_actions(cur)
        
        # Short stack push-or-fold: A_ALL_IN MUST be permitted
        self.assertIn(A_ALL_IN, legal, "Short stack (<=20 BB) must permit all-in shove!")

    def test_postflop_permits_shove_when_committed_or_short(self):
        env = HoldemPokerEnv(num_seats=5, starting_chips=200, small_blind=1, big_blind=2, dominated_masking=True)
        env.reset_session()
        
        # Advance to flop
        env.street = STREET_FLOP
        env.board = [parse_card("Td"), parse_card("8h"), parse_card("4c")]
        env.pot = 12
        for s in env.seats:
            s.round_invested = 0
            s.bet_to_call = 0
            s.has_acted = False
        env.current_bet = 0
        
        cur = env.current_seat
        legal_deep = env.get_legal_actions(cur)
        # Deep stack 100BB into unraised pot: open shove must be masked
        self.assertNotIn(A_ALL_IN, legal_deep, "Deep-stack open shove into unraised flop pot should be masked!")
        self.assertIn(A_HALF_POT, legal_deep)
        self.assertIn(A_POT, legal_deep)
        
        # When pot-committed (e.g. pot=80, chips=60), all-in is permitted
        env.pot = 80
        env.seats[cur].chips = 60
        legal_committed = env.get_legal_actions(cur)
        self.assertIn(A_ALL_IN, legal_committed, "Pot-committed postflop situation must permit all-in shove!")

    def test_postflop_flop_air_facing_bet_must_fold(self):
        env = HoldemPokerEnv(num_seats=5, starting_chips=200, small_blind=1, big_blind=2, dominated_masking=True)
        env.reset_session()
        env.street = STREET_FLOP
        env.board = [parse_card("Kh"), parse_card("5h"), parse_card("3h")]
        cur = env.current_seat
        # Hero holds Jack-high unsuited air with no draw, no overcards to King
        env.seats[cur].hand = [parse_card("Jd"), parse_card("9s")]
        env.seats[cur].bet_to_call = 10
        legal = env.get_legal_actions(cur)
        self.assertEqual(legal, [A_FOLD], "Pure air on flop facing bet must fold!")

    def test_postflop_turn_air_facing_bet_must_fold(self):
        from games.holdem.action_abstraction import STREET_TURN
        env = HoldemPokerEnv(num_seats=5, starting_chips=200, small_blind=1, big_blind=2, dominated_masking=True)
        env.reset_session()
        env.street = STREET_TURN
        env.board = [parse_card("Kh"), parse_card("5h"), parse_card("3h"), parse_card("Ac")]
        cur = env.current_seat
        # Hero holds 9-high air with no draw on turn
        env.seats[cur].hand = [parse_card("8d"), parse_card("9s")]
        env.seats[cur].bet_to_call = 20
        legal = env.get_legal_actions(cur)
        self.assertEqual(legal, [A_FOLD], "Pure air on turn facing bet must fold!")

    def test_postflop_river_air_facing_bet_must_fold(self):
        from games.holdem.action_abstraction import STREET_RIVER
        env = HoldemPokerEnv(num_seats=5, starting_chips=200, small_blind=1, big_blind=2, dominated_masking=True)
        env.reset_session()
        env.street = STREET_RIVER
        env.board = [parse_card("Kh"), parse_card("5h"), parse_card("3h"), parse_card("Ac"), parse_card("Ts")]
        cur = env.current_seat
        # Hero holds Jack-high (no pair) facing a river bet
        env.seats[cur].hand = [parse_card("Jd"), parse_card("9s")]
        env.seats[cur].bet_to_call = 30
        legal = env.get_legal_actions(cur)
        self.assertEqual(legal, [A_FOLD], "High-card air on river facing bet must fold!")

    def test_postflop_made_hand_or_draw_permits_call(self):
        env = HoldemPokerEnv(num_seats=5, starting_chips=200, small_blind=1, big_blind=2, dominated_masking=True)
        env.reset_session()
        env.street = STREET_FLOP
        env.board = [parse_card("Kh"), parse_card("5h"), parse_card("3h")]
        cur = env.current_seat
        # Top pair King
        env.seats[cur].hand = [parse_card("Kd"), parse_card("9s")]
        env.seats[cur].bet_to_call = 10
        legal = env.get_legal_actions(cur)
        self.assertIn(A_CHECK_CALL, legal, "Made top pair must be allowed to call!")
        self.assertIn(A_FOLD, legal)

    def test_postflop_zero_bet_permits_check(self):
        env = HoldemPokerEnv(num_seats=5, starting_chips=200, small_blind=1, big_blind=2, dominated_masking=True)
        env.reset_session()
        env.street = STREET_FLOP
        env.board = [parse_card("Kh"), parse_card("5h"), parse_card("3h")]
        cur = env.current_seat
        env.seats[cur].hand = [parse_card("Jd"), parse_card("9s")]
        env.seats[cur].bet_to_call = 0
        legal = env.get_legal_actions(cur)
        self.assertIn(A_CHECK_CALL, legal, "Free check must be permitted when bet_to_call == 0!")
        self.assertNotIn(A_FOLD, legal, "Free check cannot fold!")

    def test_preflop_trash_facing_raise_must_fold(self):
        env = HoldemPokerEnv(num_seats=5, starting_chips=200, small_blind=1, big_blind=2, dominated_masking=True)
        env.reset_session()
        cur = env.current_seat
        # 7-2 offsuit (worst hand) facing a raise to 6
        env.seats[cur].hand = [parse_card("7s"), parse_card("2c")]
        env.seats[cur].bet_to_call = 6
        legal = env.get_legal_actions(cur)
        self.assertEqual(legal, [A_FOLD], "Trash 7-2o facing raise must fold!")

    def test_preflop_marginal_facing_3bet_cannot_raise(self):
        env = HoldemPokerEnv(num_seats=5, starting_chips=200, small_blind=1, big_blind=2, dominated_masking=True)
        env.reset_session()
        cur = env.current_seat
        # J-9 offsuit (Chen 0.25) facing a 3-bet to 10
        env.seats[cur].hand = [parse_card("Jd"), parse_card("9s")]
        env.seats[cur].bet_to_call = 10
        legal = env.get_legal_actions(cur)
        self.assertNotIn(A_MIN_RAISE, legal, "Marginal hand cannot 4-bet min-raise into 3-bet!")
        self.assertNotIn(A_ALL_IN, legal)
        self.assertIn(A_FOLD, legal)

    def test_turn_air_checked_to_must_check(self):
        from games.holdem.action_abstraction import STREET_TURN
        env = HoldemPokerEnv(num_seats=5, starting_chips=200, small_blind=1, big_blind=2, dominated_masking=True)
        env.reset_session()
        env.street = STREET_TURN
        env.board = [parse_card("Kh"), parse_card("5h"), parse_card("3h"), parse_card("Ac")]
        cur = env.current_seat
        # 8-high air on turn when checked to
        env.seats[cur].hand = [parse_card("8d"), parse_card("7s")]
        env.seats[cur].bet_to_call = 0
        legal = env.get_legal_actions(cur)
        self.assertEqual(legal, [A_CHECK_CALL], "Turn pure air when checked to must check (cannot bet air)!")


if __name__ == "__main__":
    unittest.main()
