"""
multi_npc.py — Production-Ready NPC Interface for CardShark-RL Model B.

Provides a clean, standalone API to import the trained reinforcement learning agent
into any server, game engine, or web backend without Gym dependencies.
"""

from __future__ import annotations
import os
import numpy as np
from typing import Dict, List, Optional, Tuple

from sb3_contrib import MaskablePPO
from card_utils import (
    normalize_rank, normalize_suit, normalize_hand_score,
    hand_category, evaluate_hand
)
from multi_draw_poker_env import (
    A_FOLD, A_CALL, A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN,
    A_DRAW_START, TOTAL_ACTIONS,
    PHASE_PRE_DRAW, PHASE_DRAW, PHASE_POST_DRAW
)
from opponent_tracker import TableOpponentTracker

OBS_DIM = 63


class CardSharkNPC:
    """Drop-in multi-player NPC player powered by trained Model B."""

    def __init__(
        self,
        model_path: str = "models/model_b_multiplayer.zip",
        num_seats: int = 5,
        npc_seat: int = 0,
    ):
        self.num_seats = num_seats
        self.npc_seat = npc_seat
        self.tracker = TableOpponentTracker(num_seats=num_seats)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        print(f"[CardSharkNPC] Loading trained policy from: {model_path}")
        self.model = MaskablePPO.load(model_path)

    def get_action(
        self,
        npc_hand: List[int],
        pot: int,
        bet_to_call: int,
        phase: str,
        npc_chips: int,
        total_table_chips: int,
        button_seat: int,
        seat_chips: List[int],
        seat_invested: List[int],
        seat_alive: List[bool],
        seat_in_hand: List[bool],
        seat_draw_counts: List[int],
        big_blind: int = 2,
    ) -> dict:
        """Computes the optimal action given current table state.

        Returns:
            dict containing:
                "action_type": "fold" | "call" | "raise" | "all_in" | "draw"
                "raise_amount": integer chips to bet (if raising)
                "discard_indices": list of int indices [0..4] (if draw phase)
                "raw_action_id": integer action index (0..37)
                "inferred_tells": dict of opponent profiles and statistics
        """
        # 1. Build observation vector
        obs = self._build_observation(
            npc_hand=npc_hand,
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
        )

        # 2. Build legal action mask
        mask = self._build_mask(phase, bet_to_call, npc_chips, pot, big_blind)

        # 3. Predict action via neural network
        action_id, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        action_id = int(action_id)

        # 4. Translate action
        action_type = "call"
        raise_amount = 0
        discard_indices = []

        if phase in (PHASE_PRE_DRAW, PHASE_POST_DRAW):
            if action_id == A_FOLD:
                action_type = "fold"
            elif action_id == A_CALL:
                action_type = "call"
            elif action_id == A_MIN_RAISE:
                action_type = "raise"
                raise_amount = min(npc_chips, bet_to_call + big_blind)
            elif action_id == A_HALF_POT:
                action_type = "raise"
                raise_amount = min(npc_chips, bet_to_call + max(big_blind, pot // 2))
            elif action_id == A_POT:
                action_type = "raise"
                raise_amount = min(npc_chips, bet_to_call + max(big_blind, pot))
            elif action_id == A_ALL_IN:
                action_type = "all_in"
                raise_amount = npc_chips
        else: # Draw phase
            action_type = "draw"
            bitmask = action_id - A_DRAW_START
            discard_indices = [i for i in range(5) if bitmask & (1 << i)]

        return {
            "action_type": action_type,
            "raise_amount": raise_amount,
            "discard_indices": discard_indices,
            "raw_action_id": action_id,
            "inferred_tells": self.get_opponent_profiles(),
        }

    def record_hand_conclusion(
        self,
        seat_vpip: List[bool],
        seat_pfr: List[bool],
        seat_post_draw_raised: List[bool],
        seat_faced_raise_and_folded: List[bool],
        seat_draw_counts: List[int],
    ):
        """Updates the Bayesian opponent model at the end of each hand."""
        for s in range(self.num_seats):
            if s != self.npc_seat:
                self.tracker.record_seat_hand(
                    seat_idx=s,
                    vpip=seat_vpip[s],
                    pfr=seat_pfr[s],
                    post_draw_raised=seat_post_draw_raised[s],
                    faced_raise_and_folded=seat_faced_raise_and_folded[s],
                    cards_drawn=seat_draw_counts[s] if seat_draw_counts[s] >= 0 else None,
                )

    def get_opponent_profiles(self) -> Dict[int, dict]:
        """Returns readable behavioral habit statistics for all opponents at the table."""
        profiles = {}
        for s in range(self.num_seats):
            if s != self.npc_seat and s in self.tracker.profiles:
                p = self.tracker.profiles[s]
                arch_name, advice = p.classify_archetype()
                profiles[s] = {
                    "archetype": arch_name,
                    "exploit_advice": advice,
                    "vpip": round(p.vpip_tracker.posterior_mean * 100, 1),
                    "pfr": round(p.pfr_tracker.posterior_mean * 100, 1),
                    "aggression_factor": round(p.af_tracker.posterior_mean * 100, 1),
                    "fold_to_raise": round(p.fold_pressure_tracker.posterior_mean * 100, 1),
                    "avg_discards": round(p.avg_draw_normalized * 5.0, 1),
                    "hands_observed": p.hands_observed,
                }
        return profiles

    def _build_mask(self, phase: str, bet_to_call: int, chips: int, pot: int, big_blind: int) -> np.ndarray:
        mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
        if phase in (PHASE_PRE_DRAW, PHASE_POST_DRAW):
            mask[A_FOLD] = 1
            mask[A_CALL] = 1
            chips_after_call = chips - bet_to_call
            if chips_after_call > 0:
                mask[A_ALL_IN] = 1
                if chips_after_call >= big_blind:
                    mask[A_MIN_RAISE] = 1
                half_pot = max(big_blind, pot // 2)
                if chips_after_call >= half_pot:
                    mask[A_HALF_POT] = 1
                pot_raise = max(big_blind, pot)
                if chips_after_call >= pot_raise:
                    mask[A_POT] = 1
        elif phase == PHASE_DRAW:
            mask[A_DRAW_START:TOTAL_ACTIONS] = 1
        return mask

    def _build_observation(
        self,
        npc_hand: List[int],
        pot: int,
        bet_to_call: int,
        phase: str,
        npc_chips: int,
        total_table_chips: int,
        button_seat: int,
        seat_chips: List[int],
        seat_invested: List[int],
        seat_alive: List[bool],
        seat_in_hand: List[bool],
        seat_draw_counts: List[int],
    ) -> np.ndarray:
        total_chips = max(1.0, float(total_table_chips))
        pot_f = float(pot)

        # 1. Cards (10)
        hand_norm = []
        if npc_hand and len(npc_hand) == 5:
            for c in npc_hand:
                hand_norm.extend([normalize_rank(c), normalize_suit(c)])
        else:
            hand_norm = [0.0] * 10

        # 2. Hand evaluation (2)
        if npc_hand and len(npc_hand) == 5:
            cat_norm = float(hand_category(npc_hand)) / 8.0
            score_norm = normalize_hand_score(evaluate_hand(npc_hand))
        else:
            cat_norm = 0.0
            score_norm = 0.0

        # 3. Pot & Stacks (5)
        stack_ratio = float(npc_chips) / total_chips
        pot_ratio = pot_f / total_chips
        btc_f = float(bet_to_call)
        pot_odds = btc_f / max(1.0, pot_f + btc_f)
        spr = min(float(npc_chips) / max(1.0, pot_f), 10.0) / 10.0
        inv_ratio = float(seat_invested[self.npc_seat]) / max(1.0, pot_f) if seat_invested else 0.0

        # 4. Phase & Button (4)
        phase_vec = [0.0, 0.0, 0.0]
        if phase == PHASE_PRE_DRAW:
            phase_vec[0] = 1.0
        elif phase == PHASE_DRAW:
            phase_vec[1] = 1.0
        elif phase == PHASE_POST_DRAW:
            phase_vec[2] = 1.0

        btn_dist = float((self.npc_seat - button_seat) % self.num_seats) / max(1.0, self.num_seats - 1.0)

        # 5. Table survival (2)
        alive_count = sum(1 for a in seat_alive if a)
        in_hand_count = sum(1 for h in seat_in_hand if h)
        alive_ratio = float(alive_count) / float(self.num_seats)
        in_hand_ratio = float(in_hand_count) / max(1.0, float(alive_count))

        # 6. Opponents (40)
        opp_vector = self.tracker.get_relative_opponent_vector(
            hero_seat=self.npc_seat,
            is_alive=seat_alive,
            in_hand=seat_in_hand,
            stacks=seat_chips,
            investments=seat_invested,
            pot=pot,
            draw_counts_this_hand=seat_draw_counts,
        )

        obs = (
            hand_norm +
            [cat_norm, score_norm] +
            [stack_ratio, pot_ratio, pot_odds, spr, inv_ratio] +
            phase_vec + [btn_dist] +
            [alive_ratio, in_hand_ratio] +
            opp_vector
        )

        return np.array(obs, dtype=np.float32)
