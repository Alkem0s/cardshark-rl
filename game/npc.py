"""
game/npc.py — Production-Ready Bot & NPC Decision Engine for CardShark Poker.

Provides an isolated NPC interface for the playable game:
1. Heuristic Bot Modes ("casual", "tag", "lag", "rock", "maniac") — ZERO PyTorch / ZERO SB3 dependencies.
2. Neural Bot Modes ("normal" -> Model B, "hard" -> Model C, "champion" -> Model D).
   Loads weights lazily so game simulation never crashes if PyTorch/SB3 is missing.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from game.card_utils import (
    normalize_rank, normalize_suit, normalize_hand_score,
    hand_category, evaluate_hand
)
from game.multi_draw_poker_env import (
    A_FOLD, A_CALL, A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN,
    A_DRAW_START, TOTAL_ACTIONS,
    PHASE_PRE_DRAW, PHASE_DRAW, PHASE_POST_DRAW
)
from game.multi_opponents import (
    CallingStation, Maniac, Rock, TAG, LAG, AdversarialExploiter
)

OBS_DIM = 63
SUPERHUMAN_OBS_DIM = 87


def parse_card_to_int(c: Any) -> int:
    """Converts a card represented as int or dict {'rank': 'A', 'suit': '♠'} to integer index 0..51."""
    if isinstance(c, (int, np.integer)):
        return int(c)
    if isinstance(c, dict):
        r_str = str(c.get("rank", "2")).strip().upper()
        s_str = str(c.get("suit", "c")).strip().lower()
        rank_map = {"2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6, "9": 7, "10": 8, "T": 8, "J": 9, "Q": 10, "K": 11, "A": 12}
        suit_map = {"c": 0, "♣": 0, "d": 1, "♦": 1, "h": 2, "♥": 2, "s": 3, "♠": 3}
        r = rank_map.get(r_str, 0)
        s = suit_map.get(s_str, 0)
        return s * 13 + r
    return 0


class CardSharkNPC:
    """Drop-in bot player for 5-Card Draw supporting both Rule-Based and Neural policies."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        difficulty: str = "normal",
        num_seats: int = 5,
        npc_seat: int = 0,
    ):
        self.num_seats = num_seats
        self.npc_seat = npc_seat
        self.difficulty = difficulty.lower()
        self.is_heuristic = self.difficulty in ("casual", "heuristic", "easy", "rock", "maniac", "tag", "lag")
        self.heuristic_bot = None
        self.model = None
        self.tracker = None

        if self.is_heuristic:
            self._init_heuristic()
        else:
            self._init_neural(model_path)

    def _init_heuristic(self):
        """Initializes a pure rule-based bot."""
        if self.difficulty in ("maniac", "aggressive"):
            self.heuristic_bot = Maniac()
        elif self.difficulty in ("rock", "passive", "easy"):
            self.heuristic_bot = Rock()
        elif self.difficulty == "lag":
            self.heuristic_bot = LAG()
        elif self.difficulty == "calling_station":
            self.heuristic_bot = CallingStation()
        else:
            self.heuristic_bot = TAG()
        self.obs_dim = 0
        self.is_superhuman = False
        print(f"[CardSharkNPC] Initialized pure rule-based bot [{self.heuristic_bot.name}].")

    def _init_neural(self, model_path: Optional[str]):
        try:
            import sys
            import rl.card_attention
            sys.modules["card_attention"] = rl.card_attention
            from sb3_contrib import MaskablePPO
            from rl.opponent_tracker import TableOpponentTracker
            from rl.card_attention import CardAttentionExtractor
        except ImportError as e:
            print(f"[CardSharkNPC] Notice: ML dependencies not found ({e}). Falling back to Heuristic TAG bot.")
            self.is_heuristic = True
            self._init_heuristic()
            return

        self.tracker = TableOpponentTracker(num_seats=self.num_seats)

        if model_path is None:
            if self.difficulty in ("champion", "expert", "model_d", "d"):
                model_path = "models/model_d.zip" if os.path.exists("models/model_d.zip") else "models/model_d_champion.zip"
            elif self.difficulty in ("hard", "superhuman", "model_c", "c"):
                model_path = "models/model_c.zip" if os.path.exists("models/model_c.zip") else "models/model_c_superhuman.zip"
            else:
                model_path = "models/model_b.zip" if os.path.exists("models/model_b.zip") else "models/model_b_multiplayer.zip"

        if not os.path.exists(model_path):
            print(f"[CardSharkNPC] Warning: Checkpoint {model_path} not found. Falling back to heuristic bot.")
            self.is_heuristic = True
            self._init_heuristic()
            return

        print(f"[CardSharkNPC] Loading trained policy [{self.difficulty.upper()}] from: {model_path}")
        self.model = MaskablePPO.load(model_path)
        self.obs_dim = self.model.observation_space.shape[0]
        self.is_superhuman = (self.obs_dim == SUPERHUMAN_OBS_DIM)

    def get_action(
        self,
        npc_hand: List[Any],
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
        seat_pre_draw_actions: Optional[List[int]] = None,
        seat_post_draw_actions: Optional[List[int]] = None,
        seat_pre_draw_bets: Optional[List[int]] = None,
        seat_post_draw_bets: Optional[List[int]] = None,
    ) -> dict:
        # Normalize hand cards to integer indices 0..51
        npc_hand = [parse_card_to_int(c) for c in npc_hand]
        if len(npc_hand) != 5:
            return {"action_type": "call", "raise_amount": 0, "discard_indices": [], "raw_action_id": 1}

        # 1. Handle rule-based fallback
        if self.is_heuristic or self.model is None:
            return self._get_heuristic_action(
                npc_hand=npc_hand,
                pot=pot,
                bet_to_call=bet_to_call,
                phase=phase,
                npc_chips=npc_chips,
                big_blind=big_blind,
            )

        # 2. Build observation vector & mask for neural policy
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
            seat_pre_draw_actions=seat_pre_draw_actions,
            seat_post_draw_actions=seat_post_draw_actions,
            seat_pre_draw_bets=seat_pre_draw_bets,
            seat_post_draw_bets=seat_post_draw_bets,
        )

        mask = self._build_mask(phase, bet_to_call, npc_chips, pot, big_blind)
        action_id, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        action_id = int(action_id)

        return self._translate_action(action_id, phase, pot, bet_to_call, npc_chips, big_blind)

    def _get_heuristic_action(self, npc_hand, pot, bet_to_call, phase, npc_chips, big_blind):
        """Translates heuristic bot decision to game server response format."""
        if phase in (PHASE_PRE_DRAW, PHASE_POST_DRAW):
            legal = [A_FOLD, A_CALL]
            if npc_chips > bet_to_call:
                legal.extend([A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN])
            act_id = self.heuristic_bot.bet_action(
                hand=npc_hand,
                pot=pot,
                bet_to_call=bet_to_call,
                phase=phase,
                stack=npc_chips,
                legal_actions=legal,
            )
            return self._translate_action(act_id, phase, pot, bet_to_call, npc_chips, big_blind)
        else:
            discards = self.heuristic_bot.draw_action(npc_hand)
            return {
                "action_type": "draw",
                "raise_amount": 0,
                "discard_indices": discards,
                "raw_action_id": A_DRAW_START,
                "inferred_tells": {},
            }

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
        seat_pre_draw_actions: Optional[List[int]] = None,
        seat_post_draw_actions: Optional[List[int]] = None,
        seat_pre_draw_bets: Optional[List[int]] = None,
        seat_post_draw_bets: Optional[List[int]] = None,
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

        # 6. Opponents (40: 4 seats x 10 features via TableOpponentTracker)
        if self.tracker is not None:
            opp_vector = self.tracker.get_relative_opponent_vector(
                hero_seat=self.npc_seat,
                is_alive=seat_alive,
                in_hand=seat_in_hand,
                stacks=seat_chips,
                investments=seat_invested,
                pot=pot,
                draw_counts_this_hand=seat_draw_counts,
            )
        else:
            opp_vector = [0.0] * 40

        obs = (
            hand_norm +
            [cat_norm, score_norm] +
            [stack_ratio, pot_ratio, pot_odds, spr, inv_ratio] +
            phase_vec + [btn_dist] +
            [alive_ratio, in_hand_ratio] +
            opp_vector
        )

        if not self.is_superhuman:
            return np.array(obs[:OBS_DIM], dtype=np.float32)

        # 7. Model C / Model D Superhuman Sequence & Dynamic Recency Memory (24: 4 slots x 6 features)
        act_map = {
            -1: 0.0,
            A_FOLD: -1.0,
            A_CALL: 0.2,
            A_MIN_RAISE: 0.5,
            A_HALF_POT: 0.7,
            A_POT: 0.9,
            A_ALL_IN: 1.0,
        }

        superhuman_features: List[float] = []
        for offset in range(1, self.num_seats):
            s = (self.npc_seat + offset) % self.num_seats
            alive = s < len(seat_alive) and seat_alive[s]

            if alive:
                pre_act_raw = seat_pre_draw_actions[s] if seat_pre_draw_actions and s < len(seat_pre_draw_actions) else -1
                post_act_raw = seat_post_draw_actions[s] if seat_post_draw_actions and s < len(seat_post_draw_actions) else -1
                pre_bet = seat_pre_draw_bets[s] if seat_pre_draw_bets and s < len(seat_pre_draw_bets) else 0
                post_bet = seat_post_draw_bets[s] if seat_post_draw_bets and s < len(seat_post_draw_bets) else 0
                draw_count = seat_draw_counts[s] if seat_draw_counts and s < len(seat_draw_counts) else -1
                is_folded = s < len(seat_in_hand) and not seat_in_hand[s]

                pre_act = act_map.get(pre_act_raw, 0.0)
                pre_sizing = float(np.clip(pre_bet / total_chips, 0.0, 1.0))
                post_act = act_map.get(post_act_raw, 0.0)
                post_sizing = float(np.clip(post_bet / total_chips, 0.0, 1.0))

                line_code = 0.0
                if is_folded:
                    line_code = -1.0
                elif post_act_raw in (A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN):
                    if draw_count == 0:
                        line_code = 0.8
                    elif pre_act_raw in (-1, A_CALL) and draw_count in (1, 2):
                        line_code = 1.0
                    elif pre_act_raw in (A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN) and draw_count >= 2:
                        line_code = -0.5
                    else:
                        line_code = 0.6
                elif post_act_raw == A_CALL:
                    line_code = 0.1

                tilt_delta = self.tracker.get_seat_recency_delta(s) if self.tracker else 0.0
                slot_seq = [pre_act, pre_sizing, post_act, post_sizing, line_code, tilt_delta]
            else:
                slot_seq = [0.0] * 6

            superhuman_features.extend(slot_seq)

        obs = obs + superhuman_features
        return np.array(obs[:SUPERHUMAN_OBS_DIM], dtype=np.float32)

    def _translate_action(self, action_id: int, phase: str, pot: int, bet_to_call: int, npc_chips: int, big_blind: int) -> dict:
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
        else:
            action_type = "draw"
            bitmask = max(0, action_id - A_DRAW_START)
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
        if self.tracker is not None:
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
        if self.tracker is None:
            return profiles
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
