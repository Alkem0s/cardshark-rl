"""
games/holdem/gym_wrapper.py — Gymnasium Wrapper for No-Limit Texas Hold'em.

Features:
1. 73-dimensional bounded [-1.0, 1.0] observation vector.
2. 6-action discrete betting space (Fold, Check/Call, Min-Raise, Half-Pot, Pot, All-In).
3. ActionMasker integration for valid betting options.
4. Autonomous table stepping for heuristic opponents.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from core.card import rank_of, suit_of, normalize_rank, normalize_suit
from games.holdem.action_abstraction import (
    A_FOLD,
    A_CHECK_CALL,
    A_MIN_RAISE,
    A_HALF_POT,
    A_POT,
    A_ALL_IN,
    NUM_ACTIONS,
    A_F_FOLD,
    A_F_CHECK_CALL,
    A_F_MIN_RAISE,
    A_F_THIRD_POT,
    A_F_HALF_POT,
    A_F_POT,
    A_F_OVERBET,
    A_F_ALL_IN,
    NUM_ACTIONS_HOLDEM_F,
    STREET_PREFLOP,
    STREET_FLOP,
    STREET_TURN,
    STREET_RIVER,
    STREET_SHOWDOWN,
    STREETS,
)
from games.holdem.evaluator import evaluate_holdem_hand, evaluate_preflop_hand_chen
from games.holdem.env import HoldemPokerEnv
from games.holdem.opponents import HoldemOpponent, make_random_holdem_opponent
from games.holdem.board_texture import evaluate_board_texture_and_draws, NUM_TEXTURE_FEATURES
from games.holdem.opponent_tracker import HoldemTableOpponentTracker, NUM_OPPONENT_FEATURES

OBS_DIM_HOLDEM_A = 73
OBS_DIM_HOLDEM_B = 125
OBS_DIM_HOLDEM_C = 138
OBS_DIM_HOLDEM_D = 152
OBS_DIM_HOLDEM_E = 158
OBS_DIM_HOLDEM_F = 168
OBS_DIM_HOLDEM = OBS_DIM_HOLDEM_A


def build_observation_vector_holdem(
    env: HoldemPokerEnv,
    hero_seat: int,
    obs_version: str = "b",
    tracker: Optional[HoldemTableOpponentTracker] = None,
) -> np.ndarray:
    v = obs_version.lower()
    if v == "a":
        target_dim = OBS_DIM_HOLDEM_A
    elif v == "b":
        target_dim = OBS_DIM_HOLDEM_B
    elif v == "c":
        target_dim = OBS_DIM_HOLDEM_C
    elif v == "d":
        target_dim = OBS_DIM_HOLDEM_D
    elif v == "e":
        target_dim = OBS_DIM_HOLDEM_E
    elif v == "f":
        target_dim = OBS_DIM_HOLDEM_F
    else:
        target_dim = OBS_DIM_HOLDEM_B
    obs = np.zeros(target_dim, dtype=np.float32)
    idx = 0
    total_chips = max(1, env.num_seats * env.starting_chips)
    hero = env.seats[hero_seat]

    # 1. Hero Hole Cards (2 cards -> 4 floats)
    for c_i in range(2):
        if c_i < len(hero.hand) and hero.hand[c_i] != -1:
            c = hero.hand[c_i]
            obs[idx] = normalize_rank(rank_of(c))
            obs[idx + 1] = normalize_suit(suit_of(c))
        else:
            obs[idx] = -1.0
            obs[idx + 1] = -1.0
        idx += 2

    # 2. Community Board Cards (5 cards -> 10 floats)
    for b_i in range(5):
        if b_i < len(env.board) and env.board[b_i] != -1:
            c = env.board[b_i]
            obs[idx] = normalize_rank(rank_of(c))
            obs[idx + 1] = normalize_suit(suit_of(c))
        else:
            obs[idx] = -1.0
            obs[idx + 1] = -1.0
        idx += 2

    # 3. Current Hand Strength (1 float)
    if len(hero.hand) == 2 and len(env.board) >= 3:
        score, _ = evaluate_holdem_hand(hero.hand + env.board)
        strength = min(1.0, score / 8_000_000.0)
        obs[idx] = (strength * 2.0) - 1.0
    elif len(hero.hand) == 2:
        preflop_val = evaluate_preflop_hand_chen(hero.hand[0], hero.hand[1])
        obs[idx] = (min(1.0, preflop_val) * 2.0) - 1.0
    else:
        obs[idx] = -1.0
    idx += 1

    # 4. Pot Odds (1 float)
    pot_odds = hero.bet_to_call / max(1.0, float(env.pot + hero.bet_to_call))
    obs[idx] = (min(1.0, pot_odds) * 2.0) - 1.0
    idx += 1

    # 5. Pot Size normalized (1 float)
    obs[idx] = (min(1.0, env.pot / float(total_chips)) * 2.0) - 1.0
    idx += 1

    # 6. Hero Stack normalized (1 float)
    obs[idx] = (min(1.0, hero.chips / float(total_chips)) * 2.0) - 1.0
    idx += 1

    # 7. Hero Bet to Call normalized (1 float)
    b2c_ratio = hero.bet_to_call / max(1.0, float(hero.chips))
    obs[idx] = (min(1.0, b2c_ratio) * 2.0) - 1.0
    idx += 1

    # 8. Street Indicator (4 one-hot floats)
    street_map = {STREET_PREFLOP: 0, STREET_FLOP: 1, STREET_TURN: 2, STREET_RIVER: 3}
    curr_s_idx = street_map.get(env.street, 0)
    for s_i in range(4):
        obs[idx + s_i] = 1.0 if s_i == curr_s_idx else -1.0
    idx += 4

    # 9. Hero Position relative to Button (5 one-hot floats)
    pos_offset = (hero_seat - env.dealer_seat) % env.num_seats
    for p_i in range(env.num_seats):
        obs[idx + p_i] = 1.0 if p_i == pos_offset else -1.0
    idx += env.num_seats

    # 10. Per-Seat Status (5 seats * 5 features = 25 floats)
    for s in env.seats:
        obs[idx] = (min(1.0, s.chips / float(total_chips)) * 2.0) - 1.0
        obs[idx + 1] = (min(1.0, s.round_invested / max(1.0, float(env.pot))) * 2.0) - 1.0
        obs[idx + 2] = 1.0 if s.is_alive else -1.0
        obs[idx + 3] = 1.0 if s.folded else -1.0
        obs[idx + 4] = 1.0 if s.is_all_in else -1.0
        idx += 5

    # 11. Street Action History (4 streets * 5 seats = 20 floats)
    for s_name in [STREET_PREFLOP, STREET_FLOP, STREET_TURN, STREET_RIVER]:
        for s in env.seats:
            act = s.street_actions.get(s_name, -1)
            if act == -1:
                obs[idx] = -1.0
            else:
                denom = max(1.0, float(env.num_actions - 1)) / 2.0
                obs[idx] = float(np.clip((act / denom) - 1.0, -1.0, 1.0))
            idx += 1

    # If Model A, return base 73 dimensions
    if target_dim == OBS_DIM_HOLDEM_A:
        return obs

    # --- MODEL B AUGMENTATIONS (52 additional features -> 125 total) ---

    # 12. ICM Stack Geometry (3 floats)
    live_chips = sum(s.chips for s in env.seats if s.is_alive)
    hero_share = hero.chips / max(1.0, float(live_chips))
    obs[idx] = float((min(1.0, hero_share) * 2.0) - 1.0)
    idx += 1

    other_active = [s.chips for s in env.seats if s.seat_idx != hero_seat and s.is_alive and not s.folded]
    max_other = max(other_active) if other_active else 1
    eff_ratio = hero.chips / max(1.0, float(max_other))
    obs[idx] = float(np.clip(eff_ratio - 1.0, -1.0, 1.0))
    idx += 1

    min_live = min([s.chips for s in env.seats if s.is_alive] or [1])
    short_ratio = min_live / max(1.0, float(hero.chips))
    obs[idx] = float((min(1.0, short_ratio) * 2.0) - 1.0)
    idx += 1

    # 13. Board Texture & Draws Engine (17 floats)
    texture_feats = evaluate_board_texture_and_draws(hero.hand, env.board)
    obs[idx : idx + NUM_TEXTURE_FEATURES] = texture_feats
    idx += NUM_TEXTURE_FEATURES

    # 14. Dual-Timescale Bayesian Opponent Tracking (32 floats)
    if tracker is not None:
        opp_vec = tracker.get_relative_vector(hero_seat, env.seats, total_chips)
        obs[idx : idx + NUM_OPPONENT_FEATURES] = opp_vec
    else:
        obs[idx : idx + NUM_OPPONENT_FEATURES] = -1.0
    idx += NUM_OPPONENT_FEATURES

    if target_dim == OBS_DIM_HOLDEM_B:
        return obs

    # --- MODEL C AUGMENTATIONS (13 additional features -> 138 total) ---

    # 15. Preflop GTO Positional Features (6 floats)
    from games.holdem.gto_ranges import extract_preflop_gto_features
    c1 = hero.hand[0] if len(hero.hand) > 0 else -1
    c2 = hero.hand[1] if len(hero.hand) > 1 else -1
    pos_offset = env.get_seat_position_from_dealer(hero_seat)
    num_callers = sum(
        1 for s in env.seats
        if s.seat_idx != hero_seat and s.round_invested >= env.big_blind and not s.folded
    )
    gto_feats = extract_preflop_gto_features(
        c1, c2,
        pos_offset=pos_offset,
        num_callers=num_callers,
        bet_to_call=hero.bet_to_call,
        big_blind=env.big_blind,
    )
    obs[idx : idx + 6] = gto_feats
    idx += 6

    # 16. Postflop Pot Control & SPR Metrics (4 floats)
    # 1. Stack-to-Pot Ratio (SPR)
    spr = hero.chips / max(1.0, float(env.pot))
    obs[idx] = float(np.clip((spr / 10.0) - 1.0, -1.0, 1.0))
    idx += 1

    # 2. Showdown Value Classification
    if len(env.board) >= 3 and len(hero.hand) == 2:
        if any(texture_feats[k] > 0 for k in (5, 8, 9, 10)):
            sdv = 1.0
        elif texture_feats[6] > 0:
            sdv = 0.5
        elif texture_feats[7] > 0:
            sdv = 0.0
        elif any(texture_feats[k] > 0 for k in (11, 12, 13, 14)):
            sdv = -0.5
        else:
            sdv = -1.0
    else:
        sdv = -1.0
    obs[idx] = float(sdv)
    idx += 1

    # 3. In-Position Advantage Postflop
    active_others = [s.seat_idx for s in env.seats if s.seat_idx != hero_seat and s.is_alive and not s.folded]
    if active_others:
        hero_postflop_pos = (hero_seat - env.dealer_seat) % env.num_seats
        others_postflop_pos = [(s_i - env.dealer_seat) % env.num_seats for s_i in active_others]
        in_pos = hero_postflop_pos > max(others_postflop_pos)
        obs[idx] = 1.0 if in_pos else -1.0
    else:
        obs[idx] = 1.0
    idx += 1

    # 4. Range Advantage / Aggressor Flag
    last_raiser = None
    for st in [STREET_RIVER, STREET_TURN, STREET_FLOP, STREET_PREFLOP]:
        raisers = [s.seat_idx for s in env.seats if s.street_actions.get(st, -1) in (A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN)]
        if raisers:
            last_raiser = raisers[-1]
            break
    is_aggressor = (last_raiser == hero_seat)
    obs[idx] = 1.0 if is_aggressor else -1.0
    idx += 1

    # 17. Non-linear ICM Tournament Equities (3 floats)
    from games.holdem.icm_engine import compute_icm_features
    seats_chips = [s.chips for s in env.seats]
    icm_feats = compute_icm_features(
        hero_seat=hero_seat,
        seats_chips=seats_chips,
        bet_to_call=hero.bet_to_call,
        pot=env.pot,
    )
    obs[idx : idx + 3] = icm_feats
    idx += 3

    assert idx == OBS_DIM_HOLDEM_C, f"Expected {OBS_DIM_HOLDEM_C} features, got {idx}"
    if target_dim == OBS_DIM_HOLDEM_C:
        return obs

    # --- MODEL D AUGMENTATIONS (14 targeted exploit features -> 152 total) ---
    from games.holdem.opponent_exploiter import build_exploit_features, NUM_EXPLOIT_FEATURES
    exploit_feats = build_exploit_features(env, hero_seat, tracker=tracker)
    obs[idx : idx + NUM_EXPLOIT_FEATURES] = exploit_feats
    idx += NUM_EXPLOIT_FEATURES

    assert idx == OBS_DIM_HOLDEM_D, f"Expected {OBS_DIM_HOLDEM_D} features, got {idx}"
    if target_dim == OBS_DIM_HOLDEM_D:
        return obs

    # --- MODEL E AUGMENTATIONS (6 self-play & tournament dynamics features -> 158 total) ---
    # 1. Table 3-bet / Squeeze Intensity (-1.0 to 1.0)
    preflop_raises = sum(
        1 for s in env.seats
        if s.street_actions.get(STREET_PREFLOP, -1) in (A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN)
    )
    three_bet_val = min(1.0, max(0.0, (preflop_raises - 1.0) / 2.0))
    obs[idx] = (three_bet_val * 2.0) - 1.0
    idx += 1

    # 2. Table Aggression / Check-Raise Volatility (-1.0 to 1.0)
    had_check_raise = False
    for st in (STREET_FLOP, STREET_TURN, STREET_RIVER):
        street_acts = [s.street_actions.get(st, -1) for s in env.seats]
        if A_CHECK_CALL in street_acts and any(a in (A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN) for a in street_acts):
            had_check_raise = True
            break
    obs[idx] = 1.0 if had_check_raise else -1.0
    idx += 1

    # 3. Effective Stack Depth in Big Blinds (-1.0 to 1.0)
    alive_chips = [s.chips for s in env.seats if s.chips > 0 and not s.folded]
    if not alive_chips:
        alive_chips = [s.chips for s in env.seats if s.chips > 0]
    eff_chips = min(alive_chips) if alive_chips else hero.chips
    eff_bb = eff_chips / max(1.0, float(env.big_blind))
    obs[idx] = float(np.clip((eff_bb / 25.0) - 1.0, -1.0, 1.0))
    idx += 1

    # 4. Tournament Push/Fold Regime Flag (-1.0 to 1.0)
    is_push_fold = (eff_bb <= 15.0)
    obs[idx] = 1.0 if is_push_fold else -1.0
    idx += 1

    # 5. Multiway Pot Indicator (-1.0 to 1.0)
    active_in_hand = sum(1 for s in env.seats if s.chips > 0 and not s.folded)
    multiway_norm = (active_in_hand - 2.0) / max(1.0, float(env.num_seats - 2))
    obs[idx] = float(np.clip((multiway_norm * 2.0) - 1.0, -1.0, 1.0))
    idx += 1

    # 6. Primary Adversary Neural Sparring Flag (-1.0 to 1.0)
    from games.holdem.opponent_exploiter import get_primary_adversary_seat
    adv_seat = get_primary_adversary_seat(env, hero_seat)
    adv_opp = env.opponents.get(adv_seat) if hasattr(env, "opponents") and env.opponents else None
    is_neural = False
    if adv_opp is not None:
        if getattr(adv_opp, "is_neural", False):
            is_neural = True
        elif getattr(adv_opp, "model", None) is not None:
            is_neural = True
    obs[idx] = 1.0 if is_neural else -1.0
    idx += 1

    assert idx == OBS_DIM_HOLDEM_E, f"Expected {OBS_DIM_HOLDEM_E} features, got {idx}"
    if target_dim == OBS_DIM_HOLDEM_E:
        return obs

    # --- MODEL F AUGMENTATIONS (10 tournament closer & endgame features -> 168 total) ---
    # 1. Hands Remaining in Blind Level Ratio (-1.0 to 1.0)
    interval = max(1, env.blind_escalation_interval)
    hands_left_in_level = interval - (env.hands_played % interval)
    obs[idx] = float((hands_left_in_level / float(interval)) * 2.0 - 1.0)
    idx += 1

    # 2. Blind Level Urgency (-1.0 to 1.0)
    level = env.hands_played // interval
    obs[idx] = float(np.clip((level / 6.0) * 2.0 - 1.0, -1.0, 1.0))
    idx += 1

    # 3. Hero Chip Leaderboard Rank (-1.0 to 1.0)
    alive_players = [s for s in env.seats if s.is_alive and s.chips > 0]
    sorted_alive = sorted(alive_players, key=lambda s: s.chips, reverse=True)
    rank = 0
    for r_i, s in enumerate(sorted_alive):
        if s.seat_idx == hero_seat:
            rank = r_i
            break
    denom = max(1, len(sorted_alive) - 1)
    obs[idx] = float(1.0 - 2.0 * (rank / float(denom)))
    idx += 1

    # 4. Harrington M-factor (-1.0 to 1.0)
    round_cost = env.small_blind + env.big_blind
    m_factor = hero.chips / max(1.0, float(round_cost))
    obs[idx] = float(np.clip((m_factor / 15.0) - 1.0, -1.0, 1.0))
    idx += 1

    # 5. Harrington M-Zone (-1.0 to 1.0)
    if m_factor < 5.0:
        zone = -1.0  # Red Zone: push/fold critical
    elif m_factor < 10.0:
        zone = -0.33  # Orange Zone: resteal/shove
    elif m_factor < 20.0:
        zone = 0.33  # Yellow Zone: cautious play
    else:
        zone = 1.0  # Green Zone: deep play
    obs[idx] = float(zone)
    idx += 1

    # 6. Heads-Up Endgame Regime Flag (-1.0 to 1.0)
    num_alive = len(alive_players)
    obs[idx] = 1.0 if num_alive == 2 else -1.0
    idx += 1

    # 7. Tournament Podium / Bubble Pressure Flag (-1.0 to 1.0)
    obs[idx] = 1.0 if num_alive == 3 else -1.0
    idx += 1

    # 8. Sklansky-Chubukov Push/Fold Urgency (-1.0 to 1.0)
    if env.street == STREET_PREFLOP and m_factor < 10.0 and len(hero.hand) == 2:
        chen = evaluate_preflop_hand_chen(hero.hand[0], hero.hand[1])
        push_urg = np.clip((chen / 0.40) * 2.0 - 1.0, -1.0, 1.0)
        obs[idx] = float(push_urg)
    else:
        obs[idx] = -1.0
    idx += 1

    # 9. Fold-to-Cbet Steal Opportunity (-1.0 to 1.0)
    adv_fold_pressure = 0.5
    if tracker is not None:
        from games.holdem.opponent_exploiter import get_primary_adversary_seat
        adv_seat = get_primary_adversary_seat(env, hero_seat)
        adv_prof = tracker.profiles.get(adv_seat)
        if adv_prof is not None:
            adv_fold_pressure = adv_prof.fold_pressure_tracker.posterior_mean
    is_cbet_opp = (len(env.board) >= 3 and is_aggressor and adv_fold_pressure > 0.55)
    obs[idx] = 1.0 if is_cbet_opp else -1.0
    idx += 1

    # 10. Closer Chip Dominance Ratio (-1.0 to 1.0)
    opp_chips = [s.chips for s in env.seats if s.seat_idx != hero_seat and s.is_alive and s.chips > 0]
    second_stack = sorted(opp_chips, reverse=True)[0] if opp_chips else 1
    dom_ratio = hero.chips / max(1.0, float(second_stack))
    obs[idx] = float(np.clip(dom_ratio - 1.0, -1.0, 1.0))
    idx += 1

    assert idx == OBS_DIM_HOLDEM_F, f"Expected {OBS_DIM_HOLDEM_F} features, got {idx}"
    return obs


class HoldemPokerGymEnv(gym.Env):
    """
    Gymnasium environment for No-Limit Texas Hold'em.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_seats: int = 5,
        starting_chips: int = 200,
        small_blind: int = 1,
        big_blind: int = 2,
        hero_seat: int = 0,
        max_hands_per_session: int = 150,
        blind_escalation_interval: int = 15,
        randomize_stacks: bool = False,
        fold_penalty: float = 0.0,
        elimination_penalty: float = 0.0,
        win_bonus: float = 0.0,
        dominated_masking: bool = True,
        pos_gto_masking: Optional[bool] = None,
        exploitative_masking: Optional[bool] = None,
        obs_version: str = "a",
        icm_equity_weight: float = 0.0,
        pot_control_bonus: float = 0.0,
        exploit_value_bonus: float = 0.0,
        exploit_steal_bonus: float = 0.0,
        bluff_station_penalty: float = 0.0,
        uncontested_pot_bonus: float = 0.0,
        league_mode: bool = False,
        model_b_path: str = "models/holdem/model_holdem_b.zip",
        rng_seed: Optional[int] = None,
        curriculum_shaping_mult: float = 1.0,
        league_pool: Optional[Any] = None,
        heads_up_mode: bool = False,
    ):
        super().__init__()
        self.hero_seat = hero_seat
        self.num_seats = num_seats
        self.starting_chips = starting_chips
        self.randomize_stacks = randomize_stacks
        self.fold_penalty = fold_penalty
        self.elimination_penalty = elimination_penalty
        self.win_bonus = win_bonus
        self.dominated_masking = dominated_masking
        self.curriculum_shaping_mult = float(curriculum_shaping_mult)
        self.league_pool = league_pool
        self.heads_up_mode = bool(heads_up_mode)
        self.league_progress: float = 0.0
        self.obs_version = obs_version.lower()
        if self.obs_version == "a":
            self.obs_dim = OBS_DIM_HOLDEM_A
        elif self.obs_version == "b":
            self.obs_dim = OBS_DIM_HOLDEM_B
        elif self.obs_version == "c":
            self.obs_dim = OBS_DIM_HOLDEM_C
        elif self.obs_version == "d":
            self.obs_dim = OBS_DIM_HOLDEM_D
        elif self.obs_version == "e":
            self.obs_dim = OBS_DIM_HOLDEM_E
        elif self.obs_version == "f":
            self.obs_dim = OBS_DIM_HOLDEM_F
        else:
            self.obs_dim = OBS_DIM_HOLDEM_B

        if pos_gto_masking is None:
            self.pos_gto_masking = (self.obs_version in ("c", "d", "e", "f"))
        else:
            self.pos_gto_masking = pos_gto_masking

        if exploitative_masking is None:
            self.exploitative_masking = (self.obs_version in ("d", "e", "f"))
        else:
            self.exploitative_masking = exploitative_masking

        self.icm_equity_weight = icm_equity_weight
        self.pot_control_bonus = pot_control_bonus
        self.exploit_value_bonus = exploit_value_bonus
        self.exploit_steal_bonus = exploit_steal_bonus
        self.bluff_station_penalty = bluff_station_penalty
        if self.obs_version == "f" and uncontested_pot_bonus == 0.0:
            self.uncontested_pot_bonus = 0.015
        else:
            self.uncontested_pot_bonus = uncontested_pot_bonus
        self.num_actions = NUM_ACTIONS_HOLDEM_F if self.obs_version == "f" else NUM_ACTIONS
        self.league_mode = league_mode
        self.model_b_path = model_b_path
        self.rng = np.random.default_rng(rng_seed)

        self.tracker = HoldemTableOpponentTracker(num_seats=num_seats)
        self.prev_hands_played = 0

        self.opponents: Dict[int, HoldemOpponent] = {}
        self._init_opponents()

        self.env = HoldemPokerEnv(
            num_seats=num_seats,
            starting_chips=starting_chips,
            small_blind=small_blind,
            big_blind=big_blind,
            max_hands_per_session=max_hands_per_session,
            blind_escalation_interval=blind_escalation_interval,
            dominated_masking=dominated_masking,
            pos_gto_masking=self.pos_gto_masking,
            exploitative_masking=self.exploitative_masking,
            num_actions=self.num_actions,
            uncap_blinds=(self.obs_version == "f"),
            unshackle_bluffs=(self.obs_version == "f"),
            tracker=self.tracker,
            opponents=self.opponents,
            rng=self.rng,
            heads_up_mode=self.heads_up_mode,
        )

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

    def _init_opponents(self):
        for s in range(self.num_seats):
            if s != self.hero_seat:
                if self.heads_up_mode and s > 1:
                    continue
                if self.league_pool is not None:
                    self.opponents[s] = self.league_pool.sample_opponent(
                        s, rng=self.rng, progress=self.league_progress
                    )
                else:
                    self.opponents[s] = make_random_holdem_opponent(
                        opp_id=s,
                        rng=self.rng,
                        league_mode=self.league_mode,
                        model_b_path=self.model_b_path,
                    )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.env.rng = self.rng
            self.env.deck.rng = self.rng
            self._init_opponents()
        elif self.league_pool is not None:
            # Refresh opponents according to updated curriculum progress on session resets
            self._init_opponents()

        self.tracker.reset_all()
        self.prev_hands_played = 0
        self.env.tracker = self.tracker
        self.env.opponents = self.opponents
        self.env.reset_session(randomize_stacks=self.randomize_stacks)
        self._step_until_hero_turn()

        obs = build_observation_vector_holdem(
            self.env, self.hero_seat, obs_version=self.obs_version, tracker=self.tracker
        )
        info = self._get_info()
        return obs, info

    def action_masks(self) -> np.ndarray:
        return self.env.get_action_mask(self.hero_seat)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        p = self.env.seats[self.hero_seat]
        prev_chips = p.chips
        prev_eq = 0.0
        if self.icm_equity_weight > 0.0:
            from games.holdem.icm_engine import calculate_icm_equities
            prev_eq = calculate_icm_equities([s.chips for s in self.env.seats])[self.hero_seat]

        self.env.step_action(self.hero_seat, action)
        self._step_until_hero_turn()

        # Update Bayesian opponent tracking if a hand concluded
        if self.env.hands_played > self.prev_hands_played:
            for s in self.env.seats:
                preflop_act = s.street_actions.get(STREET_PREFLOP, -1)
                vpip = False
                pfr = False
                if preflop_act in (A_CHECK_CALL, A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN):
                    if s.total_invested > self.env.big_blind or preflop_act != A_CHECK_CALL:
                        vpip = True
                if preflop_act in (A_MIN_RAISE, A_HALF_POT, A_POT, A_ALL_IN, A_F_MIN_RAISE, A_F_THIRD_POT, A_F_HALF_POT, A_F_POT, A_F_OVERBET, A_F_ALL_IN):
                    pfr = True
                postflop_bet = any(s.street_actions.get(st, -1) >= 2 for st in (STREET_FLOP, STREET_TURN, STREET_RIVER))
                faced_bet_and_folded = s.folded and any(s.street_actions.get(st, -1) == A_FOLD for st in (STREET_PREFLOP, STREET_FLOP, STREET_TURN, STREET_RIVER))
                self.tracker.record_seat_hand(
                    s.seat_idx,
                    vpip=vpip,
                    pfr=pfr,
                    postflop_bet=postflop_bet,
                    faced_bet_and_folded=faced_bet_and_folded,
                )
            self.prev_hands_played = self.env.hands_played

        new_chips = p.chips
        chip_delta = new_chips - prev_chips

        reward = float(chip_delta) / float(self.env.big_blind)
        if action == A_FOLD:
            reward -= self.fold_penalty

        # Pot-control bonus: checking with medium showdown value on flop/turn
        if self.pot_control_bonus > 0.0 and action == A_CHECK_CALL:
            if self.env.street in (STREET_FLOP, STREET_TURN) and len(self.env.board) >= 3 and len(p.hand) == 2:
                tex = evaluate_board_texture_and_draws(p.hand, self.env.board)
                if tex[6] > 0 or tex[7] > 0:
                    reward += self.pot_control_bonus

        c_mult = self.curriculum_shaping_mult

        # Exploit reward shaping for Model D & E scaled by curriculum_shaping_mult
        if self.bluff_station_penalty > 0.0 and action >= 2:
            if self.env.street in (STREET_TURN, STREET_RIVER) and len(self.env.board) >= 3 and len(p.hand) == 2:
                tex = evaluate_board_texture_and_draws(p.hand, self.env.board)
                has_made = any(tex[k] > 0 for k in range(5, 11))
                has_draw = any(tex[k] > 0 for k in range(11, 15))
                if not has_made and not has_draw:
                    from games.holdem.opponent_exploiter import is_primary_adversary_archetype, ARCHETYPE_STATION
                    if is_primary_adversary_archetype(self.env, self.hero_seat, ARCHETYPE_STATION):
                        reward -= self.bluff_station_penalty * c_mult

        if self.exploit_value_bonus > 0.0 and action >= 2:
            if len(self.env.board) >= 3 and len(p.hand) == 2:
                tex = evaluate_board_texture_and_draws(p.hand, self.env.board)
                has_val = any(tex[k] > 0 for k in (5, 6, 8, 9, 10))
                if has_val:
                    from games.holdem.opponent_exploiter import is_primary_adversary_archetype, ARCHETYPE_STATION
                    if is_primary_adversary_archetype(self.env, self.hero_seat, ARCHETYPE_STATION):
                        reward += self.exploit_value_bonus * c_mult

        if self.exploit_steal_bonus > 0.0 and chip_delta > 0:
            if len(self.env.board) == 0:
                from games.holdem.opponent_exploiter import is_primary_adversary_archetype, ARCHETYPE_TAG
                if is_primary_adversary_archetype(self.env, self.hero_seat, ARCHETYPE_TAG):
                    reward += self.exploit_steal_bonus * c_mult

        # Uncontested pot steal / showdown-avoidance bonus
        if self.uncontested_pot_bonus > 0.0 and self.env.last_won_uncontested and self.env.last_winner_seat == self.hero_seat:
            reward += self.uncontested_pot_bonus * c_mult

        # ICM Equity Reward Shaping scaled by curriculum_shaping_mult
        if self.icm_equity_weight > 0.0:
            from games.holdem.icm_engine import calculate_icm_equities
            new_eq = calculate_icm_equities([s.chips for s in self.env.seats])[self.hero_seat]
            delta_eq = new_eq - prev_eq
            reward += float(delta_eq * 10.0 * self.icm_equity_weight * c_mult)

        terminated = (p.chips <= 0) or (self.env.session_done)
        truncated = False

        if p.chips <= 0 and self.elimination_penalty > 0:
            reward -= self.elimination_penalty
        elif self.env.session_done and self.win_bonus > 0 and self._get_info().get("is_winner", False):
            reward += self.win_bonus

        obs = build_observation_vector_holdem(
            self.env, self.hero_seat, obs_version=self.obs_version, tracker=self.tracker
        )
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _step_until_hero_turn(self):
        while not self.env.session_done:
            cur = self.env.current_seat
            if cur == self.hero_seat:
                p = self.env.seats[self.hero_seat]
                if p.is_alive and not p.folded and not p.is_all_in:
                    return

            opp_player = self.env.seats[cur]
            legal = self.env.get_legal_actions(cur)
            if not legal:
                self.env._advance_betting()
                continue

            bot = self.opponents.get(cur)
            if bot:
                if hasattr(bot, "act_with_env"):
                    act = bot.act_with_env(self.env, cur, legal)
                else:
                    act = bot.act(
                        hole_cards=opp_player.hand,
                        board=self.env.board,
                        pot=self.env.pot,
                        bet_to_call=opp_player.bet_to_call,
                        street=self.env.street,
                        stack=opp_player.chips,
                        legal_actions=legal,
                    )
            else:
                act = A_CHECK_CALL if A_CHECK_CALL in legal else legal[0]

            self.env.step_action(cur, act)

    def _get_info(self) -> dict:
        p = self.env.seats[self.hero_seat]
        chips = [s.chips for s in self.env.seats]
        leader = int(np.argmax(chips))
        is_winner = (p.chips > 0) and (
            (len(self.env.alive_seats) == 1 and self.env.alive_seats[0] == self.hero_seat)
            or (self.env.session_done and leader == self.hero_seat)
        )
        winner_seat = leader if (self.env.session_done or len(self.env.alive_seats) <= 1) else None

        # Privileged auxiliary targets for representation learning
        from games.holdem.opponent_exploiter import get_primary_adversary_seat, get_seat_archetype_probs
        adv_seat = get_primary_adversary_seat(self.env, self.hero_seat)
        probs = get_seat_archetype_probs(self.env, adv_seat)
        adv_archetype = int(np.argmax(probs))

        adv_hand_tier = 0
        adv_player = self.env.seats[adv_seat] if adv_seat < len(self.env.seats) else None
        if adv_player and len(adv_player.hand) == 2:
            if len(self.env.board) >= 3:
                from games.holdem.board_texture import evaluate_board_texture_and_draws
                adv_tex = evaluate_board_texture_and_draws(adv_player.hand, self.env.board)
                if any(adv_tex[k] > 0 for k in (8, 9, 10)):
                    adv_hand_tier = 3  # Monster
                elif adv_tex[5] > 0 or adv_tex[6] > 0:
                    adv_hand_tier = 2  # Top Pair / Overpair
                elif adv_tex[7] > 0:
                    adv_hand_tier = 1  # Weak Pair
                else:
                    adv_hand_tier = 0  # Air
            else:
                from games.holdem.evaluator import evaluate_preflop_hand_chen
                chen = evaluate_preflop_hand_chen(adv_player.hand[0], adv_player.hand[1])
                if chen >= 0.50:
                    adv_hand_tier = 3
                elif chen >= 0.35:
                    adv_hand_tier = 2
                elif chen >= 0.20:
                    adv_hand_tier = 1
                else:
                    adv_hand_tier = 0

        adv_folded = 1.0 if (adv_player and adv_player.folded) else 0.0

        return {
            "hero_chips": p.chips,
            "hands_played": self.env.hands_played,
            "is_winner": is_winner,
            "winner_seat": winner_seat,
            "pot": self.env.pot,
            "adv_seat": adv_seat,
            "adv_archetype": adv_archetype,
            "adv_hand_tier": adv_hand_tier,
            "adv_folded": adv_folded,
        }

