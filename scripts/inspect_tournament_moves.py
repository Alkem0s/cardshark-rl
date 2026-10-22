"""
scripts/inspect_tournament_moves.py — Multi-Game Poker Policy & Move Telemetry Inspector.

Simulates 50-session tournaments across Cardshark-RL's game modes:
1. No-Limit Texas Hold'em (Model Holdem-A)
2. 2-7 Triple Draw Lowball (Model 27-A)
3. Five Card Draw (Model F Co-Evolutionary Champion)

Captures and reports:
- Full action distributions (Aggressive, Passive, Fold, All-In)
- Street-by-street decision breakdowns (Preflop/Flop/Turn/River, Bet 1..4, Draw 1..3)
- Key Poker Strategic Indicators: VPIP %, PFR %, Aggression Factor (AF), Showdown Reach %, W$SD %
- Tournament survival metrics: 1st place championship %, elimination rate %, mean return
- Human-readable step-by-step sample hand transcripts showing exact moves, hole cards, boards, and outcomes.
"""

from __future__ import annotations
import os
import sys
import argparse
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sb3_contrib import MaskablePPO
from core.card import card_str, hand_str, rank_of, suit_of

# Ensure custom neural modules are registered in sys.modules for pickle loading
import core.neural.card_attention
import core.neural.decoupled_actor
sys.modules["card_attention"] = core.neural.card_attention
sys.modules["decoupled_actor"] = core.neural.decoupled_actor

# Game imports
from games.holdem.gym_wrapper import HoldemPokerGymEnv
from games.holdem.action_abstraction import (
    A_FOLD as H_FOLD,
    A_CHECK_CALL as H_CHECK_CALL,
    A_MIN_RAISE as H_MIN_RAISE,
    A_HALF_POT as H_HALF_POT,
    A_POT as H_POT,
    A_ALL_IN as H_ALL_IN,
    ACTION_NAMES as H_ACTION_NAMES,
    STREET_PREFLOP, STREET_FLOP, STREET_TURN, STREET_RIVER,
)

from games.triple_draw_27.gym_wrapper import TripleDrawPokerGymEnv
from games.triple_draw_27.env import (
    A_FOLD as TD_FOLD,
    A_CALL as TD_CALL,
    A_MIN_RAISE as TD_MIN_RAISE,
    A_HALF_POT as TD_HALF_POT,
    A_POT as TD_POT,
    A_ALL_IN as TD_ALL_IN,
    A_DRAW_START as TD_DRAW_START,
    PHASE_BET_1, PHASE_DRAW_1, PHASE_BET_2, PHASE_DRAW_2,
    PHASE_BET_3, PHASE_DRAW_3, PHASE_BET_4, PHASE_SHOWDOWN as TD_SHOWDOWN,
)

from rl.multi_gym_wrapper import MultiDrawPokerGymEnv, ModelObsDim
from games.five_card_draw.env import (
    A_FOLD as FCD_FOLD,
    A_CALL as FCD_CALL,
    A_MIN_RAISE as FCD_MIN_RAISE,
    A_HALF_POT as FCD_HALF_POT,
    A_POT as FCD_POT,
    A_ALL_IN as FCD_ALL_IN,
    A_DRAW_START as FCD_DRAW_START,
    PHASE_PRE_DRAW, PHASE_DRAW, PHASE_POST_DRAW, PHASE_SHOWDOWN as FCD_SHOWDOWN,
)


# =====================================================================
# 1. NO-LIMIT TEXAS HOLD'EM INSPECTOR
# =====================================================================
def inspect_holdem_moves(
    model_path: str = "models/holdem/model_holdem_b.zip",
    num_sessions: int = 50,
    seed_base: int = 500_000,
    max_hands: int = 100,
    sample_hands_to_print: int = 3,
) -> Dict[str, Any]:
    if not os.path.exists(model_path):
        if model_path == "models/holdem/model_holdem_b.zip" and os.path.exists("models/holdem/model_holdem_a.zip"):
            model_path = "models/holdem/model_holdem_a.zip"
        else:
            print(f"[Error] Checkpoint not found at: {model_path}")
            return {}

    model = MaskablePPO.load(model_path)
    obs_dim = model.observation_space.shape[0]
    obs_ver = "b" if obs_dim == 125 else "a"
    model_label = "MODEL HOLDEM-B" if obs_ver == "b" else "MODEL HOLDEM-A"

    print("\n" + "=" * 78)
    print(f"  GAME MODE: NO-LIMIT TEXAS HOLD'EM — {model_label} ({obs_dim} dims)")
    print("=" * 78)

    env = HoldemPokerGymEnv(
        num_seats=5,
        starting_chips=200,
        hero_seat=0,
        max_hands_per_session=max_hands,
        rng_seed=seed_base,
        obs_version=obs_ver,
    )

    action_counts = {a: 0 for a in range(6)}
    street_actions = {s: {a: 0 for a in range(6)} for s in [STREET_PREFLOP, STREET_FLOP, STREET_TURN, STREET_RIVER]}

    total_hands_played = 0
    vpip_hands = 0
    pfr_hands = 0
    showdowns_reached = 0
    showdowns_won = 0

    hero_titles = 0
    hero_eliminations = 0
    winner_counts: Dict[str, int] = {}
    total_rewards = []
    sample_hand_logs = []

    for sess_i in range(num_sessions):
        obs, info = env.reset(seed=seed_base + sess_i * 1000)
        done = False
        sess_r = 0.0

        current_hand_log = []
        last_hand_count = 0
        hero_vpip_current_hand = False
        hero_pfr_current_hand = False

        while not done:
            street = env.env.street
            mask = env.action_masks()
            act, _ = model.predict(obs, action_masks=mask, deterministic=True)
            act = int(act)

            action_counts[act] += 1
            if street in street_actions:
                street_actions[street][act] += 1

            if street == STREET_PREFLOP:
                if act in [H_CHECK_CALL, H_MIN_RAISE, H_HALF_POT, H_POT, H_ALL_IN]:
                    if env.env.seats[0].bet_to_call > 0 or act != H_CHECK_CALL:
                        hero_vpip_current_hand = True
                if act in [H_MIN_RAISE, H_HALF_POT, H_POT, H_ALL_IN]:
                    hero_pfr_current_hand = True

            # Hand history capture for sample logging
            if len(sample_hand_logs) < sample_hands_to_print:
                hero_hand_str = hand_str(env.env.seats[0].hand, symbols=True) if env.env.seats[0].hand else "None"
                board_str = hand_str([c for c in env.env.board if c != -1], symbols=True) if any(c != -1 for c in env.env.board) else "[]"
                current_hand_log.append(
                    f"  [{street.upper():8s}] Board: {board_str:16s} | Hero ({hero_hand_str}): {H_ACTION_NAMES[act]} "
                    f"(Pot: {env.env.pot}, Stack: {env.env.seats[0].chips}, ToCall: {env.env.seats[0].bet_to_call})"
                )

            obs, r, term, trunc, info = env.step(act)
            sess_r += r
            done = term or trunc

            # If hand ended
            if env.env.hands_played > last_hand_count or done:
                total_hands_played += 1
                if hero_vpip_current_hand:
                    vpip_hands += 1
                if hero_pfr_current_hand:
                    pfr_hands += 1

                if len(sample_hand_logs) < sample_hands_to_print and current_hand_log:
                    sample_hand_logs.append({
                        "hand_num": total_hands_played,
                        "session": sess_i + 1,
                        "log": list(current_hand_log),
                        "hero_final_chips": env.env.seats[0].chips,
                    })
                    current_hand_log = []

                hero_vpip_current_hand = False
                hero_pfr_current_hand = False
                last_hand_count = env.env.hands_played

        total_rewards.append(sess_r)
        if env.env.seats[0].chips == 0:
            hero_eliminations += 1
        
        chips = [s.chips for s in env.env.seats]
        leader_seat = int(np.argmax(chips))
        winner_seat = info.get("winner_seat", leader_seat)
        if info.get("is_winner", False) or winner_seat == env.hero_seat:
            hero_titles += 1
            winner_counts[model_label] = winner_counts.get(model_label, 0) + 1
        elif winner_seat in env.opponents:
            opp_name = env.opponents[winner_seat].name
            winner_counts[opp_name] = winner_counts.get(opp_name, 0) + 1
        elif leader_seat in env.opponents:
            opp_name = env.opponents[leader_seat].name
            winner_counts[opp_name] = winner_counts.get(opp_name, 0) + 1
        else:
            winner_counts["Other"] = winner_counts.get("Other", 0) + 1


    env.close()

    total_actions = max(1, sum(action_counts.values()))
    win_rate = (hero_titles / num_sessions) * 100.0
    elim_rate = (hero_eliminations / num_sessions) * 100.0
    vpip_pct = (vpip_hands / max(1, total_hands_played)) * 100.0
    pfr_pct = (pfr_hands / max(1, total_hands_played)) * 100.0
    af = (action_counts[H_MIN_RAISE] + action_counts[H_HALF_POT] + action_counts[H_POT] + action_counts[H_ALL_IN]) / max(1, action_counts[H_CHECK_CALL])

    print(f"Sessions Played: {num_sessions} | Total Hands: {total_hands_played} | Hero Decisions: {total_actions}")
    print(f"Championship Win Rate: {win_rate:5.1f}% ({hero_titles}/{num_sessions}) | Elimination Rate: {elim_rate:5.1f}% ({hero_eliminations}/{num_sessions})")
    print(f"Average Return: {np.mean(total_rewards):+6.2f} BBs | VPIP: {vpip_pct:5.1f}% | PFR: {pfr_pct:5.1f}% | Aggression Factor: {af:4.2f}")

    print("\n--- TOURNAMENT CHAMPIONSHIP LEADERBOARD ---")
    sorted_winners = sorted(winner_counts.items(), key=lambda kv: kv[1], reverse=True)
    for rank_idx, (name, count) in enumerate(sorted_winners, 1):
        pct = (count / num_sessions) * 100.0
        print(f"  #{rank_idx} {name:26s}: {count:3d}/{num_sessions} titles ({pct:5.1f}%)")


    print("\n--- ACTION DISTRIBUTION OVERALL ---")
    for a in range(6):
        cnt = action_counts[a]
        pct = (cnt / total_actions) * 100
        print(f"  {H_ACTION_NAMES[a]:12s} : {cnt:5d} ({pct:5.1f}%)")

    print("\n--- STREET-BY-STREET ACTION BREAKDOWN ---")
    for s in [STREET_PREFLOP, STREET_FLOP, STREET_TURN, STREET_RIVER]:
        s_tot = max(1, sum(street_actions[s].values()))
        s_pct_of_all = (s_tot / total_actions) * 100
        print(f"  [{s.upper():8s}] Total Moves: {s_tot:4d} ({s_pct_of_all:4.1f}% of decisions)")
        for a in range(6):
            cnt = street_actions[s][a]
            pct = (cnt / s_tot) * 100
            print(f"      {H_ACTION_NAMES[a]:12s} : {cnt:4d} ({pct:5.1f}%)")

    if sample_hand_logs:
        print("\n--- SAMPLE HAND ACTION TRANSCRIPTS ---")
        for sh in sample_hand_logs:
            print(f"  >> Hand #{sh['hand_num']} (Tournament Session #{sh['session']}) - Hero Final Chips: {sh['hero_final_chips']}")
            for line in sh["log"]:
                print(f"     {line}")
            print()

    return {
        "game": "Hold'em",
        "win_rate": win_rate,
        "elim_rate": elim_rate,
        "vpip": vpip_pct,
        "pfr": pfr_pct,
        "af": af,
        "actions": action_counts,
        "total_actions": total_actions,
    }


# =====================================================================
# 2. 2-7 TRIPLE DRAW LOWBALL INSPECTOR
# =====================================================================
def inspect_triple_draw_moves(
    model_path: str = "models/triple_draw_27/model_27_a.zip",
    num_sessions: int = 50,
    seed_base: int = 600_000,
    max_hands: int = 100,
    sample_hands_to_print: int = 3,
) -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print("  GAME MODE: 2-7 TRIPLE DRAW LOWBALL — MODEL 27-A")
    print("=" * 78)

    if not os.path.exists(model_path):
        print(f"[Error] Checkpoint not found at: {model_path}")
        return {}

    model = MaskablePPO.load(model_path)
    env = TripleDrawPokerGymEnv(
        num_seats=5,
        starting_chips=200,
        hero_seat=0,
        max_hands_per_session=max_hands,
        rng_seed=seed_base,
    )

    bet_action_counts = {a: 0 for a in range(6)}
    bet_names = {TD_FOLD: "Fold", TD_CALL: "Check/Call", TD_MIN_RAISE: "Min-Raise", TD_HALF_POT: "Half-Pot", TD_POT: "Pot", TD_ALL_IN: "All-In"}
    discard_counts = {k: 0 for k in range(6)} # 0 (pat), 1, 2, 3, 4, 5 cards

    draw_round_discards = {1: {k: 0 for k in range(6)}, 2: {k: 0 for k in range(6)}, 3: {k: 0 for k in range(6)}}

    total_hands_played = 0
    vpip_hands = 0
    pfr_hands = 0
    hero_titles = 0
    hero_eliminations = 0
    total_rewards = []
    sample_hand_logs = []

    for sess_i in range(num_sessions):
        obs, info = env.reset(seed=seed_base + sess_i * 1000)
        done = False
        sess_r = 0.0

        current_hand_log = []
        last_hand_count = 0
        hero_vpip_current_hand = False
        hero_pfr_current_hand = False

        while not done:
            phase = env.env.phase
            mask = env.action_masks()
            act, _ = model.predict(obs, action_masks=mask, deterministic=True)
            act = int(act)

            if act < TD_DRAW_START:
                bet_action_counts[act] += 1
                if phase == PHASE_BET_1:
                    if act in [TD_CALL, TD_MIN_RAISE, TD_HALF_POT, TD_POT, TD_ALL_IN]:
                        if env.env.seats[0].bet_to_call > 0 or act != TD_CALL:
                            hero_vpip_current_hand = True
                    if act in [TD_MIN_RAISE, TD_HALF_POT, TD_POT, TD_ALL_IN]:
                        hero_pfr_current_hand = True

                if len(sample_hand_logs) < sample_hands_to_print:
                    hand_display = hand_str(env.env.seats[0].hand, symbols=True)
                    current_hand_log.append(
                        f"  [{phase.replace('_', ' ').upper():11s}] Hand: {hand_display} | Hero: {bet_names[act]} "
                        f"(Pot: {env.env.pot}, Stack: {env.env.seats[0].chips}, ToCall: {env.env.seats[0].bet_to_call})"
                    )
            else:
                bitmask = act - TD_DRAW_START
                cards_discarded = bin(bitmask).count("1")
                discard_counts[cards_discarded] += 1

                round_num = 1 if phase == PHASE_DRAW_1 else (2 if phase == PHASE_DRAW_2 else 3)
                draw_round_discards[round_num][cards_discarded] += 1

                if len(sample_hand_logs) < sample_hands_to_print:
                    hand_display = hand_str(env.env.seats[0].hand, symbols=True)
                    draw_desc = "Stand Pat (0 cards)" if cards_discarded == 0 else f"Draw {cards_discarded} card(s)"
                    current_hand_log.append(f"  [DRAW ROUND {round_num}] Hand: {hand_display} | Hero: {draw_desc}")

            obs, r, term, trunc, info = env.step(act)
            sess_r += r
            done = term or trunc

            if env.env.hands_played > last_hand_count or done:
                total_hands_played += 1
                if hero_vpip_current_hand:
                    vpip_hands += 1
                if hero_pfr_current_hand:
                    pfr_hands += 1

                if len(sample_hand_logs) < sample_hands_to_print and current_hand_log:
                    sample_hand_logs.append({
                        "hand_num": total_hands_played,
                        "session": sess_i + 1,
                        "log": list(current_hand_log),
                        "hero_final_chips": env.env.seats[0].chips,
                    })
                    current_hand_log = []

                hero_vpip_current_hand = False
                hero_pfr_current_hand = False
                last_hand_count = env.env.hands_played

        total_rewards.append(sess_r)
        if env.env.seats[0].chips == 0:
            hero_eliminations += 1
        if info.get("is_winner", False):
            hero_titles += 1

    env.close()

    total_bets = max(1, sum(bet_action_counts.values()))
    total_draws = max(1, sum(discard_counts.values()))
    win_rate = (hero_titles / num_sessions) * 100.0
    elim_rate = (hero_eliminations / num_sessions) * 100.0
    vpip_pct = (vpip_hands / max(1, total_hands_played)) * 100.0
    pfr_pct = (pfr_hands / max(1, total_hands_played)) * 100.0
    af = (bet_action_counts[TD_MIN_RAISE] + bet_action_counts[TD_HALF_POT] + bet_action_counts[TD_POT] + bet_action_counts[TD_ALL_IN]) / max(1, bet_action_counts[TD_CALL])

    print(f"Sessions Played: {num_sessions} | Total Hands: {total_hands_played} | Total Betting Moves: {total_bets} | Total Draws: {total_draws}")
    print(f"Championship Win Rate: {win_rate:5.1f}% ({hero_titles}/{num_sessions}) | Elimination Rate: {elim_rate:5.1f}% ({hero_eliminations}/{num_sessions})")
    print(f"Average Return: {np.mean(total_rewards):+6.2f} BBs | VPIP: {vpip_pct:5.1f}% | PFR: {pfr_pct:5.1f}% | Aggression Factor: {af:4.2f}")

    print("\n--- BETTING ACTION DISTRIBUTION ---")
    for a in range(6):
        cnt = bet_action_counts[a]
        pct = (cnt / total_bets) * 100
        print(f"  {bet_names[a]:12s} : {cnt:5d} ({pct:5.1f}%)")

    print("\n--- DISCARD / DRAW PATTERNS ACROSS ROUNDS 1, 2, 3 ---")
    for r_i in [1, 2, 3]:
        r_tot = max(1, sum(draw_round_discards[r_i].values()))
        pat_cnt = draw_round_discards[r_i][0]
        d1_cnt = draw_round_discards[r_i][1]
        d2_cnt = draw_round_discards[r_i][2]
        d3_cnt = draw_round_discards[r_i][3]
        print(
            f"  [Draw {r_i}] Stand Pat: {pat_cnt:3d} ({pat_cnt/r_tot*100:4.1f}%) | "
            f"Draw 1: {d1_cnt:3d} ({d1_cnt/r_tot*100:4.1f}%) | "
            f"Draw 2: {d2_cnt:3d} ({d2_cnt/r_tot*100:4.1f}%) | "
            f"Draw 3+: {d3_cnt:3d} ({d3_cnt/r_tot*100:4.1f}%)"
        )

    if sample_hand_logs:
        print("\n--- SAMPLE HAND ACTION TRANSCRIPTS ---")
        for sh in sample_hand_logs:
            print(f"  >> Hand #{sh['hand_num']} (Tournament Session #{sh['session']}) - Hero Final Chips: {sh['hero_final_chips']}")
            for line in sh["log"]:
                print(f"     {line}")
            print()

    return {
        "game": "2-7 Triple Draw",
        "win_rate": win_rate,
        "elim_rate": elim_rate,
        "vpip": vpip_pct,
        "pfr": pfr_pct,
        "af": af,
        "actions": bet_action_counts,
        "total_actions": total_bets + total_draws,
    }


# =====================================================================
# 3. FIVE CARD DRAW INSPECTOR
# =====================================================================
def inspect_five_card_draw_moves(
    model_path: str = "models/five_card_draw/model_f.zip",
    num_sessions: int = 50,
    seed_base: int = 700_000,
    max_hands: int = 100,
    sample_hands_to_print: int = 3,
) -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print("  GAME MODE: FIVE CARD DRAW — MODEL F (CO-EVOLUTIONARY CHAMPION)")
    print("=" * 78)

    if not os.path.exists(model_path):
        print(f"[Error] Checkpoint not found at: {model_path}")
        return {}

    model = MaskablePPO.load(model_path)
    env = MultiDrawPokerGymEnv(
        num_seats=5,
        starting_chips=200,
        hero_seat=0,
        max_hands_per_session=max_hands,
        obs_dim=ModelObsDim.MODEL_F,
        discard_masking=True,
        rng_seed=seed_base,
    )

    bet_action_counts = {a: 0 for a in range(6)}
    bet_names = {FCD_FOLD: "Fold", FCD_CALL: "Check/Call", FCD_MIN_RAISE: "Min-Raise", FCD_HALF_POT: "Half-Pot", FCD_POT: "Pot", FCD_ALL_IN: "All-In"}
    discard_counts = {k: 0 for k in range(6)}

    total_hands_played = 0
    vpip_hands = 0
    pfr_hands = 0
    hero_titles = 0
    hero_eliminations = 0
    total_rewards = []
    sample_hand_logs = []

    for sess_i in range(num_sessions):
        obs, info = env.reset(seed=seed_base + sess_i * 1000)
        done = False
        sess_r = 0.0

        current_hand_log = []
        last_hand_count = 0
        hero_vpip_current_hand = False
        hero_pfr_current_hand = False

        while not done:
            phase = env.env.phase
            mask = env.action_masks()
            act, _ = model.predict(obs, action_masks=mask, deterministic=True)
            act = int(act)

            if act < FCD_DRAW_START:
                bet_action_counts[act] += 1
                if phase == PHASE_PRE_DRAW:
                    if act in [FCD_CALL, FCD_MIN_RAISE, FCD_HALF_POT, FCD_POT, FCD_ALL_IN]:
                        if env.env.seats[0].bet_to_call > 0 or act != FCD_CALL:
                            hero_vpip_current_hand = True
                    if act in [FCD_MIN_RAISE, FCD_HALF_POT, FCD_POT, FCD_ALL_IN]:
                        hero_pfr_current_hand = True

                if len(sample_hand_logs) < sample_hands_to_print:
                    p_name = "PRE-DRAW" if phase == PHASE_PRE_DRAW else "POST-DRAW"
                    hand_display = hand_str(env.env.seats[0].hand, symbols=True)
                    current_hand_log.append(
                        f"  [{p_name:9s}] Hand: {hand_display} | Hero: {bet_names[act]} "
                        f"(Pot: {env.env.pot}, Stack: {env.env.seats[0].chips}, ToCall: {env.env.seats[0].bet_to_call})"
                    )
            else:
                bitmask = act - FCD_DRAW_START
                cards_discarded = bin(bitmask).count("1")
                discard_counts[cards_discarded] += 1

                if len(sample_hand_logs) < sample_hands_to_print:
                    hand_display = hand_str(env.env.seats[0].hand, symbols=True)
                    draw_desc = "Stand Pat (0 cards)" if cards_discarded == 0 else f"Draw {cards_discarded} card(s)"
                    current_hand_log.append(f"  [DRAW PHASE] Hand: {hand_display} | Hero: {draw_desc}")

            obs, r, term, trunc, info = env.step(act)
            sess_r += r
            done = term or trunc

            if env.env.hands_played > last_hand_count or done:
                total_hands_played += 1
                if hero_vpip_current_hand:
                    vpip_hands += 1
                if hero_pfr_current_hand:
                    pfr_hands += 1

                if len(sample_hand_logs) < sample_hands_to_print and current_hand_log:
                    sample_hand_logs.append({
                        "hand_num": total_hands_played,
                        "session": sess_i + 1,
                        "log": list(current_hand_log),
                        "hero_final_chips": env.env.seats[0].chips,
                    })
                    current_hand_log = []

                hero_vpip_current_hand = False
                hero_pfr_current_hand = False
                last_hand_count = env.env.hands_played

        total_rewards.append(sess_r)
        if env.env.seats[0].chips == 0:
            hero_eliminations += 1
        if info.get("is_winner", False):
            hero_titles += 1

    env.close()

    total_bets = max(1, sum(bet_action_counts.values()))
    total_draws = max(1, sum(discard_counts.values()))
    win_rate = (hero_titles / num_sessions) * 100.0
    elim_rate = (hero_eliminations / num_sessions) * 100.0
    vpip_pct = (vpip_hands / max(1, total_hands_played)) * 100.0
    pfr_pct = (pfr_hands / max(1, total_hands_played)) * 100.0
    af = (bet_action_counts[FCD_MIN_RAISE] + bet_action_counts[FCD_HALF_POT] + bet_action_counts[FCD_POT] + bet_action_counts[FCD_ALL_IN]) / max(1, bet_action_counts[FCD_CALL])

    print(f"Sessions Played: {num_sessions} | Total Hands: {total_hands_played} | Total Betting Moves: {total_bets} | Total Draws: {total_draws}")
    print(f"Championship Win Rate: {win_rate:5.1f}% ({hero_titles}/{num_sessions}) | Elimination Rate: {elim_rate:5.1f}% ({hero_eliminations}/{num_sessions})")
    print(f"Average Return: {np.mean(total_rewards):+6.2f} BBs | VPIP: {vpip_pct:5.1f}% | PFR: {pfr_pct:5.1f}% | Aggression Factor: {af:4.2f}")

    print("\n--- BETTING ACTION DISTRIBUTION ---")
    for a in range(6):
        cnt = bet_action_counts[a]
        pct = (cnt / total_bets) * 100
        print(f"  {bet_names[a]:12s} : {cnt:5d} ({pct:5.1f}%)")

    print("\n--- DISCARD / DRAW PATTERNS ---")
    for k in range(6):
        cnt = discard_counts[k]
        pct = (cnt / total_draws) * 100
        desc = "Stand Pat" if k == 0 else f"Draw {k} Cards"
        print(f"  {desc:14s} : {cnt:4d} ({pct:5.1f}%)")

    if sample_hand_logs:
        print("\n--- SAMPLE HAND ACTION TRANSCRIPTS ---")
        for sh in sample_hand_logs:
            print(f"  >> Hand #{sh['hand_num']} (Tournament Session #{sh['session']}) - Hero Final Chips: {sh['hero_final_chips']}")
            for line in sh["log"]:
                print(f"     {line}")
            print()

    return {
        "game": "Five Card Draw",
        "win_rate": win_rate,
        "elim_rate": elim_rate,
        "vpip": vpip_pct,
        "pfr": pfr_pct,
        "af": af,
        "actions": bet_action_counts,
        "total_actions": total_bets + total_draws,
    }


# =====================================================================
# 4. MAIN RUNNER & CONSOLIDATED SUMMARY
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Inspect Moves and Strategic Telemetry in 50-Session Tournaments")
    parser.add_argument("--game", choices=["holdem", "27", "triple_draw_27", "fcd", "five_card_draw", "all"], default="all")
    parser.add_argument("--sessions", type=int, default=50, help="Number of tournament sessions to run")
    parser.add_argument("--max-hands", type=int, default=100, help="Max hands per session")
    parser.add_argument("--seed", type=int, default=500_000, help="Base random seed")
    parser.add_argument("--sample-hands", type=int, default=3, help="Number of sample hand history logs to display")
    parser.add_argument("--model-holdem", type=str, default="models/holdem/model_holdem_b.zip", help="Path to Hold'em model checkpoint")
    args = parser.parse_args()

    results = []

    target_game = args.game.lower()

    if target_game in ["holdem", "all"]:
        r_h = inspect_holdem_moves(
            model_path=args.model_holdem,
            num_sessions=args.sessions,
            seed_base=args.seed,
            max_hands=args.max_hands,
            sample_hands_to_print=args.sample_hands,
        )
        if r_h:
            results.append(r_h)

    if target_game in ["27", "triple_draw_27", "all"]:
        r_27 = inspect_triple_draw_moves(
            model_path="models/triple_draw_27/model_27_a.zip",
            num_sessions=args.sessions,
            seed_base=args.seed + 100_000,
            max_hands=args.max_hands,
            sample_hands_to_print=args.sample_hands,
        )
        if r_27:
            results.append(r_27)

    if target_game in ["fcd", "five_card_draw", "all"]:
        r_fcd = inspect_five_card_draw_moves(
            model_path="models/five_card_draw/model_f.zip",
            num_sessions=args.sessions,
            seed_base=args.seed + 200_000,
            max_hands=args.max_hands,
            sample_hands_to_print=args.sample_hands,
        )
        if r_fcd:
            results.append(r_fcd)

    if len(results) > 1:
        print("\n" + "=" * 88)
        print("                 CONSOLIDATED MULTI-GAME TOURNAMENT BENCHMARK (50 SESSIONS)")
        print("=" * 88)
        header = f"{'Game Mode':<20s} | {'Win Rate %':<12s} | {'Elim Rate %':<12s} | {'VPIP %':<10s} | {'PFR %':<10s} | {'AF Ratio':<10s}"
        print(header)
        print("-" * len(header))
        for res in results:
            print(
                f"{res['game']:<20s} | "
                f"{res['win_rate']:>10.1f}% | "
                f"{res['elim_rate']:>10.1f}% | "
                f"{res['vpip']:>8.1f}% | "
                f"{res['pfr']:>8.1f}% | "
                f"{res['af']:>8.2f}"
            )
        print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
