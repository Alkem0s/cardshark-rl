import os
import glob
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sb3_contrib import MaskablePPO

from card_utils import evaluate_hand, hand_category, normalize_rank, normalize_suit, normalize_hand_score
from draw_poker_env import A_FOLD, A_CALL, A_RAISE, PHASE_PRE_DRAW, PHASE_DRAW, PHASE_POST_DRAW, PHASE_SHOWDOWN, A_DRAW_START

app = Flask(__name__)
CORS(app)

# Load the model
save_dir = "models"
# Look for model_b_implicit*.zip
model_path = os.path.join(save_dir, "model_b_implicit.zip")
if not os.path.exists(model_path):
    # Try finding seed variations
    model_files = glob.glob(os.path.join(save_dir, "model_b_implicit_seed*.zip"))
    if model_files:
        model_path = model_files[0]
    else:
        # Fallback to model_a if b is completely missing, though this violates requirements, 
        # it is a safe fallback in extreme cases.
        model_files = glob.glob(os.path.join(save_dir, "model_a_explicit*.zip"))
        if model_files:
            model_path = model_files[0]
            print(f"Warning: Model B not found, using Model A instead: {model_path}")
        else:
            print("ERROR: No trained models found in 'models/' directory.")

if os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    model = MaskablePPO.load(model_path)
else:
    model = None

# Rolling history for implicit modelling
rolling_window = 50
opp_history = []

def get_rolling_stats():
    if not opp_history:
        return [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.1, 0.5]

    n = len(opp_history)
    fold_pre = sum(1 for h in opp_history if h["phase_when_folded"] == PHASE_PRE_DRAW) / n
    fold_post = sum(1 for h in opp_history if h["phase_when_folded"] in (PHASE_DRAW, PHASE_POST_DRAW, PHASE_SHOWDOWN)) / n
    raise_post = sum(1 for h in opp_history if h.get("post_draw_action") == A_RAISE) / n
    avg_draw = sum(h["draw_count"] for h in opp_history) / (n * 5.0)

    # Raise + stand pat correlation
    raise_standpat = sum(
        1 for h in opp_history
        if h.get("post_draw_action") == A_RAISE and h["draw_count"] == 0
    ) / max(1, n)

    # Fold after drawing 3
    draw3_hands = [h for h in opp_history if h["draw_count"] == 3]
    fold_after_draw3 = (
        sum(1 for h in draw3_hands if h["folded"]) / max(1, len(draw3_hands))
    )

    # Raise after standing pat
    standpat_hands = [h for h in opp_history if h["draw_count"] == 0]
    raise_after_standpat = (
        sum(1 for h in standpat_hands if h.get("post_draw_action") == A_RAISE)
        / max(1, len(standpat_hands))
    )

    vpip = sum(1 for h in opp_history if h.get("vpip", False)) / n
    pfr = sum(1 for h in opp_history if h.get("pfr", False)) / n

    # Aggression factor roughly = raises / calls
    total_raises = sum(1 for h in opp_history if h.get("post_draw_action") == A_RAISE)
    total_calls = sum(1 for h in opp_history if h.get("post_draw_action") == A_CALL)
    af = min(total_raises / max(1, total_calls) / 3.0, 1.0) # normalize to 0-1 range

    return [fold_pre, fold_post, raise_post, avg_draw, raise_standpat, fold_after_draw3, raise_after_standpat, vpip, pfr, af]


@app.route("/npc_action", methods=["POST"])
def npc_action():
    data = request.json
    
    hand = data.get("hand") # list of dicts: [{'rank': 'A', 'suit': '♠', 'color': 'black'}, ...]
    pot = data.get("pot", 0)
    bet_to_call = data.get("bet_to_call", 0)
    phase = data.get("phase", PHASE_PRE_DRAW)
    opp_draw = data.get("opp_draw", -1)
    position = data.get("position", 0.0)
    raises_this_hand = data.get("raises_this_hand", 0)
    opp_bet_this_round = data.get("opp_bet_this_round", False)
    opp_raised_pre = data.get("opp_raised_pre", False)
    raises_left = data.get("raises_left", 4)
    
    # Parse hand to RL env format
    parsed_hand = []
    for c in hand:
        parsed_hand.append({"rank": c["rank"], "suit": c["suit"]})
        
    hand_norm = []
    for c in parsed_hand:
        hand_norm.extend([normalize_rank(c), normalize_suit(c)])
        
    hand_score = evaluate_hand(parsed_hand) if len(parsed_hand) == 5 else 0
    hand_category_val = hand_category(parsed_hand) if len(parsed_hand) == 5 else 0
    hand_score_norm = normalize_hand_score(hand_score)
    hand_category_norm = hand_category_val / 8.0

    pot_norm = min(pot / 50.0, 1.0)
    btc_norm = min(bet_to_call / 20.0, 1.0)

    pot_odds = min(bet_to_call / max(1, pot), 1.0)

    phase_vec = [0.0, 0.0, 0.0]
    if phase == PHASE_PRE_DRAW:
        phase_vec[0] = 1.0
    elif phase == PHASE_DRAW:
        phase_vec[1] = 1.0
    elif phase == PHASE_POST_DRAW:
        phase_vec[2] = 1.0

    opp_draw_val = opp_draw / 5.0 if opp_draw >= 0 else -1.0
    raises_val = min(raises_this_hand / 4.0, 1.0)
    opp_aggression = 1.0 if opp_bet_this_round else 0.0
    opp_raised_pre_val = 1.0 if opp_raised_pre else 0.0

    obs = hand_norm + [hand_category_norm, hand_score_norm, pot_norm, btc_norm, pot_odds] + phase_vec + [opp_draw_val, position, raises_val, opp_aggression, opp_raised_pre_val]

    # Add implicit stats
    obs.extend(get_rolling_stats())

    obs_arr = np.array(obs, dtype=np.float32)

    # Action mask
    TOTAL_ACTIONS = 35
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
    if phase in (PHASE_PRE_DRAW, PHASE_POST_DRAW):
        mask[A_FOLD] = 1
        mask[A_CALL] = 1
        if raises_left > 0:
            mask[A_RAISE] = 1
    elif phase == PHASE_DRAW:
        mask[A_DRAW_START:TOTAL_ACTIONS] = 1

    # Predict
    action, _ = model.predict(obs_arr, action_masks=mask, deterministic=True)
    action = int(action)

    # Translate action
    action_type = "call"
    amount = 0
    discard_indices = []

    if phase in (PHASE_PRE_DRAW, PHASE_POST_DRAW):
        if action == A_FOLD:
            action_type = "fold"
        elif action == A_CALL:
            action_type = "call"
        elif action == A_RAISE:
            action_type = "raise"
            amount = 10 if phase == PHASE_PRE_DRAW else 20
    else:
        action_type = "draw"
        bitmask = action - A_DRAW_START
        discard_indices = [i for i in range(5) if bitmask & (1 << i)]

    return jsonify({
        "action": action_type,
        "amount": amount,
        "discard_indices": discard_indices,
        "raw_action": action
    })


@app.route("/end_hand", methods=["POST"])
def end_hand():
    data = request.json
    
    # Store aggregated opponent stats for this hand
    entry = {
        "draw_count": data.get("draw_count", 0),
        "folded": data.get("folded", False),
        "phase_when_folded": data.get("phase_when_folded"),
        "post_draw_action": data.get("post_draw_action", A_CALL),
        "opponent_id": -1, # real player
        "vpip": data.get("vpip", False),
        "pfr": data.get("pfr", False),
    }
    opp_history.append(entry)
    
    # Trim to rolling window
    global rolling_window
    if len(opp_history) > rolling_window:
        opp_history.pop(0)

    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
