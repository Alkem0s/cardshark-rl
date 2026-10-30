"""
rl/holdem_league.py — Multi-Agent Self-Play League Sparring for Texas Hold'em (PFSP).

Manages neural checkpoints and dynamic self-play sparring during training.
Decoupled cleanly from core game rules.
"""

from __future__ import annotations
import os
import glob
from typing import Dict, List, Optional, Any
import numpy as np

from games.holdem.opponents import (
    HoldemOpponent,
    TAGBot,
    LAGBot,
    CallingStationBot,
    RandomHoldemBot,
    get_cached_holdem_model,
)
from games.holdem.action_abstraction import A_CHECK_CALL, NUM_ACTIONS


class HoldemLeagueOpponent(HoldemOpponent):
    """
    Neural network opponent loaded from a frozen checkpoint (e.g. Model D baseline,
    Model E historical snapshot, or self-play clone).
    """

    def __init__(
        self,
        model_path: str,
        opponent_id: int = 100,
        name: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
        deterministic: bool = False,
        temperature: float = 0.0,
    ):
        base_name = name or f"League_{os.path.basename(model_path).replace('.zip', '')}"
        super().__init__(opponent_id=opponent_id, name=base_name, rng=rng)
        self.model_path = model_path
        self.deterministic = deterministic
        self.temperature = float(temperature)
        self.is_neural = True
        self.model = get_cached_holdem_model(model_path)

        # Detect observation version from model observation space
        if self.model is not None and hasattr(self.model, "observation_space"):
            dim = int(self.model.observation_space.shape[0])
            if dim == 168:
                self.obs_version = "f"
            elif dim == 158:
                self.obs_version = "e"
            elif dim == 152:
                self.obs_version = "d"
            elif dim == 138:
                self.obs_version = "c"
            elif dim in (119, 125):
                self.obs_version = "b"
            else:
                self.obs_version = "a"
        else:
            self.obs_version = "d"

    def act_with_env(self, env, seat_idx: int, legal_actions: List[int]) -> int:
        """Runs seat-perspective neural inference with action masking."""
        if not legal_actions:
            return A_CHECK_CALL

        if self.model is not None:
            try:
                from games.holdem.gym_wrapper import build_observation_vector_holdem
                obs = build_observation_vector_holdem(
                    env,
                    hero_seat=seat_idx,
                    obs_version=self.obs_version,
                    tracker=getattr(env, "tracker", None),
                )
                mask = env.get_action_mask(seat_idx)

                # Optional stochastic exploration jitter (prevents repetitive loop exploits)
                if self.temperature > 0.0 and self.rng.random() < self.temperature:
                    valid = np.where(mask == 1)[0]
                    if len(valid) > 0:
                        return int(self.rng.choice(valid))

                act, _ = self.model.predict(obs, action_masks=mask, deterministic=self.deterministic)
                action = int(act)
                if action in legal_actions:
                    return action
            except Exception:
                pass

        # Fallback to scalar evaluator heuristic if model predict fails
        seat = env.seats[seat_idx]
        return self.act(
            hole_cards=seat.hand,
            board=env.board,
            pot=env.pot,
            bet_to_call=seat.bet_to_call,
            street=env.street,
            stack=seat.chips,
            legal_actions=legal_actions,
        )

    def act(
        self,
        hole_cards: List[int],
        board: List[int],
        pot: int,
        bet_to_call: int,
        street: str,
        stack: int,
        legal_actions: List[int],
    ) -> int:
        if not legal_actions:
            return A_CHECK_CALL
        if len(hole_cards) == 2:
            try:
                from games.holdem.evaluator import evaluate_holdem_hand, evaluate_preflop_hand_chen
                from games.holdem.action_abstraction import (
                    STREET_PREFLOP, A_POT, A_MIN_RAISE, A_FOLD
                )
                if street == STREET_PREFLOP:
                    chen = evaluate_preflop_hand_chen(hole_cards[0], hole_cards[1])
                    if chen >= 0.45:
                        for a in [A_POT, A_MIN_RAISE]:
                            if a in legal_actions:
                                return a
                    elif chen >= 0.25:
                        return A_CHECK_CALL if A_CHECK_CALL in legal_actions else legal_actions[0]
                    else:
                        return A_FOLD if (A_FOLD in legal_actions and bet_to_call > 0) else A_CHECK_CALL
                else:
                    if len(board) >= 3:
                        score, _ = evaluate_holdem_hand(hole_cards + board)
                        if score >= 3_000_000:
                            for a in [A_POT, A_MIN_RAISE]:
                                if a in legal_actions:
                                    return a
                        elif score >= 1_500_000:
                            return A_CHECK_CALL if A_CHECK_CALL in legal_actions else legal_actions[0]
                        else:
                            if bet_to_call > 0:
                                return A_FOLD if A_FOLD in legal_actions else A_CHECK_CALL
                            return A_CHECK_CALL
            except Exception:
                pass
        return A_CHECK_CALL if A_CHECK_CALL in legal_actions else legal_actions[0]


class HoldemLeaguePool:
    """
    Manages a pool of historical neural checkpoints and heuristic sparring partners.
    Implements Prioritized Fictitious Self-Play (PFSP) with dynamic distribution sampling.
    """

    def __init__(
        self,
        league_dir: str = "models/holdem/league",
        anchor_model_path: str = "models/holdem/model_holdem_d_best.zip",
        max_snapshots: int = 15,
    ):
        self.league_dir = league_dir
        self.anchor_model_path = anchor_model_path
        self.max_snapshots = max_snapshots
        self.snapshots: List[str] = []
        self.current_progress: float = 0.0

        if league_dir:
            os.makedirs(league_dir, exist_ok=True)
            self.refresh_snapshots()

    def set_progress(self, progress: float):
        """Updates the training progress in [0.0, 1.0] for dynamic curriculum sampling."""
        self.current_progress = float(np.clip(progress, 0.0, 1.0))

    def refresh_snapshots(self) -> List[str]:
        """Discovers all valid .zip checkpoints in the league directory."""
        if not self.league_dir or not os.path.exists(self.league_dir):
            return []
        found = sorted(glob.glob(os.path.join(self.league_dir, "*.zip")), key=os.path.getmtime)
        self.snapshots = found[-self.max_snapshots :]
        return self.snapshots

    def add_snapshot(self, model_path: str):
        """Adds a newly saved checkpoint into the active league pool."""
        if os.path.exists(model_path) and model_path not in self.snapshots:
            self.snapshots.append(model_path)
            if len(self.snapshots) > self.max_snapshots:
                self.snapshots.pop(0)

    def register_checkpoint(self, model_path: str):
        """Compatibility alias for add_snapshot."""
        return self.add_snapshot(model_path)

    def sample_opponent(
        self,
        seat_idx: int,
        rng: Optional[np.random.Generator] = None,
        progress: Optional[float] = None,
    ) -> HoldemOpponent:
        """
        Samples an opponent according to dynamic PFSP curriculum distribution:
        - At progress = 0.0: 80% Heuristic Fundamentals (TAG, LAG, Calling Station),
                             15% Anchor Champion, 5% Random.
        - At progress = 1.0: 75% Self-Play League Sparring (45% Snapshots + 30% Anchor),
                             20% Heuristics, 5% Random.
        """
        r = rng or np.random.default_rng()
        p = float(np.clip(progress if progress is not None else self.current_progress, 0.0, 1.0))
        roll = r.random()

        has_anchor = os.path.exists(self.anchor_model_path)
        has_snapshots = len(self.snapshots) > 0

        # Dynamic probabilities:
        # Neural/Self-play probability scales smoothly from 15% to 75%
        p_neural = 0.15 + 0.60 * p
        p_random = 0.05

        # 1. Neural Self-Play Sparring
        if roll < p_neural and (has_snapshots or has_anchor):
            # If snapshots exist, snapshot share scales from 10% up to 60% of neural sparring
            snapshot_share = min(0.60, 0.10 + 0.50 * p) if has_snapshots else 0.0
            if has_snapshots and r.random() < snapshot_share:
                chosen = str(r.choice(self.snapshots))
                return HoldemLeagueOpponent(
                    model_path=chosen,
                    opponent_id=seat_idx,
                    rng=r,
                    deterministic=(r.random() < 0.70),
                    temperature=0.08,
                )
            elif has_anchor:
                anchor_name = f"Anchor_{os.path.basename(self.anchor_model_path).replace('.zip', '')}"
                return HoldemLeagueOpponent(
                    model_path=self.anchor_model_path,
                    opponent_id=seat_idx,
                    name=anchor_name,
                    rng=r,
                    deterministic=(r.random() < 0.85),
                    temperature=0.05,
                )

        # 2. Exploratory Random Bot (5%)
        elif roll >= (1.0 - p_random):
            return RandomHoldemBot(opponent_id=seat_idx, rng=r)

        # 3. Jittered Heuristic Archetypes (80% down to 20%)
        bot_choice = r.choice(["lag", "tag", "station"])
        if bot_choice == "lag":
            return LAGBot(opponent_id=seat_idx, rng=r)
        elif bot_choice == "tag":
            return TAGBot(opponent_id=seat_idx, rng=r)
        else:
            return CallingStationBot(opponent_id=seat_idx, rng=r)
