"""
draw_poker_env.py — Custom PettingZoo AECEnv for Limit Heads-Up 5-Card Draw Poker.

Game flow:
    1. Antes posted, 5 cards dealt to each player.
    2. Pre-draw betting round (limit = SMALL_BET).
    3. Draw phase — each player discards 0‑5 cards and draws replacements.
    4. Post-draw betting round (limit = BIG_BET).
    5. Showdown (if no fold).

Action space  Discrete(35):
    0  = Fold
    1  = Call / Check
    2  = Raise / Bet
    3‑34 = Draw actions  (action − 3 = 5-bit bitmask; bit i = discard card[i])

The environment embeds the opponent logic and presents a single-agent
interface to the RL agent via the Gymnasium wrapper (gym_wrapper.py).
"""

from __future__ import annotations
import functools
import numpy as np
from typing import Dict, List, Optional

from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces

from card_utils import Deck, evaluate_hand, rank_of, normalize_rank, normalize_suit, normalize_hand_score, hand_category
from opponents import Opponent, make_opponent, make_random_opponent, NUM_ARCHETYPES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SMALL_BET = 2
BIG_BET   = 4
ANTE      = 1
MAX_RAISES = 4
NUM_DRAW_ACTIONS = 32   # 2^5 bitmask combos

PHASE_PRE_DRAW  = "pre_draw"
PHASE_DRAW      = "draw"
PHASE_POST_DRAW = "post_draw"
PHASE_SHOWDOWN  = "showdown"

# Action indices
A_FOLD  = 0
A_CALL  = 1
A_RAISE = 2
A_DRAW_START = 3
A_DRAW_END   = A_DRAW_START + NUM_DRAW_ACTIONS  # 35 (exclusive)
TOTAL_ACTIONS = A_DRAW_END  # 35


# ---------------------------------------------------------------------------
# PettingZoo AEC Environment
# ---------------------------------------------------------------------------

class DrawPokerEnv(AECEnv):
    """Limit Heads-Up 5-Card Draw — AEC implementation."""

    metadata = {
        "render_modes": ["human"],
        "name": "draw_poker_v0",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        use_implicit_modeling: bool = False,
        opponent_id: int | None = None,
        rolling_window: int = 50,
        render_mode: str | None = None,
        rng_seed: int | None = None,
        opponent_schedule: str = "random",
        block_size: int = 200,
        hybrid_switch_episodes: int | None = None,
    ):
        super().__init__()

        self.use_implicit_modeling = use_implicit_modeling
        self.fixed_opponent_id = opponent_id  # None = random each episode
        self.rolling_window = rolling_window
        self.render_mode = render_mode
        self.opponent_schedule = opponent_schedule  # "random", "block", or "hybrid"
        self.block_size = block_size
        self.hybrid_switch_episodes = hybrid_switch_episodes

        # Block / hybrid scheduling state
        self._block_hand_count = 0
        self._total_episodes = 0
        self._block_current_opp_id = 0

        self.rng = np.random.default_rng(rng_seed)
        self.deck = Deck(rng=self.rng)

        # Agents
        self.possible_agents = ["player_0", "player_1"]
        self.agent_name_mapping = {a: i for i, a in enumerate(self.possible_agents)}

        # Observation & action spaces
        self._obs_size = self._compute_obs_size()

        # Rolling opponent history (for implicit modelling)
        self._opp_history: List[dict] = []

        # State (initialised in reset)
        self.hands: Dict[str, List[int]] = {}
        self.pot = 0
        self.phase = PHASE_PRE_DRAW
        self.total_invested: Dict[str, int] = {}  # Total chips invested this hand
        self.raises_left = MAX_RAISES
        self.folded: Dict[str, bool] = {}
        self.draw_counts: Dict[str, int] = {}  # cards drawn this hand
        self.current_opponent: Optional[Opponent] = None

        # Tracking for new strategic features
        self.raises_this_hand: Dict[str, int] = {a: 0 for a in self.possible_agents}
        self.raised_pre_draw: Dict[str, bool] = {a: False for a in self.possible_agents}
        self.opp_bet_this_round: bool = False

        # Rewards accumulated
        self.rewards = {a: 0 for a in self.possible_agents}
        self._cumulative_rewards = {a: 0 for a in self.possible_agents}
        self.terminations = {a: False for a in self.possible_agents}
        self.truncations = {a: False for a in self.possible_agents}
        self.infos = {a: {} for a in self.possible_agents}

        self._agent_selector = agent_selector(self.possible_agents)
        self.agent_selection = self.possible_agents[0]

        # Track who has acted in the draw phase
        self._draw_acted = {a: False for a in self.possible_agents}

        # Betting state
        self._bet_to_call: Dict[str, int] = {}
        # Track how many actions have been taken this betting round
        # A round is complete when actions_this_round >= 2 and all bets matched
        self._actions_this_round = 0

        # Track which agent is "in position" (acts second in betting)
        # player_0 is always the RL agent
        self._betting_order = ["player_0", "player_1"]

        # Who acts next in the current betting round
        self._current_bettor_idx = 0

        # For draw phase, non-dealer draws first (we alternate dealer)
        self._dealer = "player_1"  # Opponent is dealer first

        # Track opponent's post-draw action for history
        self._opp_post_draw_action = A_CALL

    def _compute_obs_size(self) -> int:
        # 10 (cards: rank+suit) + 1 (category) + 1 (score) + 1 (pot) + 1 (bet_to_call)
        # + 1 (pot_odds) + 3 (phase) + 1 (opp_draw)
        # + 1 (position) + 1 (raises) + 1 (opp_aggression) + 1 (opp_raised_pre_draw) = 23
        base = 23
        if self.use_implicit_modeling:
            base += 10   # 10 rolling stats
        else:
            base += NUM_ARCHETYPES  # 3 one-hot
        return base

    # -----------------------------------------------------------------------
    # PettingZoo API
    # -----------------------------------------------------------------------

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Dict({
            "observation": spaces.Box(low=0.0, high=1.0, shape=(self._obs_size,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(TOTAL_ACTIONS),
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(TOTAL_ACTIONS)

    def observe(self, agent):
        obs = self._build_observation(agent)
        mask = self._build_action_mask(agent)
        return {"observation": obs, "action_mask": mask}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.deck = Deck(rng=self.rng)

        self._total_episodes += 1

        # Determine effective schedule for this episode
        if self.opponent_schedule == "hybrid" and self.hybrid_switch_episodes:
            effective_schedule = "block" if self._total_episodes <= self.hybrid_switch_episodes else "random"
        else:
            effective_schedule = self.opponent_schedule

        # Select opponent
        if self.fixed_opponent_id is not None:
            self.current_opponent = make_opponent(self.fixed_opponent_id, rng=self.rng)
        elif effective_schedule == "block":
            # Block scheduling: keep the same opponent for block_size hands
            if self._block_hand_count >= self.block_size:
                self._block_hand_count = 0
                self._block_current_opp_id = (self._block_current_opp_id + 1) % NUM_ARCHETYPES
            self.current_opponent = make_opponent(self._block_current_opp_id, rng=self.rng)
            self._block_hand_count += 1
        else:
            self.current_opponent = make_random_opponent(self.rng)

        # Reset deck and deal
        self.deck.reset()
        self.hands = {
            "player_0": self.deck.deal(5),
            "player_1": self.deck.deal(5),
        }

        # Antes
        self.pot = 2 * ANTE
        self.total_invested = {"player_0": ANTE, "player_1": ANTE}
        self._bet_to_call = {"player_0": 0, "player_1": 0}
        self._actions_this_round = 0

        # Phase
        self.phase = PHASE_PRE_DRAW
        self.raises_left = MAX_RAISES

        # Status
        self.folded = {a: False for a in self.possible_agents}
        self.draw_counts = {a: 0 for a in self.possible_agents}
        self._draw_acted = {a: False for a in self.possible_agents}
        self._opp_post_draw_action = A_CALL
        self.raises_this_hand = {a: 0 for a in self.possible_agents}
        self.raised_pre_draw = {a: False for a in self.possible_agents}
        self.opp_bet_this_round = False

        # Agents
        self.agents = list(self.possible_agents)
        self.rewards = {a: 0 for a in self.possible_agents}
        self._cumulative_rewards = {a: 0 for a in self.possible_agents}
        self.terminations = {a: False for a in self.possible_agents}
        self.truncations = {a: False for a in self.possible_agents}
        self.infos = {a: {} for a in self.possible_agents}

        # Alternate dealer
        self._dealer = "player_1" if self._dealer == "player_0" else "player_0"

        # Betting order: non-dealer acts first
        if self._dealer == "player_1":
            self._betting_order = ["player_0", "player_1"]
        else:
            self._betting_order = ["player_1", "player_0"]

        self._current_bettor_idx = 0
        self.agent_selection = self._betting_order[0]

        # If the first to act is the opponent, process their action immediately
        self._auto_step_opponent()

    def step(self, action):
        """Process an action for the current agent."""
        agent = self.agent_selection

        if self.terminations.get(agent, False) or self.truncations.get(agent, False):
            # PettingZoo requires None for dead agents
            self._was_dead_step(action)
            return

        # Execute the action
        self._execute_action(agent, action)

        # Check for fold
        if self._check_fold_end():
            return

        # Advance game state
        self._advance_game_state()

        # Auto-step opponent if it's their turn
        if not self._is_game_over():
            self._auto_step_opponent()

    def _execute_action(self, agent: str, action: int):
        """Execute a single action."""
        if self.phase in (PHASE_PRE_DRAW, PHASE_POST_DRAW):
            self._process_bet_action(agent, action)
        elif self.phase == PHASE_DRAW:
            self._process_draw_action(agent, action)

    def _advance_game_state(self):
        """Advance to the next agent or next phase."""
        if self._is_game_over():
            return

        if self.phase in (PHASE_PRE_DRAW, PHASE_POST_DRAW):
            # Check if betting round is complete
            all_matched = all(v == 0 for v in self._bet_to_call.values())
            if all_matched and self._actions_this_round >= 2:
                # Betting round done
                if self.phase == PHASE_PRE_DRAW:
                    self._start_draw_phase()
                else:
                    self._showdown()
                return
            else:
                # Next bettor
                self._current_bettor_idx = 1 - self._current_bettor_idx
                self.agent_selection = self._betting_order[self._current_bettor_idx]

        elif self.phase == PHASE_DRAW:
            if all(self._draw_acted.values()):
                self._start_post_draw_betting()
                return
            else:
                # Next drawer
                self._current_bettor_idx = 1 - self._current_bettor_idx
                self.agent_selection = self._betting_order[self._current_bettor_idx]

    def _auto_step_opponent(self):
        """If it's the opponent's turn, auto-step until it's the RL agent's turn or game ends."""
        max_iter = 100  # Safety guard against infinite loops
        iters = 0
        while (
            self.agent_selection == "player_1"
            and not self._is_game_over()
            and iters < max_iter
        ):
            iters += 1
            opp_action = self._get_opponent_action()
            self._execute_action("player_1", opp_action)

            if self._check_fold_end():
                return

            self._advance_game_state()

    def _is_game_over(self) -> bool:
        return all(self.terminations.values())

    # -----------------------------------------------------------------------
    # Action processing
    # -----------------------------------------------------------------------

    def _process_bet_action(self, agent: str, action: int):
        """Handle a betting action (fold/call/raise)."""
        action = min(action, 2)  # Clamp to valid betting range

        bet_size = SMALL_BET if self.phase == PHASE_PRE_DRAW else BIG_BET
        other = "player_1" if agent == "player_0" else "player_0"

        self._actions_this_round += 1

        if action == A_FOLD:
            self.folded[agent] = True
            return

        elif action == A_CALL:
            # Match the current bet (or check if nothing to call)
            call_amount = self._bet_to_call[agent]
            self.total_invested[agent] += call_amount
            self.pot += call_amount
            self._bet_to_call[agent] = 0

        elif action == A_RAISE:
            if self.raises_left <= 0:
                # Can't raise — treat as call
                call_amount = self._bet_to_call[agent]
                self.total_invested[agent] += call_amount
                self.pot += call_amount
                self._bet_to_call[agent] = 0
            else:
                # First call any outstanding bet, then raise
                call_amount = self._bet_to_call[agent]
                total = call_amount + bet_size
                self.total_invested[agent] += total
                self.pot += total
                self._bet_to_call[agent] = 0
                self._bet_to_call[other] += bet_size
                self.raises_left -= 1
                self.raises_this_hand[agent] += 1
                if self.phase == PHASE_PRE_DRAW:
                    self.raised_pre_draw[agent] = True
                # After a raise, the opponent needs to act again,
                # so we reset the action count to 1 (only raiser has acted)
                self._actions_this_round = 1

        # Track opponent post-draw action and aggression
        if agent == "player_1":
            if self.phase == PHASE_POST_DRAW:
                self._opp_post_draw_action = action
            if action == A_RAISE or (action == A_CALL and self._bet_to_call["player_1"] > 0):
                self.opp_bet_this_round = True

    def _process_draw_action(self, agent: str, action: int):
        """Handle a draw-phase action (discard bitmask)."""
        if action < A_DRAW_START:
            action = A_DRAW_START  # Stand pat if invalid

        bitmask = action - A_DRAW_START
        discard_indices = [i for i in range(5) if bitmask & (1 << i)]

        num_discard = len(discard_indices)
        self.draw_counts[agent] = num_discard

        if num_discard > 0 and self.deck.remaining >= num_discard:
            new_cards = self.deck.draw(num_discard)
            hand = self.hands[agent]
            for idx, new_card in zip(sorted(discard_indices), new_cards):
                hand[idx] = new_card

        self._draw_acted[agent] = True

    # -----------------------------------------------------------------------
    # Phase management
    # -----------------------------------------------------------------------

    def _check_fold_end(self) -> bool:
        """Check if someone folded → end the hand."""
        for agent in self.possible_agents:
            if self.folded[agent]:
                winner = "player_1" if agent == "player_0" else "player_0"
                self._end_hand(winner)
                return True
        return False

    def _start_draw_phase(self):
        """Transition to draw phase."""
        self.phase = PHASE_DRAW
        self._draw_acted = {a: False for a in self.possible_agents}
        self._current_bettor_idx = 0
        self.agent_selection = self._betting_order[0]

    def _start_post_draw_betting(self):
        """Transition to post-draw betting."""
        self.phase = PHASE_POST_DRAW
        self.raises_left = MAX_RAISES
        self._bet_to_call = {a: 0 for a in self.possible_agents}
        self._actions_this_round = 0
        self.opp_bet_this_round = False
        self._current_bettor_idx = 0
        self.agent_selection = self._betting_order[0]

    def _showdown(self):
        """Compare hands and determine winner."""
        score_0 = evaluate_hand(self.hands["player_0"])
        score_1 = evaluate_hand(self.hands["player_1"])

        if score_0 > score_1:
            self._end_hand("player_0")
        elif score_1 > score_0:
            self._end_hand("player_1")
        else:
            # Tie — split pot
            self._end_hand(None)

    def _end_hand(self, winner: str | None):
        """Distribute pot and mark hand as done."""
        if winner is not None:
            loser = "player_1" if winner == "player_0" else "player_0"
            # Reward = net profit (pot won minus what you invested)
            self.rewards[winner] = self.pot - self.total_invested[winner]
            self.rewards[loser] = -self.total_invested[loser]
        else:
            # Split: each gets back their contribution (net 0)
            for a in self.possible_agents:
                self.rewards[a] = 0

        # Accumulate
        for a in self.possible_agents:
            self._cumulative_rewards[a] += self.rewards[a]

        # Record opponent history for implicit modelling
        self._record_opponent_history()

        # Mark done
        for a in self.possible_agents:
            self.terminations[a] = True

        self.phase = PHASE_SHOWDOWN

    # -----------------------------------------------------------------------
    # Opponent auto-play
    # -----------------------------------------------------------------------

    def _get_opponent_action(self) -> int:
        """Query the hardcoded opponent for its action."""
        opp = self.current_opponent
        hand = self.hands["player_1"]

        if self.phase in (PHASE_PRE_DRAW, PHASE_POST_DRAW):
            phase_str = "pre_draw" if self.phase == PHASE_PRE_DRAW else "post_draw"
            can_raise = self.raises_left > 0
            bet_to_call = self._bet_to_call["player_1"]
            action = opp.bet_action(hand, self.pot, bet_to_call, phase_str, can_raise)
            return min(action, 2)

        elif self.phase == PHASE_DRAW:
            discard_indices = opp.draw_action(hand)
            # Convert to bitmask action
            bitmask = 0
            for idx in discard_indices:
                bitmask |= (1 << idx)
            return A_DRAW_START + bitmask

        return A_CALL  # Fallback

    # -----------------------------------------------------------------------
    # Observation construction
    # -----------------------------------------------------------------------

    def _build_observation(self, agent: str) -> np.ndarray:
        """Build the observation vector for the given agent."""
        hand = self.hands.get(agent, [0, 0, 0, 0, 0])
        hand_norm = []
        for c in hand:
            hand_norm.extend([normalize_rank(c), normalize_suit(c)])

        hand_score = evaluate_hand(hand) if len(hand) == 5 else 0
        hand_category_val = hand_category(hand) if len(hand) == 5 else 0
        hand_score_norm = normalize_hand_score(hand_score)
        hand_category_norm = hand_category_val / 8.0

        pot_norm = min(self.pot / 50.0, 1.0)
        btc_norm = min(self._bet_to_call.get(agent, 0) / 20.0, 1.0)

        # Pot odds: ratio of bet-to-call vs pot size (what a human player would consider)
        btc_raw = self._bet_to_call.get(agent, 0)
        pot_odds = min(btc_raw / max(1, self.pot), 1.0)

        # Phase one-hot
        phase_vec = [0.0, 0.0, 0.0]
        if self.phase == PHASE_PRE_DRAW:
            phase_vec[0] = 1.0
        elif self.phase == PHASE_DRAW:
            phase_vec[1] = 1.0
        elif self.phase == PHASE_POST_DRAW:
            phase_vec[2] = 1.0

        # Opponent's draw count from current hand (-1.0 if draw hasn't happened)
        other = "player_1" if agent == "player_0" else "player_0"
        if self._draw_acted[other]:
            opp_draw = self.draw_counts.get(other, 0) / 5.0
        else:
            opp_draw = -1.0

        # Strategic features
        position = 1.0 if self._betting_order[0] == agent else 0.0
        raises = min(self.raises_this_hand.get(agent, 0) / 4.0, 1.0)
        opp_aggression = 1.0 if (agent == "player_0" and self.opp_bet_this_round) else 0.0
        opp_raised_pre = 1.0 if self.raised_pre_draw.get(other, False) else 0.0

        obs = hand_norm + [hand_category_norm, hand_score_norm, pot_norm, btc_norm, pot_odds] + phase_vec + [opp_draw, position, raises, opp_aggression, opp_raised_pre]

        if self.use_implicit_modeling:
            stats = self._get_rolling_stats()
            obs.extend(stats)
        else:
            # One-hot opponent ID
            opp_id_vec = [0.0] * NUM_ARCHETYPES
            if self.current_opponent is not None:
                opp_id_vec[self.current_opponent.opponent_id] = 1.0
            obs.extend(opp_id_vec)

        return np.array(obs, dtype=np.float32)

    def _build_action_mask(self, agent: str) -> np.ndarray:
        """Return a binary mask of legal actions."""
        mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)

        if self.phase in (PHASE_PRE_DRAW, PHASE_POST_DRAW):
            # Betting actions legal
            mask[A_FOLD] = 1
            mask[A_CALL] = 1
            if self.raises_left > 0:
                mask[A_RAISE] = 1

        elif self.phase == PHASE_DRAW:
            # All 32 draw combinations legal
            mask[A_DRAW_START:A_DRAW_END] = 1

        return mask

    # -----------------------------------------------------------------------
    # Rolling statistics (for implicit modelling)
    # -----------------------------------------------------------------------

    def _record_opponent_history(self):
        """Record opponent's behavior this hand for rolling stats."""
        entry = {
            "draw_count": self.draw_counts.get("player_1", 0),
            "folded": self.folded.get("player_1", False),
            "phase_when_folded": self.phase if self.folded.get("player_1", False) else None,
            "post_draw_action": self._opp_post_draw_action,
            "opponent_id": self.current_opponent.opponent_id if self.current_opponent else -1,
            "vpip": self.total_invested.get("player_1", 0) > ANTE,
            "pfr": self.raised_pre_draw.get("player_1", False),
        }
        self._opp_history.append(entry)
        # Trim to rolling window
        if len(self._opp_history) > self.rolling_window:
            self._opp_history = self._opp_history[-self.rolling_window:]

    def _get_rolling_stats(self) -> List[float]:
        """Compute rolling statistics over the last N opponent hands.

        Returns 10 floats:
            0: Frequency of folding PRE-draw
            1: Frequency of folding POST-draw
            2: Frequency of raising post-draw
            3: Average cards drawn (normalised to 0‑1)
            4: Frequency of raising, then standing pat (draw 0)
            5: Frequency of folding after drawing 3 cards
            6: Frequency of raising after standing pat
            7: VPIP (Voluntary Put In Pot %)
            8: PFR (Pre-Flop Raise %)
            9: Aggression Factor (Raises / Calls)
        """
        if not self._opp_history:
            return [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.1, 0.5]

        n = len(self._opp_history)
        fold_pre = sum(1 for h in self._opp_history if h["phase_when_folded"] == PHASE_PRE_DRAW) / n
        fold_post = sum(1 for h in self._opp_history if h["phase_when_folded"] in (PHASE_DRAW, PHASE_POST_DRAW, PHASE_SHOWDOWN)) / n
        raise_post = sum(
            1 for h in self._opp_history if h.get("post_draw_action") == A_RAISE
        ) / n
        avg_draw = sum(h["draw_count"] for h in self._opp_history) / (n * 5.0)

        # Raise + stand pat correlation
        raise_standpat = sum(
            1 for h in self._opp_history
            if h.get("post_draw_action") == A_RAISE and h["draw_count"] == 0
        ) / max(1, n)

        # Fold after drawing 3
        draw3_hands = [h for h in self._opp_history if h["draw_count"] == 3]
        fold_after_draw3 = (
            sum(1 for h in draw3_hands if h["folded"]) / max(1, len(draw3_hands))
        )

        # Raise after standing pat
        standpat_hands = [h for h in self._opp_history if h["draw_count"] == 0]
        raise_after_standpat = (
            sum(1 for h in standpat_hands if h.get("post_draw_action") == A_RAISE)
            / max(1, len(standpat_hands))
        )

        vpip = sum(1 for h in self._opp_history if h.get("vpip", False)) / n
        pfr = sum(1 for h in self._opp_history if h.get("pfr", False)) / n

        # Aggression factor roughly = raises / calls
        total_raises = sum(1 for h in self._opp_history if h.get("post_draw_action") == A_RAISE)
        total_calls = sum(1 for h in self._opp_history if h.get("post_draw_action") == A_CALL)
        af = min(total_raises / max(1, total_calls) / 3.0, 1.0) # normalize to 0-1 range

        return [fold_pre, fold_post, raise_post, avg_draw, raise_standpat, fold_after_draw3, raise_after_standpat, vpip, pfr, af]

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def render(self):
        if self.render_mode == "human":
            print(f"\n{'='*50}")
            print(f"Phase: {self.phase} | Pot: {self.pot}")
            from card_utils import hand_str, hand_category_name
            for a in self.possible_agents:
                h = self.hands[a]
                cat = hand_category_name(h)
                print(f"  {a}: {hand_str(h)}  ({cat})")
            print(f"  Invested: {self.total_invested}")
            print(f"  Folded: {self.folded}")
            print(f"  Draws: {self.draw_counts}")
