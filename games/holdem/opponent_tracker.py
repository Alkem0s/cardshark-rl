"""
games/holdem/opponent_tracker.py — Dual-Timescale Bayesian Opponent Profiler for Texas Hold'em.

Tracks behavioral tendencies for all 4 opponents relative to Hero (+1 to +4 clockwise):
1. Macro Lifetime Trackers (Beta-Binomial conjugate priors):
   - VPIP (Voluntarily Put $ in Pot)
   - PFR (Preflop Raise)
   - AF (Postflop Aggression Factor)
   - Fold to Bet / Pressure (C-Bet / Raise folding)
2. Micro Recency Trackers (Exponential Decay lambda = 0.90):
   - Sudden strategy shifts / Tilt detection
3. Clockwise Relative Seating:
   - Positions relative to Hero (+1 = Direct Left / SB, +2, +3, +4 = Button)
   - Guarantees position-invariant generalization.

Outputs 32 bounded [-1.0, 1.0] features (4 opponents x 8 features).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

NUM_OPPONENT_FEATURES = 32  # 4 opponents * 8 features


class HoldemBetaBinomialTracker:
    """Tracks a binary habit with Beta(alpha, beta) conjugate updating."""
    def __init__(self, prior_mean: float, prior_weight: float = 4.0):
        self.alpha0 = float(prior_mean * prior_weight)
        self.beta0 = float((1.0 - prior_mean) * prior_weight)
        self.successes = 0
        self.trials = 0

    def update(self, success: bool):
        self.trials += 1
        if success:
            self.successes += 1

    @property
    def posterior_mean(self) -> float:
        return (self.alpha0 + self.successes) / (self.alpha0 + self.beta0 + self.trials)

    def reset(self):
        self.successes = 0
        self.trials = 0


class HoldemDecayingTracker:
    """Tracks dynamic short-term recency shifts with exponential forgetting."""
    def __init__(self, prior_mean: float, prior_weight: float = 2.0, decay_rate: float = 0.90):
        self.alpha0 = float(prior_mean * prior_weight)
        self.beta0 = float((1.0 - prior_mean) * prior_weight)
        self.decay_rate = float(decay_rate)
        self.s_decayed = 0.0
        self.n_decayed = 0.0

    def update(self, success: bool):
        self.n_decayed = self.decay_rate * self.n_decayed + 1.0
        self.s_decayed = self.decay_rate * self.s_decayed + (1.0 if success else 0.0)

    @property
    def posterior_mean(self) -> float:
        total_weight = self.alpha0 + self.beta0 + self.n_decayed
        if total_weight <= 0.0:
            return 0.5
        return (self.alpha0 + self.s_decayed) / total_weight

    def reset(self):
        self.s_decayed = 0.0
        self.n_decayed = 0.0


class HoldemSeatProfile:
    """Tracks all behavioral habits for a single physical table seat."""
    def __init__(self, seat_idx: int):
        self.seat_idx = seat_idx

        # 1. Macro Lifetime Trackers
        self.vpip_tracker = HoldemBetaBinomialTracker(prior_mean=0.28, prior_weight=4.0)
        self.pfr_tracker = HoldemBetaBinomialTracker(prior_mean=0.18, prior_weight=4.0)
        self.af_tracker = HoldemBetaBinomialTracker(prior_mean=0.35, prior_weight=4.0)
        self.fold_pressure_tracker = HoldemBetaBinomialTracker(prior_mean=0.50, prior_weight=4.0)

        # 2. Micro Recency Trackers (Horizon ~8-10 hands)
        self.micro_vpip = HoldemDecayingTracker(prior_mean=0.28, prior_weight=2.0, decay_rate=0.90)
        self.micro_pfr = HoldemDecayingTracker(prior_mean=0.18, prior_weight=2.0, decay_rate=0.90)
        self.micro_af = HoldemDecayingTracker(prior_mean=0.35, prior_weight=2.0, decay_rate=0.90)
        self.micro_fold = HoldemDecayingTracker(prior_mean=0.50, prior_weight=2.0, decay_rate=0.90)

        self.hands_observed = 0

    def record_hand(
        self,
        vpip: bool,
        pfr: bool,
        postflop_bet: Optional[bool] = None,
        faced_bet_and_folded: Optional[bool] = None,
    ):
        self.hands_observed += 1
        self.vpip_tracker.update(vpip)
        self.pfr_tracker.update(pfr)
        self.micro_vpip.update(vpip)
        self.micro_pfr.update(pfr)

        if postflop_bet is not None:
            self.af_tracker.update(postflop_bet)
            self.micro_af.update(postflop_bet)

        if faced_bet_and_folded is not None:
            self.fold_pressure_tracker.update(faced_bet_and_folded)
            self.micro_fold.update(faced_bet_and_folded)

    @property
    def tilt_delta(self) -> float:
        """Delta between micro-recent and macro-lifetime aggression [-1.0, 1.0]."""
        d_vpip = self.micro_vpip.posterior_mean - self.vpip_tracker.posterior_mean
        d_pfr = self.micro_pfr.posterior_mean - self.pfr_tracker.posterior_mean
        d_af = self.micro_af.posterior_mean - self.af_tracker.posterior_mean
        return float(np.clip(d_vpip + d_pfr + d_af, -1.0, 1.0))

    def reset(self):
        self.vpip_tracker.reset()
        self.pfr_tracker.reset()
        self.af_tracker.reset()
        self.fold_pressure_tracker.reset()
        self.micro_vpip.reset()
        self.micro_pfr.reset()
        self.micro_af.reset()
        self.micro_fold.reset()
        self.hands_observed = 0


class HoldemTableOpponentTracker:
    """Manages all opponent profiles and builds relative feature vectors for Hero."""
    def __init__(self, num_seats: int = 5):
        self.num_seats = num_seats
        self.profiles: Dict[int, HoldemSeatProfile] = {
            i: HoldemSeatProfile(i) for i in range(num_seats)
        }

    def record_seat_hand(
        self,
        seat_idx: int,
        vpip: bool,
        pfr: bool,
        postflop_bet: Optional[bool] = None,
        faced_bet_and_folded: Optional[bool] = None,
    ):
        if seat_idx in self.profiles:
            self.profiles[seat_idx].record_hand(
                vpip=vpip,
                pfr=pfr,
                postflop_bet=postflop_bet,
                faced_bet_and_folded=faced_bet_and_folded,
            )

    def get_relative_vector(
        self,
        hero_seat: int,
        seats: List[Any],
        total_chips: int,
    ) -> np.ndarray:
        """
        Builds a 32-dimensional vector (4 opponents x 8 features) bounded in [-1.0, 1.0].
        Features per opponent offset k in [1, 2, 3, 4]:
        0: is_alive (1.0 or -1.0)
        1: in_hand (1.0 or -1.0)
        2: stack_ratio [-1.0, 1.0]
        3: vpip [-1.0, 1.0]
        4: pfr [-1.0, 1.0]
        5: af [-1.0, 1.0]
        6: fold_to_bet [-1.0, 1.0]
        7: tilt_delta [-1.0, 1.0]
        """
        vec = np.full(NUM_OPPONENT_FEATURES, -1.0, dtype=np.float32)
        idx = 0

        for offset in range(1, self.num_seats):
            s_idx = (hero_seat + offset) % self.num_seats
            if s_idx < len(seats):
                seat = seats[s_idx]
                prof = self.profiles.get(s_idx)

                if seat.is_alive and seat.chips > 0:
                    vec[idx] = 1.0
                    vec[idx + 1] = 1.0 if not seat.folded else -1.0
                    stack_r = min(1.0, float(seat.chips) / max(1.0, float(total_chips)))
                    vec[idx + 2] = float((stack_r * 2.0) - 1.0)

                    if prof:
                        vec[idx + 3] = float((prof.vpip_tracker.posterior_mean * 2.0) - 1.0)
                        vec[idx + 4] = float((prof.pfr_tracker.posterior_mean * 2.0) - 1.0)
                        vec[idx + 5] = float((prof.af_tracker.posterior_mean * 2.0) - 1.0)
                        vec[idx + 6] = float((prof.fold_pressure_tracker.posterior_mean * 2.0) - 1.0)
                        vec[idx + 7] = float(prof.tilt_delta)
                    else:
                        vec[idx + 3 : idx + 8] = 0.0
                else:
                    # Busted player
                    vec[idx : idx + 8] = -1.0
            idx += 8

        return vec

    def reset_all(self):
        for p in self.profiles.values():
            p.reset()
