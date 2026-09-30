"""
opponent_tracker.py — Model B Bayesian Opponent Profiler for 5-Seat Poker.

Uses Beta-Binomial conjugate updating to solve the cold-start problem (no 30-50 hand delay).
Provides clockwise relative seating alignment (+1 to +4) so the RL model is position-agnostic.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np


class BetaBinomialTracker:
    """Tracks a binary habit (e.g. VPIP, PFR) with a Beta(alpha, beta) conjugate prior."""
    def __init__(self, prior_mean: float, prior_weight: float = 5.0):
        # alpha0 + beta0 = prior_weight
        # alpha0 / (alpha0 + beta0) = prior_mean
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
        """E[theta] = (alpha0 + k) / (alpha0 + beta0 + n)"""
        return (self.alpha0 + self.successes) / (self.alpha0 + self.beta0 + self.trials)

    @property
    def posterior_variance(self) -> float:
        """Var[theta] for uncertainty estimation."""
        a = self.alpha0 + self.successes
        b = self.beta0 + (self.trials - self.successes)
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def reset(self):
        self.successes = 0
        self.trials = 0


class DecayingBetaBinomialTracker:
    """Tracks dynamic short-term habit shifts with exponential forgetting (lambda ~ 0.92).

    Effective half-life is ~10 hands:
        N_t = lambda * N_{t-1} + 1.0
        K_t = lambda * K_{t-1} + (1.0 if success else 0.0)
    Posterior mean: (alpha0 + K_t) / (alpha0 + beta0 + N_t)
    """
    def __init__(self, prior_mean: float, prior_weight: float = 3.0, decay_rate: float = 0.92):
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

    @property
    def posterior_variance(self) -> float:
        a = self.alpha0 + self.s_decayed
        b = self.beta0 + (self.n_decayed - self.s_decayed)
        total = a + b
        if total <= 0.0:
            return 0.0
        return (a * b) / (total ** 2 * (total + 1.0))

    def reset(self):
        self.s_decayed = 0.0
        self.n_decayed = 0.0


class SeatProfile:
    """Tracks all behavioral habits for a single physical table seat using dual-timescale Bayesian models."""
    def __init__(self, seat_idx: int):
        self.seat_idx = seat_idx

        # 1. Lifetime Macro Trackers (Beta-Binomial, prior weight 5.0)
        self.vpip_tracker = BetaBinomialTracker(prior_mean=0.28, prior_weight=5.0)
        self.pfr_tracker = BetaBinomialTracker(prior_mean=0.15, prior_weight=5.0)
        self.af_tracker = BetaBinomialTracker(prior_mean=0.30, prior_weight=5.0)
        self.fold_pressure_tracker = BetaBinomialTracker(prior_mean=0.50, prior_weight=5.0)

        # 2. Dynamic Micro Recency Trackers (Exponential Decay lambda = 0.90, horizon ~10 hands)
        self.micro_vpip_tracker = DecayingBetaBinomialTracker(prior_mean=0.28, prior_weight=2.0, decay_rate=0.90)
        self.micro_pfr_tracker = DecayingBetaBinomialTracker(prior_mean=0.15, prior_weight=2.0, decay_rate=0.90)
        self.micro_af_tracker = DecayingBetaBinomialTracker(prior_mean=0.30, prior_weight=2.0, decay_rate=0.90)
        self.micro_fold_tracker = DecayingBetaBinomialTracker(prior_mean=0.50, prior_weight=2.0, decay_rate=0.90)

        # 3. Average cards drawn (normalized 0 to 1): prior ~0.40 (2 cards drawn)
        self.prior_draw = 0.40
        self.draw_prior_weight = 5.0
        self.draw_sum = 0.0
        self.draw_count = 0

        self.hands_observed = 0

    def record_hand(
        self,
        vpip: bool,
        pfr: bool,
        post_draw_raised: Optional[bool] = None,
        faced_raise_and_folded: Optional[bool] = None,
        cards_drawn: Optional[int] = None,
    ):
        """Update posterior beliefs from actions observed in a hand."""
        self.hands_observed += 1

        # Update macro lifetime trackers
        self.vpip_tracker.update(vpip)
        self.pfr_tracker.update(pfr)

        # Update micro recency trackers
        self.micro_vpip_tracker.update(vpip)
        self.micro_pfr_tracker.update(pfr)

        if post_draw_raised is not None:
            self.af_tracker.update(post_draw_raised)
            self.micro_af_tracker.update(post_draw_raised)

        if faced_raise_and_folded is not None:
            self.fold_pressure_tracker.update(faced_raise_and_folded)
            self.micro_fold_tracker.update(faced_raise_and_folded)

        if cards_drawn is not None:
            self.draw_sum += (cards_drawn / 5.0)
            self.draw_count += 1

    @property
    def avg_draw_normalized(self) -> float:
        """Weighted mean cards drawn (0.0 to 1.0)."""
        total_weight = self.draw_prior_weight + self.draw_count
        return (self.prior_draw * self.draw_prior_weight + self.draw_sum) / total_weight

    def get_feature_vector(self) -> List[float]:
        """Returns 5 behavioral features [VPIP, PFR, AF, Fold_Pressure, Avg_Draw] for Model B compatibility."""
        return [
            float(np.clip(self.vpip_tracker.posterior_mean, 0.0, 1.0)),
            float(np.clip(self.pfr_tracker.posterior_mean, 0.0, 1.0)),
            float(np.clip(self.af_tracker.posterior_mean, 0.0, 1.0)),
            float(np.clip(self.fold_pressure_tracker.posterior_mean, 0.0, 1.0)),
            float(np.clip(self.avg_draw_normalized, 0.0, 1.0)),
        ]

    def get_recency_delta(self) -> float:
        """Returns the sudden strategy shift / tilt delta in [-1.0, 1.0].
        Positive indicates sudden aggressive tilt (higher VPIP + PFR + AF than lifetime).
        Negative indicates sudden tightening / freezing.
        """
        delta_vpip = self.micro_vpip_tracker.posterior_mean - self.vpip_tracker.posterior_mean
        delta_pfr = self.micro_pfr_tracker.posterior_mean - self.pfr_tracker.posterior_mean
        delta_af = self.micro_af_tracker.posterior_mean - self.af_tracker.posterior_mean
        combined = delta_vpip + delta_pfr + delta_af
        return float(np.clip(combined, -1.0, 1.0))


    def detect_tilt(self, threshold: float = 0.25) -> Tuple[bool, str]:
        """Detects whether this seat is undergoing a significant behavioral shift."""
        delta = self.get_recency_delta()
        if delta > threshold:
            return True, f"Aggressive Tilt (+{delta:.2f})"
        elif delta < -threshold:
            return True, f"Rock Freeze ({delta:.2f})"
        return False, "Stable"

    def classify_archetype(self) -> Tuple[str, str]:
        """Infers opponent archetype and returns (archetype_name, actionable_exploit_advice)."""
        vpip = self.vpip_tracker.posterior_mean
        pfr = self.pfr_tracker.posterior_mean
        af = self.af_tracker.posterior_mean
        fold = self.fold_pressure_tracker.posterior_mean

        if vpip > 0.45 and af < 0.22:
            return "Calling Station", "Never bluff; purely value-bet made pairs with larger sizing."
        elif pfr > 0.35 and af > 0.45:
            return "Maniac", "Induce bluffs by check-calling medium/strong hands; trap pre-draw."
        elif vpip < 0.22 and pfr < 0.15:
            return "Rock", "Steal blinds aggressively; fold to post-draw raises unless holding the nuts."
        elif vpip >= 0.20 and vpip <= 0.38 and pfr >= 0.16:
            return "TAG", "Solid player; respect 3-bets and rely on positional advantage."
        elif vpip > 0.35 and pfr > 0.20:
            return "LAG", "Aggressive wide range; re-raise pre-draw to isolate and call down lighter."
        else:
            return "Balanced", "Insufficient tell divergence; play baseline GTO strategy."

    def reset(self):
        self.vpip_tracker.reset()
        self.pfr_tracker.reset()
        self.af_tracker.reset()
        self.fold_pressure_tracker.reset()
        self.micro_vpip_tracker.reset()
        self.micro_pfr_tracker.reset()
        self.micro_af_tracker.reset()
        self.micro_fold_tracker.reset()
        self.draw_sum = 0.0
        self.draw_count = 0
        self.hands_observed = 0


class TableOpponentTracker:
    """Manages behavioral profiles for all table seats and provides Hero-centric alignment."""
    def __init__(self, num_seats: int = 5):
        self.num_seats = num_seats
        self.profiles: Dict[int, SeatProfile] = {
            i: SeatProfile(i) for i in range(num_seats)
        }

    def record_seat_hand(
        self,
        seat_idx: int,
        vpip: bool,
        pfr: bool,
        post_draw_raised: Optional[bool] = None,
        faced_raise_and_folded: Optional[bool] = None,
        cards_drawn: Optional[int] = None,
    ):
        if seat_idx in self.profiles:
            self.profiles[seat_idx].record_hand(
                vpip=vpip,
                pfr=pfr,
                post_draw_raised=post_draw_raised,
                faced_raise_and_folded=faced_raise_and_folded,
                cards_drawn=cards_drawn,
            )

    def get_relative_opponent_vector(
        self,
        hero_seat: int,
        is_alive: List[bool],
        in_hand: List[bool],
        stacks: List[int],
        investments: List[int],
        pot: int,
        draw_counts_this_hand: List[int],
    ) -> List[float]:
        """Builds a 40-dimensional vector representing the 4 opponents relative to Hero clockwise (+1 to +4).

        For each slot k in [1, 2, 3, 4]:
            seat = (hero_seat + k) % num_seats
            10 features per slot:
            0: is_alive (1.0 or 0.0)
            1: is_in_hand (1.0 or 0.0)
            2: stack_ratio (stack / total_chips)
            3: pot_share (investment / max(1, pot))
            4: draw_count_this_hand (draws / 5.0, or -1.0 if pre-draw)
            5: vpip_bayesian
            6: pfr_bayesian
            7: af_bayesian
            8: fold_rate_bayesian
            9: avg_draw_bayesian
        """
        total_chips = max(1, sum(stacks) + pot)
        features: List[float] = []

        for offset in range(1, self.num_seats):
            seat = (hero_seat + offset) % self.num_seats

            alive = 1.0 if (seat < len(is_alive) and is_alive[seat]) else 0.0
            in_h = 1.0 if (seat < len(in_hand) and in_hand[seat]) else 0.0

            if alive > 0.0:
                stack = float(stacks[seat]) if seat < len(stacks) else 0.0
                inv = float(investments[seat]) if seat < len(investments) else 0.0
                draw_this = float(draw_counts_this_hand[seat]) if seat < len(draw_counts_this_hand) else -1.0

                stack_ratio = float(np.clip(stack / total_chips, 0.0, 1.0))
                pot_share = float(np.clip(inv / max(1.0, float(pot)), 0.0, 1.0))
                draw_norm = float(draw_this / 5.0) if draw_this >= 0 else -1.0

                behavioral = self.profiles[seat].get_feature_vector()
                slot_features = [alive, in_h, stack_ratio, pot_share, draw_norm] + behavioral
            else:
                # Dead / Busted player: all features zeroed
                slot_features = [0.0] * 10

            features.extend(slot_features)

        return features

    def get_seat_recency_delta(self, seat_idx: int) -> float:
        """Returns the dynamic tilt / strategy shift delta for seat_idx."""
        if seat_idx in self.profiles:
            return self.profiles[seat_idx].get_recency_delta()
        return 0.0

    def reset_all(self):
        for profile in self.profiles.values():
            profile.reset()
