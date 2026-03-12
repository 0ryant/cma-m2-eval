"""
replay_buffer.py  —  P0.4
===========================
PEACETIME-stratified experience replay buffer for the online MLP world model.

Prevents catastrophic forgetting of PEACETIME dynamics when CATASTROPHE episodes
overwrite recent experience. §36.13 V1 — required before any Regime 2 run.

Design choice: Variant 2 (stratified replay) over Variant 1 (EWC) per A.89 recommendation.
  - Simpler: no Fisher information estimation required
  - Transparent: PEACETIME fraction is an explicit hyperparameter, not implicit in λ
  - Debuggable: regime_retention_score directly measures what it guards against

Cold-start decay gate (P1.4):
  world_model_error ≈ 0.90 at t=0
  Target: < 0.30 by tick 150 in PEACETIME
  Target: < 0.50 by tick 100 in REGIME_2 STRESS
  If gate missed: call double_peacetime_fraction() then retest.

§ references: M2 v1.18 §36.13 V1, §36.2, CMA v4.1 §36.2
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from collections import deque
import random
import math


# ──────────────────────────────────────────────────────────────
# Regime labels
# ──────────────────────────────────────────────────────────────

class Regime:
    PEACETIME   = "PEACETIME"
    STRESS      = "STRESS"
    CATASTROPHE = "CATASTROPHE"
    RECOVERY    = "RECOVERY"

    # Treated as PEACETIME-equivalent for retention purposes
    PEACETIME_EQUIVALENT = {PEACETIME, RECOVERY}


# ──────────────────────────────────────────────────────────────
# Transition experience tuple
# ──────────────────────────────────────────────────────────────

@dataclass
class Transition:
    """
    One world-model training example.
    state_t + action → state_t1 is what the MLP must learn to predict.
    """
    state_t:   List[float]    # flattened agent state vector at tick t
    action:    int             # action taken
    state_t1:  List[float]    # resulting state at tick t+1
    regime:    str             # Regime label — used for stratified sampling
    tick:      int
    agent_id:  str
    loss:      Optional[float] = None  # prediction error at insertion time (for prioritised replay)


# ──────────────────────────────────────────────────────────────
# Stratified replay buffer
# ──────────────────────────────────────────────────────────────

class StratifiedReplayBuffer:
    """
    Fixed-size buffer with separate PEACETIME and non-PEACETIME pools.
    Sampling always includes at least `min_peacetime_fraction` PEACETIME transitions.

    Default: 500 episodes, 30% PEACETIME minimum per SGD batch.
    If CATASTROPHE epochs overwrite PEACETIME data, the minimum fraction
    prevents complete forgetting.

    Usage:
        buf = StratifiedReplayBuffer(capacity=500, min_peacetime_fraction=0.30)
        buf.add(transition)
        batch = buf.sample(batch_size=64)
        # → feed batch to world model SGD step
    """

    def __init__(
        self,
        capacity: int = 500,
        min_peacetime_fraction: float = 0.30,
        seed: int = 42,
    ):
        self.capacity = capacity
        self.min_peacetime_fraction = min_peacetime_fraction
        self._rng = random.Random(seed)

        # Two pools — separate capacities, proportional to fraction
        self._peacetime_capacity = max(10, int(capacity * min_peacetime_fraction * 2))
        self._other_capacity = capacity - self._peacetime_capacity

        self._peacetime: deque[Transition] = deque(maxlen=self._peacetime_capacity)
        self._other:     deque[Transition] = deque(maxlen=self._other_capacity)

        self._total_added = 0
        self._peacetime_added = 0

    def add(self, t: Transition) -> None:
        if t.regime in Regime.PEACETIME_EQUIVALENT:
            self._peacetime.append(t)
            self._peacetime_added += 1
        else:
            self._other.append(t)
        self._total_added += 1

    def sample(self, batch_size: int) -> List[Transition]:
        """
        Stratified sample: guarantee ≥ min_peacetime_fraction from PEACETIME pool.
        Falls back to available if pools are underpopulated.
        """
        n_peace  = max(1, int(math.ceil(batch_size * self.min_peacetime_fraction)))
        n_other  = batch_size - n_peace

        # Draw from PEACETIME pool
        peace_pool = list(self._peacetime)
        if len(peace_pool) < n_peace:
            peace_sample = peace_pool  # take all we have
        else:
            peace_sample = self._rng.sample(peace_pool, n_peace)

        # Draw from other pool
        other_pool = list(self._other)
        n_other_actual = min(n_other, len(other_pool))
        other_sample = self._rng.sample(other_pool, n_other_actual) if n_other_actual > 0 else []

        # If either pool is short, pad from the other
        shortfall = batch_size - len(peace_sample) - len(other_sample)
        if shortfall > 0:
            combined = peace_pool + other_pool
            if combined:
                pad = self._rng.choices(combined, k=shortfall)
                other_sample.extend(pad)

        batch = peace_sample + other_sample
        self._rng.shuffle(batch)
        return batch

    def regime_retention_score(
        self,
        world_model_fn,
        peacetime_test_transitions: List[Transition],
    ) -> float:
        """
        Compute world_model_regime_retention_score.
        Measures whether model accuracy on PEACETIME test transitions has degraded
        after CATASTROPHE exposure. Returns mean L2 prediction error on test set.

        Wire to §36.6 telemetry stream as world_model_regime_retention_score.

        Args:
            world_model_fn: callable(state_t, action) → predicted_state_t1
            peacetime_test_transitions: held-out PEACETIME transitions (separate from buffer)
        """
        if not peacetime_test_transitions:
            return float("nan")
        errors = []
        for t in peacetime_test_transitions:
            pred = world_model_fn(t.state_t, t.action)
            err = sum((p - a) ** 2 for p, a in zip(pred, t.state_t1)) / max(1, len(t.state_t1))
            errors.append(err)
        return sum(errors) / len(errors)

    def double_peacetime_fraction(self) -> None:
        """
        Cold-start decay gate miss remediation (P1.4 gate).
        Call if world_model_error fails the decay gate. Doubles PEACETIME minimum.
        """
        new_fraction = min(0.60, self.min_peacetime_fraction * 2)
        print(f"[ReplayBuffer] Doubling PEACETIME fraction: {self.min_peacetime_fraction:.0%} → {new_fraction:.0%}")
        self.min_peacetime_fraction = new_fraction
        # Rebuild capacity allocation
        self._peacetime_capacity = max(10, int(self.capacity * new_fraction * 2))
        self._other_capacity = self.capacity - self._peacetime_capacity
        # Prune other pool if needed
        while len(self._other) > self._other_capacity:
            self._other.popleft()

    def stats(self) -> Dict[str, object]:
        return {
            "total_added":           self._total_added,
            "peacetime_added":       self._peacetime_added,
            "peacetime_in_buffer":   len(self._peacetime),
            "other_in_buffer":       len(self._other),
            "peacetime_capacity":    self._peacetime_capacity,
            "other_capacity":        self._other_capacity,
            "min_peacetime_fraction": self.min_peacetime_fraction,
            "actual_peacetime_fraction": (
                len(self._peacetime) / max(1, len(self._peacetime) + len(self._other))
            ),
        }


# ──────────────────────────────────────────────────────────────
# Cold-start decay gate checker
# ──────────────────────────────────────────────────────────────

class ColdStartDecayGate:
    """
    Enforces the cold-start decay gate requirement (P1.4).

    Gate 1 (Regime 1 PEACETIME):  error < 0.30 by tick 150
    Gate 2 (Regime 2 STRESS):     error < 0.50 by tick 100

    Call check() each tick with current mean world_model_error.
    Call report() after Regime 1 completes to determine if Regime 2 is unblocked.
    """

    def __init__(self):
        self.error_by_tick: Dict[int, float] = {}
        self.regime_1_gate_passed: Optional[bool] = None
        self.regime_2_gate_passed: Optional[bool] = None

    def record(self, tick: int, mean_error: float) -> None:
        self.error_by_tick[tick] = mean_error

    def check_regime_1_gate(self) -> Tuple[bool, str]:
        """Check: world_model_error < 0.30 by tick 150 in PEACETIME."""
        if 150 not in self.error_by_tick:
            # Find nearest tick
            ticks_at_or_after_150 = [t for t in self.error_by_tick if t >= 150]
            if not ticks_at_or_after_150:
                return False, "Tick 150 not yet reached — gate cannot be evaluated."
            tick = min(ticks_at_or_after_150)
        else:
            tick = 150
        err = self.error_by_tick[tick]
        passed = err < 0.30
        self.regime_1_gate_passed = passed
        if passed:
            return True, f"Gate 1 PASSED: world_model_error = {err:.4f} < 0.30 at tick {tick}"
        return False, (
            f"Gate 1 FAILED: world_model_error = {err:.4f} ≥ 0.30 at tick {tick}. "
            f"Regime 2 is BLOCKED. "
            f"Fix: call buffer.double_peacetime_fraction() and rerun Regime 1."
        )

    def check_regime_2_gate(self, tick_100_error: float) -> Tuple[bool, str]:
        """Check: world_model_error < 0.50 by tick 100 in Regime 2 STRESS."""
        passed = tick_100_error < 0.50
        self.regime_2_gate_passed = passed
        if passed:
            return True, f"Gate 2 PASSED: world_model_error = {tick_100_error:.4f} < 0.50 at tick 100"
        return False, (
            f"Gate 2 FAILED: world_model_error = {tick_100_error:.4f} ≥ 0.50 at tick 100. "
            f"Regime 3 is BLOCKED. Fix: increase EWC λ or PEACETIME fraction, rerun."
        )

    def decay_curve_summary(self) -> str:
        if not self.error_by_tick:
            return "No data recorded."
        ticks = sorted(self.error_by_tick.keys())
        lines = [f"  t={t:4d}: error={self.error_by_tick[t]:.4f}" for t in ticks[::10]]
        return "\n".join(lines)
