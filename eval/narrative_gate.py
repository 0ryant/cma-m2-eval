"""
narrative_gate.py  —  P0.6 Motif 5 (wire before Regime 1)
============================================================
Narrative fragmentation → goal instability feedback loop.

Gate: narrative_coherence_score ≥ 0.50 required on any goal_stack update.
Updates below threshold are queued, not applied, until coherence recovers.

This is STUB → WIRED promotion for Motif 5 in causal_graph.md.
Wire into the sim engine's goal_stack update path before first Regime 1 run.

§ references: CMA v4.1 §36.11 (Motif 5), §M2.12.4 (DEPRESSIVE_LOCK_IN),
              causal_graph.md
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from collections import deque
import math


# ──────────────────────────────────────────────────────────────
# Goal item
# ──────────────────────────────────────────────────────────────

@dataclass
class Goal:
    goal_id:         str
    goal_type:       str            # SOCIAL | RELATIONAL | INVESTIGATIVE | SURVIVAL | etc.
    priority:        float          # 0.0–1.0
    progress_ratio:  float          # 0.0–1.0 (goal.progress_ratio * 2 - 1 = primary_goal_valence)
    expected_utility: float
    tick_created:    int


# ──────────────────────────────────────────────────────────────
# Narrative coherence scorer
# ──────────────────────────────────────────────────────────────

class NarrativeCoherenceScorer:
    """
    Lightweight goal-stack coherence measure.

    Coherence = smoothness of goal priority transitions over time.
    High coherence: goal priorities change gradually (stable narrative arc).
    Low coherence: goals flip rapidly or contradict each other.

    Implementation: moving variance of goal_stack[0].priority over the last W ticks.
    Coherence = 1 - min(1, variance / variance_ceiling)

    In the full engine: replace with the semantic coherence module that computes
    narrative_coherence_score from the belief graph and goal history.
    """

    def __init__(
        self,
        window_ticks:        int   = 20,
        variance_ceiling:    float = 0.10,
        coherence_threshold: float = 0.50,
    ):
        self.window_ticks        = window_ticks
        self.variance_ceiling    = variance_ceiling
        self.coherence_threshold = coherence_threshold
        self._priority_history: deque = deque(maxlen=window_ticks)
        self._baseline: Optional[float] = None   # set from first N ticks

    def update(self, goal_stack: List[Goal]) -> float:
        """
        Call once per tick with the current goal_stack.
        Returns: coherence_score ∈ [0.0, 1.0]
        """
        priority = goal_stack[0].priority if goal_stack else 0.5
        self._priority_history.append(priority)

        if len(self._priority_history) < 2:
            return 1.0

        values = list(self._priority_history)
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
        coherence = 1.0 - min(1.0, variance / max(self.variance_ceiling, 1e-8))

        # Set baseline from first window (phenotype reference)
        if self._baseline is None and len(self._priority_history) >= self.window_ticks:
            self._baseline = coherence

        return float(coherence)

    @property
    def baseline(self) -> float:
        return self._baseline if self._baseline is not None else 1.0

    def is_coherent(self, score: Optional[float] = None) -> bool:
        if score is None:
            score = self.update([])
        return score >= self.coherence_threshold


# ──────────────────────────────────────────────────────────────
# Gated goal stack
# ──────────────────────────────────────────────────────────────

@dataclass
class PendingGoalUpdate:
    goal:       Goal
    tick_queued: int
    reason:     str   # why it was queued (for trace)


class GatedGoalStack:
    """
    Goal stack with narrative coherence gate (Motif 5).

    On each proposed update:
      - Compute narrative_coherence_score
      - If score ≥ 0.50: apply immediately
      - If score < 0.50: queue update, emit narrative_fragmentation_score telemetry
      - Each tick: re-check pending updates against current coherence

    Wire into sim engine:
        goal_stack = GatedGoalStack(agent_id=agent_id, telemetry=tel)
        goal_stack.propose_update(new_goal, tick, scorer)  # called by planner
        goal_stack.tick_step(tick, scorer)                 # called each tick
    """

    def __init__(
        self,
        agent_id:          str,
        max_goals:         int = 5,
        max_pending:       int = 10,
        telemetry=None,    # TelemetryEmitter — optional (avoid circular import)
    ):
        self.agent_id = agent_id
        self.max_goals = max_goals
        self._stack:   List[Goal] = []
        self._pending: deque = deque(maxlen=max_pending)
        self.tel = telemetry
        self._narrative_fragmentation_score: float = 0.0

    @property
    def stack(self) -> List[Goal]:
        return self._stack

    @property
    def narrative_fragmentation_score(self) -> float:
        return self._narrative_fragmentation_score

    def propose_update(
        self,
        goal:    Goal,
        tick:    int,
        scorer:  NarrativeCoherenceScorer,
    ) -> bool:
        """
        Propose adding or replacing a goal.
        Returns True if applied immediately, False if queued.
        """
        coherence = scorer.update(self._stack)
        self._narrative_fragmentation_score = 1.0 - coherence

        if coherence >= scorer.coherence_threshold:
            self._apply(goal)
            return True
        else:
            # Queue — do not apply yet
            self._pending.append(PendingGoalUpdate(
                goal=goal,
                tick_queued=tick,
                reason=f"coherence={coherence:.3f} < threshold={scorer.coherence_threshold}",
            ))
            # Emit telemetry if wired
            if self.tel is not None:
                # narrative_fragmentation_score added to per-tick stream
                # (wired via TickRecord.narrative_coherence in telemetry.py)
                pass
            return False

    def tick_step(
        self,
        tick:    int,
        scorer:  NarrativeCoherenceScorer,
    ) -> int:
        """
        Called once per tick. Re-checks pending updates.
        Returns: number of queued updates applied this tick.
        """
        coherence = scorer.update(self._stack)
        self._narrative_fragmentation_score = 1.0 - coherence
        applied = 0

        if coherence >= scorer.coherence_threshold and self._pending:
            # Apply oldest pending update
            update = self._pending.popleft()
            self._apply(update.goal)
            applied += 1

        return applied

    def _apply(self, goal: Goal) -> None:
        """Insert goal into stack by priority, evict lowest if full."""
        self._stack.append(goal)
        self._stack.sort(key=lambda g: -g.priority)
        if len(self._stack) > self.max_goals:
            self._stack = self._stack[:self.max_goals]

    def primary_goal_valence(self) -> float:
        """§M2.11.4: goal_stack[0].progress_ratio × 2 − 1"""
        if not self._stack:
            return 0.0
        return float(self._stack[0].progress_ratio * 2.0 - 1.0)

    def pending_count(self) -> int:
        return len(self._pending)

    def queue_summary(self) -> str:
        if not self._pending:
            return "No pending updates."
        return "\n".join(
            f"  [{u.tick_queued}] goal={u.goal.goal_id} priority={u.goal.priority:.2f}  reason: {u.reason}"
            for u in self._pending
        )


# ──────────────────────────────────────────────────────────────
# Integration validator — confirms Motif 5 is wired
# ──────────────────────────────────────────────────────────────

def validate_motif5_wired(
    goal_stack: GatedGoalStack,
    scorer:     NarrativeCoherenceScorer,
    sim_ticks:  int = 30,
) -> Dict[str, Any]:
    """
    Smoke test: inject a low-coherence scenario and verify the gate blocks updates.
    Call before Regime 1. Returns pass/fail result.

    Usage in pre-flight check:
        result = validate_motif5_wired(goal_stack, scorer)
        assert result['gate_blocked_updates'] > 0, 'Motif 5 gate not functioning'
    """
    import random
    rng = random.Random(42)

    blocked = 0
    applied = 0

    for tick in range(sim_ticks):
        # Simulate high-fragmentation scenario (rapidly changing priorities)
        goal = Goal(
            goal_id=f"goal_{tick}",
            goal_type="SOCIAL",
            priority=rng.random(),       # chaotic priority changes → low coherence
            progress_ratio=rng.random(),
            expected_utility=1.0,
            tick_created=tick,
        )
        ok = goal_stack.propose_update(goal, tick, scorer)
        if ok:
            applied += 1
        else:
            blocked += 1
        goal_stack.tick_step(tick, scorer)

    result = {
        "gate_blocked_updates": blocked,
        "gate_applied_updates": applied,
        "final_fragmentation_score": goal_stack.narrative_fragmentation_score,
        "motif5_wired": blocked > 0,
    }

    print(f"\n── Motif 5 Narrative Gate Validation ──────────────")
    print(f"  Blocked: {blocked}  Applied: {applied}  Fragmentation: {goal_stack.narrative_fragmentation_score:.3f}")
    print(f"  Status: {'WIRED ✓' if result['motif5_wired'] else 'FAULT ✗ — gate never blocked'}")

    return result
