"""
telemetry.py  —  P0.1 + P0.2
==============================
Canonical telemetry definitions for the CMA / M2 simulation.

Two streams:
  1. Per-tick stream   — one record per agent per tick
  2. Event stream      — fired when state transitions occur

All fields are typed dataclasses. Consumers (reporters, the battery, the validator)
import from here. The simulation engine calls emit_tick() and emit_event() once wired.

§ references: M2 v1.18 §11.1 (per-tick), §11.2 (events), CMA v4.1 §36.6
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import List, Optional, Dict, Any
import json
import time


# ──────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────

class M2Family(str, Enum):
    DEFEND    = "DEFEND"
    WITHDRAW  = "WITHDRAW"
    REPAIR    = "REPAIR"
    EXPLORE   = "EXPLORE"
    DOMINATE  = "DOMINATE"
    SEEK_HELP = "SEEK_HELP"
    DECEIVE   = "DECEIVE"
    BASELINE  = "BASELINE"       # no family scored above threshold
    NONE      = "NONE"           # uninitialized


class M2Overlay(str, Enum):
    MOURN     = "MOURN"
    TEACH     = "TEACH"
    PLAY      = "PLAY"
    CONSERVE  = "CONSERVE"


class PrecedenceTag(str, Enum):
    OVERRIDE     = "OVERRIDE"      # PRIMITIVE_REFLEX or trigger_override fired
    SCORE_WIN    = "SCORE_WIN"     # policy_score selected family against competing pressure
    WEIGHTED_ARB = "WEIGHTED_ARB"  # U(a) arbitration resolved the action — correlational only


class ControllerState(str, Enum):
    PRIMITIVE_REFLEX = "PRIMITIVE_REFLEX"
    TRIGGER_OVERRIDE = "TRIGGER_OVERRIDE"
    BASELINE         = "BASELINE"
    POLICY_FAMILY    = "POLICY_FAMILY"
    WEIGHTED_ARB     = "WEIGHTED_ARB"


# ──────────────────────────────────────────────────────────────
# P0.1 — Per-Tick Stream
# §M2.11.1  All fields mandatory unless Optional.
# ──────────────────────────────────────────────────────────────

@dataclass
class TickRecord:
    """One record per agent per tick. Append to tick_log list or write to parquet."""

    # Simulation coordinates
    tick:               int
    agent_id:           str
    regime:             str           # PEACETIME | STRESS | CATASTROPHE | RECOVERY
    seed:               int

    # M2 policy state  (§M2.11.1 mandatory additions)
    active_policy_family:   M2Family
    policy_score_vector:    List[float]  # float[7], index order = M2Family enum order (excl BASELINE/NONE)
    switch_cost_paid:       bool
    switch_cost_magnitude:  float        # 0.0 if no switch this tick
    accessible_families:    List[M2Family]
    active_overlays:        List[M2Overlay]
    policy_conflict_detected: bool

    # Tactic
    tactic_class:           str          # semantic class label for CV(tactic_class) observable
    action_taken:           int          # action index

    # Precedence (§M2.7.6, P0.2)
    precedence_tag:         PrecedenceTag
    dominant_module:        str          # "M2_policy", "PRIMITIVE_REFLEX", "trigger_override", "U(a)"

    # Stress / regression
    regression_depth:       float        # rd ∈ [0.0, 1.0]

    # Overlay telemetry (BASELINE tracking for depressive lock-in)
    baseline_ticks_running:    int       # consecutive ticks in BASELINE state
    mourn_during_baseline_ticks: int     # ticks MOURN active while in BASELINE

    # Narrative coherence (for depressive lock-in alarm trigger)
    narrative_coherence:    float        # 0.0–1.0

    # World model
    world_model_error:      float        # L2 loss on held-out transitions

    # Goal
    primary_goal_valence:   float        # -1.0 to +1.0

    # Raw observation vector — stored for CF probe contexts in LSM battery (P2.4)
    # Optional: None in quick/smoke mode to save memory; set in publication runs.
    obs_raw:                Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['active_policy_family'] = self.active_policy_family.value
        d['accessible_families'] = [f.value for f in self.accessible_families]
        d['active_overlays'] = [o.value for o in self.active_overlays]
        d['precedence_tag'] = self.precedence_tag.value
        d['policy_score_vector'] = list(self.policy_score_vector)
        return d


# ──────────────────────────────────────────────────────────────
# P0.2 — Event Stream
# §M2.11.2  Events fired when state transitions occur.
# ──────────────────────────────────────────────────────────────

@dataclass
class PolicyTransitionEvent:
    """Fired when active_policy_family changes. §M2.11.2"""
    event_type:     str = "POLICY_TRANSITION"
    tick:           int = 0
    agent_id:       str = ""
    prev_family:    M2Family = M2Family.NONE
    new_family:     M2Family = M2Family.NONE
    cause:          str = ""     # "score_selection" | "trigger_override" | "stress_collapse" | "refractory_expired"
    switch_cost:    float = 0.0
    rd_at_switch:   float = 0.0


@dataclass
class TriggerOverrideEvent:
    """Fired when trigger_override fires (§5.2c). §M2.11.2"""
    event_type:      str = "TRIGGER_OVERRIDE"
    tick:            int = 0
    agent_id:        str = ""
    override_type:   str = ""    # e.g. "ATTACHMENT_THREAT", "SURVIVAL_THREAT", "MORAL_VIOLATION"
    suppressed_family: M2Family = M2Family.NONE
    override_to_family: M2Family = M2Family.NONE


@dataclass
class PrimitiveReflexEvent:
    """Fired when PRIMITIVE_REFLEX overrides all family scoring. §M2.11.2"""
    event_type:      str = "PRIMITIVE_REFLEX"
    tick:            int = 0
    agent_id:        str = ""
    trigger:         str = ""    # what caused it
    suppressed_family: M2Family = M2Family.NONE


@dataclass
class PrecedenceTagEvent:
    """
    P0.2 — Fired on EVERY action. The source of truth for §M2.7.6 publication rule.
    OVERRIDE + SCORE_WIN fraction must be > 0.70 for Tier A mechanism claim.
    """
    event_type:      str = "PRECEDENCE_TAG"
    tick:            int = 0
    agent_id:        str = ""
    event_tag:       PrecedenceTag = PrecedenceTag.WEIGHTED_ARB
    dominant_module: str = ""
    action_taken:    int = 0
    family_active:   M2Family = M2Family.NONE
    rd_at_action:    float = 0.0


@dataclass
class DepressionLockInEvent:
    """
    P0.2 — Fired when depressive lock-in pattern detected. §M2.12.4, §CMA.36.6.
    Trigger: BASELINE_ticks > 20 AND mourn_during_baseline_ticks > 15
              AND narrative_coherence < 0.70 of phenotype baseline.
    """
    event_type:              str = "DEPRESSIVE_LOCK_IN"
    tick:                    int = 0
    agent_id:                int = 0
    BASELINE_ticks_count:    int = 0
    mourn_ticks_count:       int = 0
    narrative_coherence_ratio: float = 0.0   # current / phenotype_baseline


@dataclass
class FeedbackLoopAlertEvent:
    """Fired when feedback_loop_gain exceeds instability threshold (§36.11)."""
    event_type:    str = "FEEDBACK_LOOP_ALERT"
    tick:          int = 0
    agent_id:      str = ""
    motif:         str = ""    # e.g. "FEAR_AMPLIFICATION", "SHAME_CONCEALMENT"
    gain_estimate: float = 0.0
    attractor_lock: bool = False


# ──────────────────────────────────────────────────────────────
# Emitter — thin wrapper used by the simulation engine
# ──────────────────────────────────────────────────────────────

class TelemetryEmitter:
    """
    Collect tick records and events in memory.
    In production: swap append() calls for writes to parquet / protobuf stream.

    Usage in sim engine:
        tel = TelemetryEmitter()
        tel.tick(TickRecord(...))
        tel.event(PrecedenceTagEvent(...))

    Downstream consumers:
        battery.py reads tel.tick_log
        yaml_validator.py reads tel.event_log for depressive lock-in counts
    """

    def __init__(self, max_buffer: int = 100_000):
        self.tick_log:  List[TickRecord]  = []
        self.event_log: List[Any]         = []
        self._max = max_buffer

    def tick(self, record: TickRecord) -> None:
        if len(self.tick_log) < self._max:
            self.tick_log.append(record)

    def event(self, ev: Any) -> None:
        self.event_log.append(ev)
        # Depressive lock-in auto-check from tick record is handled by the engine;
        # this method just stores. See check_depressive_lock_in() below.

    def precedence_fraction(self) -> Dict[str, float]:
        """
        Compute (OVERRIDE + SCORE_WIN) / total for §M2.7.6 publication rule.
        Call after a full Regime 1 run.
        """
        tags = [e for e in self.event_log if isinstance(e, PrecedenceTagEvent)]
        if not tags:
            return {"override": 0.0, "score_win": 0.0, "weighted_arb": 0.0, "total": 0, "strong_fraction": 0.0, "tier_a_gate_pass": False}
        n = len(tags)
        n_override   = sum(1 for t in tags if t.event_tag == PrecedenceTag.OVERRIDE)
        n_score_win  = sum(1 for t in tags if t.event_tag == PrecedenceTag.SCORE_WIN)
        n_weighted   = sum(1 for t in tags if t.event_tag == PrecedenceTag.WEIGHTED_ARB)
        strong = (n_override + n_score_win) / n
        return {
            "override":        n_override / n,
            "score_win":       n_score_win / n,
            "weighted_arb":    n_weighted / n,
            "total":           n,
            "strong_fraction": strong,
            "tier_a_gate_pass": strong > 0.70,
        }

    def world_model_decay(self) -> Dict[str, Any]:
        """
        Extract world_model_error by tick for cold-start decay gate (P0.4, P1.4).
        Returns: {tick: mean_error_across_agents}
        """
        from collections import defaultdict
        by_tick: Dict[int, List[float]] = defaultdict(list)
        for r in self.tick_log:
            by_tick[r.tick].append(r.world_model_error)
        return {t: sum(v)/len(v) for t, v in sorted(by_tick.items())}

    def flush_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({
                "ticks":  [r.to_dict() for r in self.tick_log],
                "events": [asdict(e) for e in self.event_log],
            }, f)

    def clear(self) -> None:
        self.tick_log.clear()
        self.event_log.clear()


# ──────────────────────────────────────────────────────────────
# Engine-side helpers: what the sim loop calls each tick
# ──────────────────────────────────────────────────────────────

# Biological collapse order (index = collapse priority, 0 = first to collapse)
BIOLOGICAL_COLLAPSE_ORDER = [
    M2Family.REPAIR,
    M2Family.EXPLORE,
    M2Family.SEEK_HELP,
    M2Family.DECEIVE,
    M2Family.DOMINATE,
    M2Family.WITHDRAW,
    M2Family.DEFEND,
]

# Canonical policy score vector index order
FAMILY_INDEX = {f: i for i, f in enumerate([
    M2Family.DEFEND, M2Family.WITHDRAW, M2Family.REPAIR,
    M2Family.EXPLORE, M2Family.DOMINATE, M2Family.SEEK_HELP, M2Family.DECEIVE
])}


def resolve_precedence_tag(
    primitive_reflex_fired: bool,
    trigger_override_fired: bool,
    is_baseline: bool,
    policy_score_won_against_pressure: bool,
) -> PrecedenceTag:
    """
    §M2.7.6 — Every action must have a tagged dominant module.
    Called by the sim engine to produce the PrecedenceTagEvent.
    """
    if primitive_reflex_fired or trigger_override_fired:
        return PrecedenceTag.OVERRIDE
    if policy_score_won_against_pressure:
        return PrecedenceTag.SCORE_WIN
    return PrecedenceTag.WEIGHTED_ARB


def check_depressive_lock_in(
    tel: TelemetryEmitter,
    agent_id: str,
    tick: int,
    baseline_ticks: int,
    mourn_ticks: int,
    narrative_coherence: float,
    narrative_coherence_baseline: float,
    threshold_baseline_ticks: int = 20,
    threshold_mourn_ticks: int = 15,
    threshold_coherence_ratio: float = 0.70,
) -> bool:
    """
    Called by sim engine each tick to detect depressive lock-in.
    Emits DEPRESSIVE_LOCK_IN event if triggered. Returns True if fired.
    """
    coherence_ratio = narrative_coherence / max(0.01, narrative_coherence_baseline)
    if (baseline_ticks > threshold_baseline_ticks
            and mourn_ticks > threshold_mourn_ticks
            and coherence_ratio < threshold_coherence_ratio):
        tel.event(DepressionLockInEvent(
            tick=tick,
            agent_id=agent_id,
            BASELINE_ticks_count=baseline_ticks,
            mourn_ticks_count=mourn_ticks,
            narrative_coherence_ratio=coherence_ratio,
        ))
        return True
    return False
