"""
fmb_suite.py  —  P3.2 + P4.1
==============================
Failure Morphology Benchmark — four dimensions.

The FMB is the fastest publication path (P4.1 workshop paper).
It does NOT require Tier A battery_win. It requires:
  - Regime 1 PEACETIME data (≥ 32 agents)
  - topology_suite.py results (collapse traces)
  - M2 vs flat U(a) comparison on all four dimensions

Four dimensions (§M2.15):

  Dim 1 — Collapse predictability
    Is the M2 failure mode foreseeable from the current state?
    Metric: KL(empirical_collapse_dist || uniform) — higher = more structured
    M2 target: KL > 0.30 (non-uniform collapse distribution)
    Flat U(a): KL ≈ 0 (no structural failure mode)

  Dim 2 — Failure-onset speed
    How quickly does the agent enter a failure state after stress peaks?
    Metric: mean ticks from rd_peak to first BASELINE lock-in or FMB event
    M2 target: onset delayed by ≥ 5 ticks vs flat U(a)
    (Persistence minimum provides natural delay buffer)

  Dim 3 — Recovery tractability
    Can the agent recover from failure, and how fast?
    Metric: fraction of episodes where BASELINE lock-in resolves within 30 ticks
    M2 target: recovery_rate ≥ 0.60 (typed families re-activate as rd drops)
    Flat U(a): recovery_rate ≈ 0 (no state structure to re-activate)

  Dim 4 — Cross-agent failure contagion
    Does one agent's failure propagate to neighbours?
    Metric: Pearson r between agent failure-onset ticks across population
    M2 target: r < 0.30 (individuality limits contagion)
    Flat U(a): r ≈ 0 (no structure, random contagion)

The FMB paper contributes dimensions 1–4 as a benchmarking contribution
independent of the Tier A mechanism claim. This means it can be submitted
to a workshop before Regime 2 data is available.

§ references: CMA v4.1 §M2.15, roadmap §P4.1
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math
import statistics
import numpy as np
from collections import Counter, defaultdict

from telemetry import TickRecord, M2Family, PrecedenceTag
from topology_suite import CollapseTrace


# ──────────────────────────────────────────────────────────────
# Failure event detection
# ──────────────────────────────────────────────────────────────

@dataclass
class FailureEvent:
    agent_id:    str
    tick:        int
    event_type:  str   # "BASELINE_LOCK_IN" | "DEPRESSIVE_LOCK_IN" | "COLLAPSE_CASCADE"
    rd_at_onset: float
    recovery_tick: Optional[int] = None   # tick when failure resolved (None = unresolved)


def detect_failure_events(
    records:   List[TickRecord],
    lock_in_threshold_ticks: int   = 20,
    rd_stress_threshold:     float = 0.40,
) -> List[FailureEvent]:
    """
    Detect failure events from per-tick records.
    BASELINE_LOCK_IN: ≥ lock_in_threshold_ticks consecutive BASELINE ticks during stress.
    """
    by_agent: Dict[str, List[TickRecord]] = defaultdict(list)
    for r in sorted(records, key=lambda r: (r.agent_id, r.tick)):
        by_agent[r.agent_id].append(r)

    events: List[FailureEvent] = []
    for agent_id, agent_records in by_agent.items():
        baseline_run = 0
        baseline_start_tick = None
        in_failure = False
        failure_event: Optional[FailureEvent] = None

        for r in agent_records:
            is_baseline = (r.active_policy_family == M2Family.BASELINE)
            is_stress   = (r.regression_depth >= rd_stress_threshold)

            if is_baseline and is_stress:
                if baseline_run == 0:
                    baseline_start_tick = r.tick
                baseline_run += 1
            else:
                if in_failure and failure_event is not None and not is_baseline:
                    # Failure resolved
                    failure_event.recovery_tick = r.tick
                    in_failure = False
                baseline_run = 0

            if baseline_run >= lock_in_threshold_ticks and not in_failure:
                fe = FailureEvent(
                    agent_id=agent_id,
                    tick=baseline_start_tick or r.tick,
                    event_type="BASELINE_LOCK_IN",
                    rd_at_onset=r.regression_depth,
                )
                events.append(fe)
                failure_event = fe
                in_failure = True

    return events


# ──────────────────────────────────────────────────────────────
# Dim 1 — Collapse predictability
# ──────────────────────────────────────────────────────────────

@dataclass
class Dim1Result:
    kl_vs_uniform:          float   # KL(empirical || uniform) — higher = more structured
    collapse_distribution:  Dict[str, float]
    top_collapse_signature: str
    passes_threshold:       bool    # target KL > 0.30

    def summary(self) -> str:
        flag = "✓" if self.passes_threshold else "✗"
        return (f"  {flag} Dim 1 — Collapse predictability  "
                f"KL={self.kl_vs_uniform:.4f}  (target > 0.30)\n"
                f"      Top signature: {self.top_collapse_signature}")


def compute_dim1(traces: List[CollapseTrace]) -> Dim1Result:
    """KL(empirical collapse signature distribution || uniform)."""
    sigs = [t.collapse_sig for t in traces]
    counts = Counter(sigs)
    vocab   = sorted(counts)
    probs   = np.array([counts[s] / len(sigs) for s in vocab], dtype=np.float64)
    K       = len(vocab)
    uniform = np.ones(K, dtype=np.float64) / K
    eps     = 1e-12
    p_smooth = (probs + eps) / (probs + eps).sum()
    q_smooth = (uniform + eps) / (uniform + eps).sum()
    kl = float(np.sum(p_smooth * np.log(p_smooth / q_smooth)))

    top_sig = vocab[int(np.argmax(probs))] if vocab else "N/A"
    return Dim1Result(
        kl_vs_uniform=kl,
        collapse_distribution={v: float(probs[i]) for i, v in enumerate(vocab)},
        top_collapse_signature=top_sig,
        passes_threshold=kl > 0.30,
    )


# ──────────────────────────────────────────────────────────────
# Dim 2 — Failure-onset speed
# ──────────────────────────────────────────────────────────────

@dataclass
class Dim2Result:
    mean_onset_ticks_m2:     float   # ticks from rd_peak to failure onset
    mean_onset_ticks_flat:   float   # same for flat U(a)
    delay_advantage:         float   # m2 - flat; target ≥ 5 (longer = better)
    passes_threshold:        bool

    def summary(self) -> str:
        flag = "✓" if self.passes_threshold else "✗"
        return (f"  {flag} Dim 2 — Failure-onset speed  "
                f"M2={self.mean_onset_ticks_m2:.1f}t  Flat={self.mean_onset_ticks_flat:.1f}t  "
                f"advantage={self.delay_advantage:+.1f}t  (target ≥ +5)")


def compute_dim2(
    m2_events:   List[FailureEvent],
    flat_events: List[FailureEvent],
    rd_peak_tick: int = 90,
) -> Dim2Result:
    """Mean ticks from rd_peak to first failure onset."""
    def mean_onset(events: List[FailureEvent]) -> float:
        if not events:
            return float("inf")
        delays = [max(0, e.tick - rd_peak_tick) for e in events]
        return sum(delays) / len(delays)

    m2_onset   = mean_onset(m2_events)
    flat_onset = mean_onset(flat_events)

    # If no failures detected, assign large delay (good)
    m2_onset   = m2_onset   if m2_onset   < float("inf") else 1000.0
    flat_onset = flat_onset if flat_onset < float("inf") else 0.0   # flat has no structure → immediate

    advantage = m2_onset - flat_onset
    return Dim2Result(
        mean_onset_ticks_m2=m2_onset,
        mean_onset_ticks_flat=flat_onset,
        delay_advantage=advantage,
        passes_threshold=advantage >= 5.0,
    )


# ──────────────────────────────────────────────────────────────
# Dim 3 — Recovery tractability
# ──────────────────────────────────────────────────────────────

@dataclass
class Dim3Result:
    recovery_rate_m2:       float   # fraction resolved within recovery_window ticks
    recovery_rate_flat:     float
    rate_advantage:         float   # m2 - flat; target ≥ 0.40
    passes_threshold:       bool
    recovery_window:        int

    def summary(self) -> str:
        flag = "✓" if self.passes_threshold else "✗"
        return (f"  {flag} Dim 3 — Recovery tractability  "
                f"M2={self.recovery_rate_m2:.2f}  Flat={self.recovery_rate_flat:.2f}  "
                f"advantage={self.rate_advantage:+.2f}  (target ≥ +0.40)")


def compute_dim3(
    m2_events:   List[FailureEvent],
    flat_events: List[FailureEvent],
    recovery_window: int = 30,
) -> Dim3Result:
    """Fraction of failure events where recovery_tick ≤ onset_tick + recovery_window."""
    def recovery_rate(events: List[FailureEvent]) -> float:
        if not events:
            return 1.0   # no failures = full recovery
        resolved = sum(
            1 for e in events
            if e.recovery_tick is not None and (e.recovery_tick - e.tick) <= recovery_window
        )
        return resolved / len(events)

    m2_rate   = recovery_rate(m2_events)
    flat_rate = recovery_rate(flat_events)
    advantage = m2_rate - flat_rate

    return Dim3Result(
        recovery_rate_m2=m2_rate,
        recovery_rate_flat=flat_rate,
        rate_advantage=advantage,
        passes_threshold=m2_rate >= 0.60 and advantage >= 0.40,
        recovery_window=recovery_window,
    )


# ──────────────────────────────────────────────────────────────
# Dim 4 — Cross-agent failure contagion
# ──────────────────────────────────────────────────────────────

@dataclass
class Dim4Result:
    pearson_r_m2:    float   # lower = less contagion
    pearson_r_flat:  float
    individuality_advantage: float  # flat.r - m2.r; target ≥ 0.10
    passes_threshold: bool

    def summary(self) -> str:
        flag = "✓" if self.passes_threshold else "✗"
        return (f"  {flag} Dim 4 — Cross-agent contagion  "
                f"M2_r={self.pearson_r_m2:.3f}  Flat_r={self.pearson_r_flat:.3f}  "
                f"individuality_advantage={self.individuality_advantage:+.3f}  (target ≥ +0.10)")


def compute_dim4(
    m2_events:   List[FailureEvent],
    flat_events: List[FailureEvent],
    agent_ids:   List[str],
) -> Dim4Result:
    """
    Pearson r between failure-onset tick vectors across agents.
    High r = failure times are correlated = contagion.
    Low r = agents fail independently = individuality preserved.
    """
    def failure_onset_vector(events: List[FailureEvent], ids: List[str]) -> np.ndarray:
        onset: Dict[str, int] = {}
        for e in events:
            if e.agent_id not in onset:
                onset[e.agent_id] = e.tick
        return np.array([onset.get(aid, 10000) for aid in ids], dtype=np.float64)

    if len(agent_ids) < 2:
        return Dim4Result(0.0, 0.0, 0.0, False)

    m2_vec   = failure_onset_vector(m2_events,   agent_ids)
    flat_vec = failure_onset_vector(flat_events, agent_ids)

    def pearson_r(v: np.ndarray) -> float:
        if v.std() < 1e-8:
            return 0.0
        return float(np.corrcoef(v, np.arange(len(v)))[0, 1])

    m2_r   = pearson_r(m2_vec)
    flat_r = pearson_r(flat_vec)
    advantage = abs(flat_r) - abs(m2_r)

    return Dim4Result(
        pearson_r_m2=m2_r,
        pearson_r_flat=flat_r,
        individuality_advantage=advantage,
        passes_threshold=abs(m2_r) < 0.30 and advantage >= 0.10,
    )


# ──────────────────────────────────────────────────────────────
# Full FMB suite
# ──────────────────────────────────────────────────────────────

@dataclass
class FMBResult:
    dim1: Dim1Result
    dim2: Dim2Result
    dim3: Dim3Result
    dim4: Dim4Result
    dims_passed:         int     # 0–4
    fmb_paper_ready:     bool    # all 4 pass = workshop paper contribution ready
    m2_failure_events:   List[FailureEvent]
    flat_failure_events: List[FailureEvent]

    def summary(self) -> str:
        lines = ["\n══ Failure Morphology Benchmark (FMB) ═════════════════"]
        lines.append(self.dim1.summary())
        lines.append(self.dim2.summary())
        lines.append(self.dim3.summary())
        lines.append(self.dim4.summary())
        lines.append(f"\n  Dimensions passed: {self.dims_passed}/4")
        if self.fmb_paper_ready:
            lines.append("  ✓ FMB PAPER READY — all 4 dimensions contribute")
        else:
            lines.append(f"  ✗ FMB paper not yet complete ({4 - self.dims_passed} dimension(s) failing)")
            if not self.dim1.passes_threshold:
                lines.append("    → Dim 1: need more agents or longer arc for collapse diversity")
            if not self.dim2.passes_threshold:
                lines.append("    → Dim 2: persistence minimum may be too short")
            if not self.dim3.passes_threshold:
                lines.append("    → Dim 3: typed families not re-activating during recovery")
            if not self.dim4.passes_threshold:
                lines.append("    → Dim 4: check phenotype prior diversity")
        return "\n".join(lines)


def run_fmb_suite(
    m2_records:     List[TickRecord],
    flat_records:   List[TickRecord],
    m2_traces:      List[CollapseTrace],   # from topology_suite
    agent_ids:      List[str],
    rd_peak_tick:   int = 90,
    lock_in_threshold: int = 20,
    recovery_window:   int = 30,
) -> FMBResult:
    """
    Run all four FMB dimensions.

    Args:
        m2_records:   TickRecords from M2 Regime 1 run
        flat_records: TickRecords from flat U(a) baseline run
        m2_traces:    collapse traces from topology_suite (for Dim 1)
        agent_ids:    list of all agent IDs for Dim 4
    """
    m2_events   = detect_failure_events(m2_records,   lock_in_threshold)
    flat_events = detect_failure_events(flat_records, lock_in_threshold)

    print(f"\n  Failure events detected — M2: {len(m2_events)}  Flat: {len(flat_events)}")

    d1 = compute_dim1(m2_traces if m2_traces else [])
    d2 = compute_dim2(m2_events, flat_events, rd_peak_tick)
    d3 = compute_dim3(m2_events, flat_events, recovery_window)
    d4 = compute_dim4(m2_events, flat_events, agent_ids)

    dims_passed = sum([d1.passes_threshold, d2.passes_threshold, d3.passes_threshold, d4.passes_threshold])
    result = FMBResult(
        dim1=d1, dim2=d2, dim3=d3, dim4=d4,
        dims_passed=dims_passed,
        fmb_paper_ready=(dims_passed == 4),
        m2_failure_events=m2_events,
        flat_failure_events=flat_events,
    )
    print(result.summary())
    return result
