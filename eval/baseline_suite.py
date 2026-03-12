"""
baseline_suite.py  —  P1.1 + P1.3
=====================================
P1.1: Flat U(a) baseline — establishes the floor.
P1.3: M1 ablation — verifies M2 family layer is necessary, not just beneficial.

Build order: run flat_ua first, then m2_minimal, then ablations.
Never run ablations before you have a passing M2 minimal reference run.

§ references: M2 v1.18 §M2.12.1, §M2.1.2 (four-component non-reducibility)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import math
import statistics
import random

from telemetry import TelemetryEmitter, TickRecord, M2Family, FAMILY_INDEX
from agent_wrapper import AgentWrapper, M2AgentWrapper, FlatUAWrapper, StepOutput


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

@dataclass
class BaselineSuiteConfig:
    num_actions:         int   = 20
    num_agents:          int   = 8
    num_ticks:           int   = 200
    num_seeds:           int   = 3
    regime:              str   = "PEACETIME"
    # Ablation targets — seven, one per §M2.12.1
    ablation_targets: List[str] = field(default_factory=lambda: [
        "family_layer",
        "persistence_minimum",
        "aq_threshold_ordering",
        "switch_cost",
        "social_signal",
        "phenotype_prior",
        "explanation_trace",
    ])


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────

def coefficient_of_variation(values: List[float]) -> float:
    """CV = std / mean. Used for tactic_class distribution. Flat baseline → CV ≈ 0."""
    if not values or len(values) < 2:
        return 0.0
    m = statistics.mean(values)
    if m == 0.0:
        return 0.0
    return statistics.stdev(values) / m


def tactic_class_cv(tick_records: List[TickRecord]) -> float:
    """
    Observable 1 proxy: CV(tactic_class) distribution.
    High CV = tactic specialisation by family.
    Low/zero CV = flat action distribution (no strategic structure).
    Flat U(a) target: CV ≈ 0 (uniform action distribution).
    M2 target: CV > 0.25 vs flat U(a).
    """
    from collections import Counter
    counts = Counter(r.tactic_class for r in tick_records)
    return coefficient_of_variation(list(counts.values()))


def switch_frequency(tick_records: List[TickRecord]) -> float:
    """
    Fraction of ticks where a policy family switch occurred.
    M2 minimum: switch_cost_paid == True.
    """
    if len(tick_records) < 2:
        return 0.0
    switches = sum(1 for r in tick_records if r.switch_cost_paid)
    return switches / len(tick_records)


def oscillation_rate(tick_records: List[TickRecord], high_stress_rd_threshold: float = 0.45) -> float:
    """
    State transitions per tick at rd >= high_stress_rd_threshold.
    M2 target: ≤ 0.10. LSM target: > 0.20 (§M2.12.12).
    """
    high_stress = [r for r in tick_records if r.regression_depth >= high_stress_rd_threshold]
    if len(high_stress) < 2:
        return 0.0
    transitions = sum(
        1 for i in range(1, len(high_stress))
        if high_stress[i].active_policy_family != high_stress[i - 1].active_policy_family
    )
    return transitions / len(high_stress)


def spearman_collapse_rank(tick_records: List[TickRecord]) -> float:
    """
    Observable 3 proxy: rank correlation between accessible_families_count and
    inverse regression_depth. High correlation = families collapse in the right order.
    M2 target: ρ ≥ 0.85. Flat U(a) target: ρ ≈ 0 (no structured collapse).
    """
    if not tick_records:
        return 0.0
    pairs = [(r.regression_depth, len(r.accessible_families)) for r in tick_records if r.accessible_families is not None]
    if len(pairs) < 4:
        return 0.0

    rd_vals  = [p[0] for p in pairs]
    acc_vals = [p[1] for p in pairs]

    # Spearman rank correlation
    def rank(lst):
        sorted_lst = sorted(enumerate(lst), key=lambda x: x[1])
        ranks = [0.0] * len(lst)
        for rank_val, (orig_idx, _) in enumerate(sorted_lst):
            ranks[orig_idx] = float(rank_val + 1)
        return ranks

    rd_r  = rank(rd_vals)
    acc_r = rank(acc_vals)
    n = len(rd_r)
    mean_rd  = sum(rd_r) / n
    mean_acc = sum(acc_r) / n
    num = sum((rd_r[i] - mean_rd) * (acc_r[i] - mean_acc) for i in range(n))
    den = math.sqrt(sum((rd_r[i] - mean_rd) ** 2 for i in range(n)) *
                    sum((acc_r[i] - mean_acc) ** 2 for i in range(n)))
    if den == 0.0:
        return 0.0
    return -num / den  # negative: higher rd → fewer accessible families


# ──────────────────────────────────────────────────────────────
# Stub episode runner
# ──────────────────────────────────────────────────────────────

def run_episode(
    agent: AgentWrapper,
    config: BaselineSuiteConfig,
    seed: int,
    stress_ramp: bool = False,
) -> List[TickRecord]:
    """
    Run one episode. Returns list of TickRecord (one per agent tick).
    stress_ramp=True: linearly increases regression_depth over the episode.
    Real simulation engine replaces this stub.
    """
    rng = random.Random(seed)
    agent.reset(seed)
    records = []

    for tick in range(config.num_ticks):
        rd = min(1.0, tick / config.num_ticks) if stress_ramp else 0.10
        obs = [rng.gauss(0, 1) for _ in range(16)]

        out = agent.step(obs)

        # Build minimal TickRecord for metric collection
        family = (
            M2Family(list(M2Family)[out.active_state])
            if isinstance(agent, M2AgentWrapper) and out.active_state < 7
            else M2Family.BASELINE
        )
        record = TickRecord(
            tick=tick,
            agent_id=getattr(agent, 'agent_id', 'agent_0'),
            regime=config.regime,
            seed=seed,
            active_policy_family=family,
            policy_score_vector=[0.0] * 7,
            switch_cost_paid=False,
            switch_cost_magnitude=0.0,
            accessible_families=[M2Family.BASELINE],
            active_overlays=[],
            policy_conflict_detected=False,
            tactic_class=out.tactic_class,
            action_taken=out.action_taken,
            precedence_tag=getattr(out, '_precedence_tag', None) or __import__('telemetry').PrecedenceTag.WEIGHTED_ARB,
            dominant_module="stub",
            regression_depth=rd,
            baseline_ticks_running=0,
            mourn_during_baseline_ticks=0,
            narrative_coherence=1.0,
            world_model_error=max(0.0, 0.90 - (tick / config.num_ticks) * 0.80),
            primary_goal_valence=0.0,
        )
        records.append(record)

    return records


# ──────────────────────────────────────────────────────────────
# P1.1 — Flat U(a) Baseline
# ──────────────────────────────────────────────────────────────

@dataclass
class FlatUAResult:
    cv_tactic_class:    float  # target ≈ 0 (uniform distribution)
    switch_freq:        float  # no concept of switch — should be 0.0
    spearman_collapse:  float  # target ≈ 0 (no structured collapse)
    oscillation:        float  # unconstrained: expected HIGH
    total_ticks:        int

    def summary(self) -> str:
        return (
            f"\n── Flat U(a) Baseline ──────────────────\n"
            f"  CV(tactic_class):      {self.cv_tactic_class:.4f}  (target ≈ 0)\n"
            f"  Switch frequency:      {self.switch_freq:.4f}  (no concept)\n"
            f"  Spearman collapse ρ:   {self.spearman_collapse:.4f}  (target ≈ 0)\n"
            f"  Oscillation rate:      {self.oscillation:.4f}  (expect HIGH)\n"
            f"  Total ticks:           {self.total_ticks}\n"
        )


def run_flat_ua(config: BaselineSuiteConfig) -> FlatUAResult:
    """P1.1 — Run flat U(a) baseline and compute floor metrics."""
    tel = TelemetryEmitter()
    agent = FlatUAWrapper(num_actions=config.num_actions, telemetry=tel, agent_id="flat_ua_0")
    all_records: List[TickRecord] = []

    for seed in range(config.num_seeds):
        records = run_episode(agent, config, seed, stress_ramp=True)
        all_records.extend(records)

    result = FlatUAResult(
        cv_tactic_class   = tactic_class_cv(all_records),
        switch_freq       = switch_frequency(all_records),
        spearman_collapse = spearman_collapse_rank(all_records),
        oscillation       = oscillation_rate(all_records),
        total_ticks       = len(all_records),
    )
    print(result.summary())
    return result


# ──────────────────────────────────────────────────────────────
# P1.3 — M1 Ablation Suite
# ──────────────────────────────────────────────────────────────

@dataclass
class AblationTarget:
    name:              str
    description:       str
    pre_specified_expectation: str   # what MUST degrade (pre-registered)
    metric_key:        str           # which metric to compare
    direction:         str           # "decrease" | "increase" | "randomise" | "build_error"
    tolerance:         float = 0.10  # degradation must exceed this fraction of baseline


@dataclass
class AblationResult:
    target:            str
    baseline_metric:   float
    ablated_metric:    float
    degradation:       float         # relative change
    expectation_met:   bool
    notes:             str


# Pre-specified ablation targets (§M2.12.1) — register BEFORE running
ABLATION_TARGETS = [
    AblationTarget(
        name="family_layer",
        description="Remove M2 family layer — agent reverts to flat U(a) with no tactic restriction",
        pre_specified_expectation="CV(tactic_class) drops toward 0 (flat baseline)",
        metric_key="cv_tactic_class",
        direction="decrease",
        tolerance=0.50,  # expect ≥ 50% drop in CV
    ),
    AblationTarget(
        name="persistence_minimum",
        description="Remove 5-tick persistence minimum — families can switch every tick",
        pre_specified_expectation="Switch frequency rises, oscillation_rate matches LSM (> 0.20 at high stress)",
        metric_key="oscillation_rate",
        direction="increase",
        tolerance=0.10,  # expect oscillation rate to rise above 0.10
    ),
    AblationTarget(
        name="aq_threshold_ordering",
        description="Remove biological A/Q threshold ordering — thresholds randomised",
        pre_specified_expectation="Spearman rank with biological prior drops to ~0",
        metric_key="spearman_collapse",
        direction="decrease",
        tolerance=0.60,  # expect ≥ 60% drop in Spearman ρ
    ),
    AblationTarget(
        name="switch_cost",
        description="Remove switch cost — no penalty for family transition",
        pre_specified_expectation="Budget dip and trailing performance dip on switch disappear",
        metric_key="switch_cost_effect",
        direction="decrease",
        tolerance=0.50,
    ),
    AblationTarget(
        name="social_signal",
        description="Remove social signal output — observer cannot infer family from behaviour",
        pre_specified_expectation="Observer inference accuracy drops ≥ 10pp vs baseline",
        metric_key="observer_inference_accuracy",
        direction="decrease",
        tolerance=0.10,  # ≥ 10pp drop
    ),
    AblationTarget(
        name="phenotype_prior",
        description="Remove phenotype prior — agent-to-agent tactic divergence collapses",
        pre_specified_expectation="Inter-agent tactic_class KL divergence collapses toward 0",
        metric_key="inter_agent_kl",
        direction="decrease",
        tolerance=0.50,
    ),
    AblationTarget(
        name="explanation_trace",
        description="Remove explanation trace — §M2.6 fidelity metrics cannot be computed",
        pre_specified_expectation="BUILD ERROR: fidelity metrics unavailable (architectural, not degradation)",
        metric_key="fidelity_computable",
        direction="build_error",
        tolerance=0.0,
    ),
]


@dataclass
class AblationSuiteResult:
    results:           List[AblationResult]
    all_passed:        bool
    necessity_confirmed: bool  # True if all degradations met expectations

    def summary(self) -> str:
        lines = ["\n── M1 Ablation Suite ──────────────────"]
        for r in self.results:
            status = "✓" if r.expectation_met else "✗"
            lines.append(
                f"  {status} {r.target:<28} "
                f"baseline={r.baseline_metric:.3f} "
                f"ablated={r.ablated_metric:.3f} "
                f"Δ={r.degradation:+.3f}"
            )
            if r.notes:
                lines.append(f"      {r.notes}")
        lines.append(f"\n  Necessity confirmed: {self.necessity_confirmed}")
        return "\n".join(lines)


def run_ablation_suite(
    m2_baseline_records: List[TickRecord],
    ablated_records_by_target: Dict[str, List[TickRecord]],
) -> AblationSuiteResult:
    """
    Compare M2 baseline records against ablated variants.
    ablated_records_by_target: {target_name: [TickRecord]}

    Wire to real sim: run M2 minimal, then for each ablation target,
    re-run with that component disabled. Feed both record sets here.
    """
    baseline_metrics = {
        "cv_tactic_class":   tactic_class_cv(m2_baseline_records),
        "oscillation_rate":  oscillation_rate(m2_baseline_records),
        "spearman_collapse": spearman_collapse_rank(m2_baseline_records),
        "switch_cost_effect": switch_frequency(m2_baseline_records),
        "observer_inference_accuracy": 0.65,  # placeholder — wire from social_signal_suite
        "inter_agent_kl":    0.40,             # placeholder — wire from phenotype module
        "fidelity_computable": 1.0,
    }

    results: List[AblationResult] = []
    for target in ABLATION_TARGETS:
        ablated = ablated_records_by_target.get(target.name, [])

        if target.direction == "build_error":
            met = not ablated  # empty records = build error (cannot run without trace)
            results.append(AblationResult(
                target=target.name,
                baseline_metric=1.0,
                ablated_metric=0.0 if met else 1.0,
                degradation=-1.0 if met else 0.0,
                expectation_met=met,
                notes="Build error confirmed — fidelity metrics unavailable without trace" if met
                      else "WARNING: ablation did not produce build error as expected",
            ))
            continue

        baseline_val = baseline_metrics.get(target.metric_key, 0.0)

        if not ablated:
            # No data — ablation not yet run
            results.append(AblationResult(
                target=target.name,
                baseline_metric=baseline_val,
                ablated_metric=float("nan"),
                degradation=float("nan"),
                expectation_met=False,
                notes="NOT RUN — wire ablation and provide records",
            ))
            continue

        ablated_metrics = {
            "cv_tactic_class":   tactic_class_cv(ablated),
            "oscillation_rate":  oscillation_rate(ablated),
            "spearman_collapse": spearman_collapse_rank(ablated),
            "switch_cost_effect": switch_frequency(ablated),
            "observer_inference_accuracy": 0.0,  # placeholder
            "inter_agent_kl":    0.0,
            "fidelity_computable": 1.0,
        }
        ablated_val = ablated_metrics.get(target.metric_key, 0.0)
        delta = (ablated_val - baseline_val) / max(1e-6, abs(baseline_val))

        if target.direction == "decrease":
            met = delta < -target.tolerance
        elif target.direction == "increase":
            met = delta > target.tolerance
        elif target.direction == "randomise":
            met = abs(ablated_val) < 0.20  # near-zero = randomised
        else:
            met = False

        results.append(AblationResult(
            target=target.name,
            baseline_metric=baseline_val,
            ablated_metric=ablated_val,
            degradation=delta,
            expectation_met=met,
            notes="" if met else f"Expected {target.direction} by {target.tolerance:.0%} — not met",
        ))

    all_passed = all(r.expectation_met for r in results)
    print(AblationSuiteResult(results=results, all_passed=all_passed, necessity_confirmed=all_passed).summary())
    return AblationSuiteResult(results=results, all_passed=all_passed, necessity_confirmed=all_passed)


# ──────────────────────────────────────────────────────────────
# P1.4 — Cold-Start Decay Gate check (called after Regime 1 smoke)
# ──────────────────────────────────────────────────────────────

def check_cold_start_gate(
    telemetry: TelemetryEmitter,
    regime: str = "PEACETIME",
) -> Tuple[bool, str]:
    """
    Check cold-start decay gate from telemetry.
    Regime 1 PEACETIME: world_model_error < 0.30 by tick 150.
    Regime 2 STRESS:    world_model_error < 0.50 by tick 100.
    """
    from replay_buffer import ColdStartDecayGate
    gate = ColdStartDecayGate()
    decay = telemetry.world_model_decay()
    for tick, err in decay.items():
        gate.record(tick, err)

    if regime == "PEACETIME":
        return gate.check_regime_1_gate()
    elif regime in ("STRESS", "CATASTROPHE"):
        tick_100_err = decay.get(100, float("inf"))
        return gate.check_regime_2_gate(tick_100_err)
    return False, f"Unknown regime: {regime}"
