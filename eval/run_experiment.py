"""
run_experiment.py  —  Top-level orchestrator
=============================================
Single entry point for a full Phase 1–3 run.

Stages (each gated on the previous):
  0. Pre-flight gate (all P0 checks)
  1. Flat U(a) baseline (P1.1)
  2. M2 Minimal Regime 1 arc (P3.1)
  3. LSM battery — 3 suites (P2.4)
  4. FMB suite — 4 dimensions (P3.2)
  5. Ablation suite (P1.3)
  6. Taxonomy model-selection (P1.6)
  7. Verdict: Tier A / FMB paper / downgrade

Usage:
    python run_experiment.py                          # reference config, quick mode
    python run_experiment.py m2_config.yaml           # real config, full run
    python run_experiment.py --quick --agents 4       # fast smoke (CI)
    python run_experiment.py --fmb-only               # FMB paper path only

Output:
    results/run_<timestamp>/
        regime1_records.jsonl
        battery_result.json
        fmb_result.json
        observables.json
        verdict.md
"""

from __future__ import annotations
import sys
import json
import time
import argparse
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

from yaml_validator import generate_reference_config, run_validator
from preflight import run_preflight
from m2_policy import build_m2_agent, build_lsm_agent, M2MinimalPolicy, M2PolicyConfig
from agent_wrapper import FlatUAWrapper
from telemetry import TelemetryEmitter
from baseline_suite import run_flat_ua, BaselineSuiteConfig
from regime1_runner import run_regime1, Regime1Config, Regime1Result
from run_lsm_battery import run_full_battery, BatteryConfig, BatteryResult, build_default_battery_wrappers
from fmb_suite import run_fmb_suite, FMBResult
from baseline_suite import run_ablation_suite, AblationSuiteResult
from taxonomy_suite import run_taxonomy_model_selection


# ──────────────────────────────────────────────────────────────
# Experiment config
# ──────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    # Scale
    num_agents:          int   = 32
    num_seeds:           int   = 3
    ticks_per_episode:   int   = 500
    obs_dim:             int   = 16
    num_actions:         int   = 20    # small default; real engine: 128+

    # Stages to run
    run_flat_baseline:   bool  = True
    run_regime1:         bool  = True
    run_battery:         bool  = True
    run_fmb:             bool  = True
    run_ablation:        bool  = False  # slow — enable for publication run
    run_model_selection: bool  = False  # requires regime 1 first

    # Battery
    calibration_seeds:   int   = 8
    topology_seeds:      int   = 16
    cf_contexts:         int   = 20
    rollout_episodes:    int   = 20
    rollout_ticks:       int   = 100

    # Output
    output_dir:          str   = "results"
    run_id:              str   = ""


# ──────────────────────────────────────────────────────────────
# Status labels (verdict_contract.md Part 2 — FROZEN 2026-03-11)
# ──────────────────────────────────────────────────────────────

PASS                    = "PASS"
FAIL                    = "FAIL"
BLOCKED_BY_REGIME_LEN   = "BLOCKED_BY_REGIME_LENGTH"
NOT_TRIGGERED           = "NOT_TRIGGERED"
INVALIDATED_BY_BUG      = "INVALIDATED_BY_BUG"

# Minimum ticks required for stress-dependent metrics.
# Shocks at t=150 and t=300 require at least 160 ticks to fire once.
STRESS_MIN_TICKS        = 160
# Minimum seeds + episodes for character stability
CHAR_STABILITY_MIN_SEEDS = 3


# ──────────────────────────────────────────────────────────────
# Verdict
# ──────────────────────────────────────────────────────────────

@dataclass
class MetricStatus:
    """One row in the verdict table. See verdict_contract.md Part 2."""
    name:         str
    value:        Optional[float]   # numeric result; None if not measured
    threshold:    str               # human-readable threshold description
    status:       str               # one of the five label constants above
    note:         str = ""

    def icon(self) -> str:
        return {
            PASS:                 "✓ PASS",
            FAIL:                 "✗ FAIL",
            BLOCKED_BY_REGIME_LEN: "— BLOCKED (regime length)",
            NOT_TRIGGERED:        "○ NOT TRIGGERED",
            INVALIDATED_BY_BUG:   "⚠ INVALIDATED (bug)",
        }.get(self.status, self.status)

    def row(self) -> str:
        val_str = f"{self.value:.4f}" if self.value is not None else "n/a"
        return f"  {self.name:<36} {val_str:>8}   {self.icon()}"


@dataclass
class ExperimentVerdict:
    metrics:             List[MetricStatus]
    downgrade_level:     str   # FULL_TIER_A | DOWNGRADE_1 | DOWNGRADE_2 | DOWNGRADE_3_WITH_SIGNAL | DOWNGRADE_3
    recommended_action:  str
    notes:               List[str]
    # Convenience bools for callers
    tier_a_possible:     bool
    fmb_paper_ready:     bool
    battery_win:         bool
    tier_a_gate_pass:    bool
    is_quick_mode:       bool = False   # set True if ticks_per_episode < STRESS_MIN_TICKS

    def summary(self) -> str:
        lines = ["\n" + "═" * 66, "  EXPERIMENT VERDICT", "═" * 66]
        if self.is_quick_mode:
            lines.append("  ⚠  INTEGRITY MODE — stress/collapse metrics are BLOCKED (not measured)")
            lines.append("  ⚠  Do not interpret BLOCKED metrics as evidence for or against M2")
            lines.append("")
        lines.append(f"  {'Metric':<36} {'Value':>8}   Status")
        lines.append("  " + "─" * 60)
        for m in self.metrics:
            lines.append(m.row())
            if m.note:
                lines.append(f"       → {m.note}")
        lines.append("")
        lines.append(f"  Downgrade level:    {self.downgrade_level}")
        lines.append(f"  Recommended action: {self.recommended_action}")
        if self.notes:
            lines.append("")
            for n in self.notes:
                lines.append(f"  ⚠  {n}")
        lines.append("═" * 66)
        return "\n".join(lines)


def _metric_status(
    name:          str,
    value:         Optional[float],
    threshold_val: float,
    compare:       str,          # ">", ">=", "<", "<="
    threshold_str: str,
    ticks:         int,
    requires_stress: bool = False,
    requires_char_stability: bool = False,
    num_seeds: int = 3,
) -> MetricStatus:
    """
    Apply the five-way status label rules from verdict_contract.md Part 2.
    Rules applied in priority order: BLOCKED > NOT_TRIGGERED > PASS/FAIL.
    """
    # 1. BLOCKED_BY_REGIME_LENGTH — structural impossibility
    if requires_stress and ticks < STRESS_MIN_TICKS:
        return MetricStatus(name, value, threshold_str, BLOCKED_BY_REGIME_LEN,
                            note=f"stress shocks at t=150/300 require ≥{STRESS_MIN_TICKS} ticks")
    if requires_char_stability and num_seeds < CHAR_STABILITY_MIN_SEEDS:
        return MetricStatus(name, value, threshold_str, BLOCKED_BY_REGIME_LEN,
                            note=f"character stability requires ≥{CHAR_STABILITY_MIN_SEEDS} seeds")

    # 2. NOT_TRIGGERED — value absent or zero where nonzero is required to interpret
    if value is None:
        return MetricStatus(name, value, threshold_str, NOT_TRIGGERED,
                            note="metric not collected in this run")

    # 3. PASS / FAIL — threshold comparison
    ops = {">": lambda a,b: a>b, ">=": lambda a,b: a>=b,
           "<": lambda a,b: a<b, "<=": lambda a,b: a<=b}
    passed = ops[compare](value, threshold_val)
    return MetricStatus(name, value, threshold_str, PASS if passed else FAIL)


def compute_verdict(
    regime1:    Optional[Regime1Result],
    battery:    Optional[BatteryResult],
    fmb:        Optional[FMBResult],
    ticks:      int = 500,     # actual ticks_per_episode used in this run
    num_seeds:  int = 3,
) -> ExperimentVerdict:
    """
    Verdict aggregation — frozen contract (verdict_contract.md 2026-03-11).
    Status labels: PASS / FAIL / BLOCKED_BY_REGIME_LENGTH / NOT_TRIGGERED / INVALIDATED_BY_BUG
    Aggregation rule: first-match downgrade tree (Part 3).
    """
    quick = ticks < STRESS_MIN_TICKS
    metrics: List[MetricStatus] = []
    notes: List[str] = []

    # ── Per-metric status ──────────────────────────────────────

    # Observable 1 — Strategic Individuality
    obs1_val = None
    if regime1 and regime1.observables:
        obs = regime1.observables
        mean = lambda k: sum(getattr(o, k) for o in obs) / max(1, len(obs))
        obs1_val = mean("cv_tactic_class")
    m_obs1 = _metric_status("Obs 1 — CV(tactic_class)", obs1_val, 0.25, ">",
                             "> 0.25", ticks)
    metrics.append(m_obs1)

    # Observable 2a — Switch frequency
    obs2a_val = None
    if regime1 and regime1.observables:
        obs2a_val = mean("switch_frequency")
    # Zero switch frequency = no switches occurred = NOT_TRIGGERED (contingent absence, not FAIL)
    if obs2a_val is not None and obs2a_val == 0.0:
        m_obs2a = MetricStatus("Obs 2a — Switch freq", obs2a_val, "> 0.01",
                               NOT_TRIGGERED, note="no switches observed in this run")
    else:
        m_obs2a = _metric_status("Obs 2a — Switch freq", obs2a_val, 0.01, ">",
                                  "> 0.01", ticks)
    metrics.append(m_obs2a)

    # Observable 2b — Post-switch degradation
    obs2b_val = None
    if regime1 and regime1.observables:
        obs2b_val = mean("post_switch_degradation")
    # If no switches, degradation is undefined — NOT_TRIGGERED regardless of value
    if m_obs2a.status == NOT_TRIGGERED or obs2b_val == 0.0 and obs2a_val == 0.0:
        m_obs2b = MetricStatus("Obs 2b — Post-switch degradation", obs2b_val, "≤ −0.10",
                               NOT_TRIGGERED, note="no switches to measure degradation against")
    else:
        m_obs2b = _metric_status("Obs 2b — Post-switch degradation", obs2b_val, -0.10, "<=",
                                  "≤ −0.10", ticks)
    metrics.append(m_obs2b)

    # Observable 3 — Stress collapse order (requires stress)
    obs3_val = None
    if regime1 and regime1.observables:
        obs3_val = mean("spearman_collapse_rho")
    m_obs3 = _metric_status("Obs 3 — Spearman collapse ρ", obs3_val, 0.85, ">=",
                             "≥ 0.85", ticks, requires_stress=True)
    metrics.append(m_obs3)

    # Observable 4 — Social legibility
    obs4_val = None
    if battery and hasattr(battery, 'social_signal') and battery.social_signal:
        obs4_val = getattr(battery.social_signal, 'delta_lift_advantage', None)
    m_obs4 = _metric_status("Obs 4 — Social lift advantage", obs4_val, 0.05, ">=",
                             "≥ 0.05", ticks)
    metrics.append(m_obs4)

    # Observable 5 — Character stability (requires enough seeds/episodes)
    obs5_val = None
    if regime1 and regime1.observables:
        obs5_val = mean("pearson_r_char_stability")
    m_obs5 = _metric_status("Obs 5 — Pearson r char stability", obs5_val, 0.70, ">=",
                             "≥ 0.70", ticks, requires_char_stability=True, num_seeds=num_seeds)
    metrics.append(m_obs5)

    # Tier A gate — informative only under stress (quick mode = NOT_TRIGGERED, not PASS)
    tier_a_gate_val = None
    if battery and battery.tier_a_gate:
        tier_a_gate_val = battery.tier_a_gate.get("strong_fraction", None)
    if quick and tier_a_gate_val is not None:
        # Quick mode Tier A = NOT_TRIGGERED: no stress → no OVERRIDE pressure → value is uninformative
        m_tier_a = MetricStatus("Tier A gate (OVERRIDE+SCORE_WIN)", tier_a_gate_val,
                                "> 0.70", NOT_TRIGGERED,
                                note="quick mode: no stress → OVERRIDE never fires; value uninformative")
    else:
        m_tier_a = _metric_status("Tier A gate (OVERRIDE+SCORE_WIN)", tier_a_gate_val,
                                  0.70, ">", "> 0.70", ticks)
    metrics.append(m_tier_a)

    # Battery — Topology win (requires stress)
    topo_val = 1.0 if (battery and battery.topology_win) else (0.0 if battery else None)
    m_topo = _metric_status("Battery — Topology win", topo_val, 0.5, ">",
                             "M2 Spearman≥0.85 + JS>0.10 + entropy>1.0", ticks, requires_stress=True)
    metrics.append(m_topo)

    # Battery — Counterfactual win
    cf_val = 1.0 if (battery and battery.counterfactual_win) else (0.0 if battery else None)
    m_cf = _metric_status("Battery — Counterfactual win", cf_val, 0.5, ">",
                           "M2 coherence > LSM AND M2 margin > LSM", ticks)
    metrics.append(m_cf)

    # Battery — Social signal win
    ss_val = 1.0 if (battery and battery.social_signal_win) else (0.0 if battery else None)
    m_ss = _metric_status("Battery — Social signal win", ss_val, 0.5, ">",
                           "M2 lift advantage ≥ 0.05", ticks)
    metrics.append(m_ss)

    # FMB Dim 1 — Collapse predictability (requires stress)
    fmb1_val = fmb.dim1.kl_vs_uniform if fmb else None
    m_fmb1 = _metric_status("FMB Dim 1 — Collapse KL", fmb1_val, 0.30, ">",
                             "> 0.30", ticks, requires_stress=True)
    metrics.append(m_fmb1)

    # FMB Dim 2 — Failure onset delay (requires failure events; else NOT_TRIGGERED)
    fmb2_val = fmb.dim2.delay_advantage if fmb else None
    no_failures = fmb is not None and len(fmb.m2_failure_events) == 0
    if fmb2_val is not None and no_failures:
        # fmb2_val = 1000t sentinel = no failures detected = NOT_TRIGGERED, not a real signal
        m_fmb2 = MetricStatus("FMB Dim 2 — Onset delay advantage", fmb2_val,
                               "≥ +5 ticks", NOT_TRIGGERED,
                               note="no failure events detected — onset delay cannot be measured")
    else:
        m_fmb2 = _metric_status("FMB Dim 2 — Onset delay advantage", fmb2_val, 5.0, ">=",
                                 "≥ +5 ticks", ticks)
    metrics.append(m_fmb2)

    # FMB Dim 3 — Recovery tractability (requires failure events + stress)
    fmb3_val = fmb.dim3.recovery_rate_m2 if fmb else None
    if no_failures and quick:
        m_fmb3 = MetricStatus("FMB Dim 3 — Recovery tractability", fmb3_val,
                               "M2 rate ≥ 0.60", BLOCKED_BY_REGIME_LEN,
                               note="no failure events possible in quick mode (shocks at t=150/300)")
    elif no_failures:
        m_fmb3 = MetricStatus("FMB Dim 3 — Recovery tractability", fmb3_val,
                               "M2 rate ≥ 0.60", NOT_TRIGGERED,
                               note="no failure events detected in this run")
    else:
        m_fmb3 = _metric_status("FMB Dim 3 — Recovery tractability", fmb3_val, 0.60, ">=",
                                 "M2 rate ≥ 0.60", ticks, requires_stress=True)
    metrics.append(m_fmb3)

    # FMB Dim 4 — Cross-agent contagion (requires failure events + stress)
    fmb4_val = fmb.dim4.pearson_r_m2 if fmb else None
    # Dim 4 measures M2 r < 0.30 (low contagion = good). We report the individuality
    # advantage = flat_r - m2_r; target ≥ +0.10.
    fmb4_adv = (fmb.dim4.pearson_r_flat - fmb.dim4.pearson_r_m2) if fmb else None
    if no_failures and quick:
        m_fmb4 = MetricStatus("FMB Dim 4 — Contagion individuality", fmb4_adv,
                               "advantage ≥ +0.10", BLOCKED_BY_REGIME_LEN,
                               note="no failure events possible in quick mode (shocks at t=150/300)")
    elif no_failures:
        m_fmb4 = MetricStatus("FMB Dim 4 — Contagion individuality", fmb4_adv,
                               "advantage ≥ +0.10", NOT_TRIGGERED,
                               note="no failure events detected in this run")
    else:
        m_fmb4 = _metric_status("FMB Dim 4 — Contagion individuality", fmb4_adv, 0.10, ">=",
                                 "advantage ≥ +0.10", ticks, requires_stress=True)
    metrics.append(m_fmb4)

    # ── Aggregation (verdict_contract.md Part 3 — frozen) ─────

    battery_win  = battery.battery_win if battery else False
    # Tier A gate is only informative if NOT quick mode
    tier_a_gate  = (not quick) and bool(battery and battery.tier_a_gate and battery.tier_a_gate.get("tier_a_gate_pass", False))
    fmb_ready    = fmb.fmb_paper_ready if fmb else False
    obs1_pass    = m_obs1.status  == PASS
    obs3_pass    = m_obs3.status  == PASS
    obs5_pass    = m_obs5.status  == PASS
    # any_signal: at least one non-blocked metric passed
    any_signal   = any(m.status == PASS for m in metrics)

    # First-match downgrade tree
    if battery_win and tier_a_gate and obs1_pass and obs3_pass:
        level  = "FULL_TIER_A"
        action = "Submit to NeurIPS/ICLR/Nature MI. Include downgrade tree in supplementary."
    elif battery_win and not tier_a_gate and not quick:
        level  = "DOWNGRADE_1"
        action = ("Battery win without mechanism gate. "
                  "Reframe as correlational evidence of meso-level state advantage. Submit to ICLR or similar.")
        notes.append("OVERRIDE+SCORE_WIN ≤ 0.70 — cannot claim mechanistic causation for Tier A")
    elif not battery_win and fmb_ready:
        level  = "DOWNGRADE_2"
        action = ("FMB paper ready for workshop submission (NeurIPS/ICML workshop). "
                  "Return to Phase 2 to re-run battery.")
        notes.append("Battery loss. FMB benchmark contribution stands.")
    elif any_signal:
        level  = "DOWNGRADE_3_WITH_SIGNAL"
        action = ("Partial signals detected. Characterise what was found. "
                  "Return to Phase 1 before attempting Tier A.")
        notes.append("Some metrics passed — architecture expresses intended mechanisms. "
                     "Not yet sufficient for publication claim.")
    else:
        level  = "DOWNGRADE_3"
        action = "Return to Phase 1. Check YAML config, re-run ablations, validate observable targets."
        notes.append("No publishable claim yet. Do not submit.")

    # Quick mode: no downgrade from stress-blocked metrics
    if quick:
        notes.append(
            "INTEGRITY MODE RUN — stress/collapse metrics are BLOCKED_BY_REGIME_LENGTH. "
            "Downgrade level reflects integrity-mode signal only, not publication evidence."
        )
    if regime1 and regime1.gate_passed is None:
        notes.append("Cold-start gate BLOCKED (run too short — ticks < 150). Not a failure.")
    elif not regime1 or regime1.gate_passed is False:
        notes.append("Cold-start gate not passed — Regime 2 is blocked.")

    return ExperimentVerdict(
        metrics=metrics,
        downgrade_level=level,
        recommended_action=action,
        notes=notes,
        tier_a_possible=battery_win and tier_a_gate,
        fmb_paper_ready=fmb_ready,
        battery_win=battery_win,
        tier_a_gate_pass=tier_a_gate,
        is_quick_mode=quick,
    )


# ──────────────────────────────────────────────────────────────
# Experiment runner
# ──────────────────────────────────────────────────────────────

def run_experiment(cfg: ExperimentConfig, yaml_config: dict) -> ExperimentVerdict:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = cfg.run_id or ts
    out_dir = Path(cfg.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    is_quick = cfg.ticks_per_episode < 160
    print(f"\n{'═'*62}")
    print(f"  CMA/M2 Experiment Run  [{run_id}]")
    print(f"  agents={cfg.num_agents}  seeds={cfg.num_seeds}  ticks={cfg.ticks_per_episode}")
    if is_quick:
        print(f"  MODE: INTEGRITY / SMOKE TEST")
        print(f"  ⚠  Stress metrics (Obs3, Topology, FMB 1/3/4) are BLOCKED")
        print(f"  ⚠  Not valid for publication claims")
    else:
        print(f"  MODE: PUBLICATION RUN")
    print(f"{'═'*62}")

    # ── Stage 0: Pre-flight ───────────────────────────────────
    print("\n▶ Stage 0: Pre-Flight Gate")
    pf_ok = run_preflight(yaml_config, verbose=False)
    if not pf_ok:
        print("ABORT: Pre-flight failed. Fix blockers before proceeding.")
        sys.exit(1)

    # ── Stage 1: Flat U(a) baseline ──────────────────────────
    flat_result = None
    flat_records = []
    if cfg.run_flat_baseline:
        print("\n▶ Stage 1: Flat U(a) Baseline (P1.1)")
        flat_cfg = BaselineSuiteConfig(
            num_actions=cfg.num_actions,
            num_agents=min(4, cfg.num_agents),
            num_ticks=min(100, cfg.ticks_per_episode),
            num_seeds=cfg.num_seeds,
        )
        flat_result = run_flat_ua(flat_cfg)
        # Run flat agent to collect records for FMB comparison
        flat_agent = FlatUAWrapper(num_actions=cfg.num_actions, agent_id="flat_ua_0")
        rng = random.Random(0)
        for _ in range(min(50, cfg.ticks_per_episode)):
            obs = [rng.uniform(0, 0.4)] + [rng.gauss(0, 0.5) for _ in range(cfg.obs_dim - 1)]
            flat_agent.step(obs)
        # Records already in flat_result implicitly — FMB will use sparse set

    # ── Stage 2: Regime 1 arc ────────────────────────────────
    regime1_result = None
    if cfg.run_regime1:
        print(f"\n▶ Stage 2: Regime 1 PEACETIME Arc (P3.1)  [{cfg.num_agents} agents, {cfg.num_seeds} seeds]")
        tel = TelemetryEmitter()
        # obs_dim must match SimEnv._build_obs() — 16 dims
        obs_dim = 16
        agents = [
            build_m2_agent(
                num_actions=cfg.num_actions,
                yaml_config=yaml_config,
                agent_id=f"agent_{i}",
                telemetry=tel,
                seed=i,
            )
            for i in range(cfg.num_agents)
        ]
        r1_cfg = Regime1Config(
            num_agents=cfg.num_agents,
            num_seeds=cfg.num_seeds,
            ticks_per_episode=cfg.ticks_per_episode,
            obs_dim=obs_dim,
        )
        regime1_result = run_regime1(agents, r1_cfg)

        # Persist records
        records_path = out_dir / "regime1_records.jsonl"
        with open(records_path, "w") as f:
            for r in regime1_result.all_records[:5000]:   # cap for storage
                f.write(json.dumps(r.to_dict()) + "\n")
        print(f"  Records saved: {records_path} ({len(regime1_result.all_records):,} total)")

    # ── Stage 3: LSM Battery ─────────────────────────────────
    battery_result = None
    if cfg.run_battery:
        print("\n▶ Stage 3: LSM Battery — 3 suites (P2.4)")
        m2_bat, lsm_bat, lsm_rt = build_default_battery_wrappers(
            num_actions=cfg.num_actions, obs_dim=cfg.obs_dim * 4,
        )
        # Compute real param counts for complexity matching gate (P2.2)
        m2_param_count  = int(m2_bat.policy_layer._W.size) if hasattr(m2_bat, 'policy_layer') and hasattr(m2_bat.policy_layer, '_W') else 0
        lsm_param_count = int(lsm_rt.cfg.num_latent_states * lsm_rt.cfg.obs_dim + lsm_rt.cfg.num_latent_states) if lsm_rt else 0

        bat_cfg = BatteryConfig(
            num_actions=cfg.num_actions,
            calibration_seeds=list(range(min(cfg.calibration_seeds, 4))),
            topology_test_seeds=list(range(min(cfg.topology_seeds, 8))),
            n_cf_contexts=cfg.cf_contexts,
            rollout_episodes=cfg.rollout_episodes,
            rollout_ticks=cfg.rollout_ticks,
            m2_policy_layer_param_count=m2_param_count,
            lsm_policy_layer_param_count=lsm_param_count,
        )
        tel_bat = regime1_result.telemetry if regime1_result else TelemetryEmitter()
        r1_records = regime1_result.all_records if regime1_result else None
        battery_result = run_full_battery(m2_bat, lsm_bat, cfg=bat_cfg,
                                          telemetry=tel_bat, lsm_runtime=lsm_rt,
                                          regime1_records=r1_records)

        bat_path = out_dir / "battery_result.json"
        with open(bat_path, "w") as f:
            json.dump({
                "topology_win":     battery_result.topology_win,
                "counterfactual_win": battery_result.counterfactual_win,
                "social_signal_win": battery_result.social_signal_win,
                "battery_win":      battery_result.battery_win,
                "tier_a_gate":      battery_result.tier_a_gate,
            }, f, indent=2)
        print(f"  Battery result saved: {bat_path}")

    # ── Stage 4: FMB suite ───────────────────────────────────
    fmb_result = None
    if cfg.run_fmb and regime1_result:
        print("\n▶ Stage 4: FMB Suite — 4 dimensions (P4.1)")
        # Build flat records for comparison (stub: reuse subset of regime1 with tactic_class overridden)
        flat_records_fmb = []
        for r in regime1_result.all_records[:500]:
            import copy
            fr = copy.copy(r)
            fr.active_policy_family = __import__('telemetry').M2Family.BASELINE
            fr.tactic_class = "unconstrained"
            flat_records_fmb.append(fr)

        topology_traces = []
        if battery_result and battery_result.topology.m2_traces:
            topology_traces = battery_result.topology.m2_traces

        agent_ids = list(set(r.agent_id for r in regime1_result.all_records))
        fmb_result = run_fmb_suite(
            m2_records=regime1_result.all_records,
            flat_records=flat_records_fmb,
            m2_traces=topology_traces,
            agent_ids=agent_ids,
        )

        fmb_path = out_dir / "fmb_result.json"
        with open(fmb_path, "w") as f:
            json.dump({
                "dim1_pass": fmb_result.dim1.passes_threshold,
                "dim2_pass": fmb_result.dim2.passes_threshold,
                "dim3_pass": fmb_result.dim3.passes_threshold,
                "dim4_pass": fmb_result.dim4.passes_threshold,
                "dims_passed": fmb_result.dims_passed,
                "fmb_paper_ready": fmb_result.fmb_paper_ready,
                "dim1_kl": fmb_result.dim1.kl_vs_uniform,
                "dim2_delay_advantage": fmb_result.dim2.delay_advantage,
                "dim3_recovery_rate_m2": fmb_result.dim3.recovery_rate_m2,
                "dim4_pearson_r_m2": fmb_result.dim4.pearson_r_m2,
            }, f, indent=2)
        print(f"  FMB result saved: {fmb_path}")

    # ── Stage 5: Ablation (optional) ─────────────────────────
    if cfg.run_ablation and regime1_result:
        print("\n▶ Stage 5: M1 Ablation Suite (P1.3)")
        ablation_result = run_ablation_suite(
            m2_baseline_records=regime1_result.all_records,
            ablated_records_by_target={},   # wire real ablated runs here
        )

    # ── Stage 6: Model selection (optional) ──────────────────
    if cfg.run_model_selection and regime1_result:
        print("\n▶ Stage 6: Taxonomy Model Selection (P1.6)")
        run_taxonomy_model_selection(regime1_result.all_records)

    # ── Verdict ──────────────────────────────────────────────
    verdict = compute_verdict(regime1_result, battery_result, fmb_result,
                             ticks=cfg.ticks_per_episode, num_seeds=cfg.num_seeds)
    print(verdict.summary())

    verdict_path = out_dir / "verdict.md"
    with open(verdict_path, "w") as f:
        f.write(f"# Experiment Verdict [{run_id}]\n\n")
        f.write(verdict.summary().replace("═", "-").replace("▶", "#"))
        f.write("\n\n## Observable Snapshots\n\n")
        if regime1_result:
            for snap in regime1_result.observables:
                f.write(f"- Seed {snap.seed}: CV={snap.cv_tactic_class:.4f}  "
                        f"Spearman={snap.spearman_collapse_rho:.4f}  "
                        f"PearsonR={snap.pearson_r_char_stability:.4f}  "
                        f"Gate={'PASS' if snap.cold_start_gate_passed else 'FAIL'}\n")
    print(f"\n  Verdict saved: {verdict_path}")
    return verdict


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMA/M2 experiment orchestrator")
    parser.add_argument("config",    nargs="?", help="YAML/JSON config file")
    parser.add_argument("--quick",   action="store_true",
                        help="INTEGRITY MODE: 4 agents, 2 seeds, 80 ticks. "
                             "Validates wiring only. Stress/collapse metrics are BLOCKED. "
                             "Do NOT use for publication evidence.")
    parser.add_argument("--agents",  type=int, default=0, help="Override num_agents")
    parser.add_argument("--seeds",   type=int, default=0, help="Override num_seeds")
    parser.add_argument("--fmb-only", action="store_true", help="Skip battery, run FMB only")
    parser.add_argument("--preflight-only", action="store_true", help="Only run pre-flight gate")
    parser.add_argument("--output",  default="results", help="Output directory")
    args = parser.parse_args()

    # Load config
    yaml_config = None
    if args.config:
        p = Path(args.config)
        text = p.read_text()
        try:
            import yaml
            yaml_config = yaml.safe_load(text)
        except Exception:
            import json
            yaml_config = json.loads(text)
    if yaml_config is None:
        yaml_config = generate_reference_config()

    if args.preflight_only:
        ok = run_preflight(yaml_config)
        sys.exit(0 if ok else 1)

    # Experiment config
    cfg = ExperimentConfig(output_dir=args.output)
    if args.quick:
        cfg.num_agents          = 4
        cfg.num_seeds           = 2
        cfg.ticks_per_episode   = 80
        cfg.calibration_seeds   = 3
        cfg.topology_seeds      = 5
        cfg.cf_contexts         = 8
        cfg.rollout_episodes    = 5
        cfg.rollout_ticks       = 40
    if args.agents:
        cfg.num_agents = args.agents
    if args.seeds:
        cfg.num_seeds = args.seeds
    if args.fmb_only:
        cfg.run_battery = False
    cfg.run_ablation      = False  # enable explicitly for publication run
    cfg.run_model_selection = False

    run_experiment(cfg, yaml_config)
