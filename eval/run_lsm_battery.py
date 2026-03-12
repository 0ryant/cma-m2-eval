"""
run_lsm_battery.py  —  P2.4 + P2.2 + P2.5
============================================
Orchestrates all three hostile-control suites in dependency order.
Enforces complexity matching gate (P2.2) and OVERRIDE+SCORE_WIN gate (P2.5).

Build order (Chat R1 / A.95):
  flat_ua → m2_minimal → lsm → battery suites → FMB

Battery paper claim:
  M2 wins against the LSM hostile baseline if and only if it wins on ALL THREE:
    (1) topology_win      — collapse structure and seed stability
    (2) counterfactual_win — causal coherence of forced states
    (3) social_signal_win  — legible public signal lift

Tier A publication rule (§M2.7.6):
  OVERRIDE + SCORE_WIN fraction > 0.70 from Regime 1 PRECEDENCE_TAG events.
  battery_win alone is not sufficient — causal attribution requires this gate.

numpy-only. No torch.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from agent_wrapper import AgentWrapper, M2AgentWrapper, LSMAgentWrapper, FlatUAWrapper
from lsm_model import LSMConfig, LSMRuntime, LSMModel, check_complexity_match
from topology_suite import (
    test_latent_vs_m2_topology_suite, LatentVsM2TopologySuiteResult,
    CatastropheSchedule,
)
from counterfactual_suite import (
    test_latent_vs_m2_counterfactual_suite, CounterfactualSuiteResult,
)
from social_signal_suite import (
    test_latent_vs_m2_social_signal_suite, SocialSignalSuiteResult, RolloutConfig,
)
from telemetry import TelemetryEmitter


# ──────────────────────────────────────────────────────────────
# Battery config
# ──────────────────────────────────────────────────────────────

@dataclass
class BatteryConfig:
    # Shared
    num_actions:         int        = 128
    seed:                int        = 0

    # Topology suite
    calibration_seeds:   List[int]  = field(default_factory=lambda: list(range(8)))
    topology_test_seeds: List[int]  = field(default_factory=lambda: list(range(16)))
    calibration_steps:   int        = 80
    calibration_rd:      float      = 0.10

    # Counterfactual suite
    n_cf_contexts:       int        = 20
    n_cf_repeats:        int        = 3

    # Social signal suite
    rollout_episodes:    int        = 20
    rollout_ticks:       int        = 100

    # Complexity matching (P2.2) — MUST set before publication run
    m2_policy_layer_param_count:  int = 0
    lsm_policy_layer_param_count: int = 0
    complexity_tolerance:         float = 0.15

    # Catastrophe schedule
    warmup_ticks:    int   = 30
    ramp_ticks:      int   = 60
    peak_ticks:      int   = 80
    recovery_ticks:  int   = 40
    rd_peak:         float = 0.95


# ──────────────────────────────────────────────────────────────
# Complexity matching gate (P2.2)
# ──────────────────────────────────────────────────────────────

def run_complexity_check(cfg: BatteryConfig, lsm_runtime: Optional[LSMRuntime] = None) -> None:
    """
    Raise ValueError if LSM exceeds M2 param count by > tolerance.
    Warn if counts are unset.
    If lsm_runtime is provided, auto-compute LSM param count.
    """
    lsm_count = cfg.lsm_policy_layer_param_count
    if lsm_runtime is not None and lsm_count == 0:
        lsm_count = lsm_runtime.param_count()

    if cfg.m2_policy_layer_param_count == 0 or lsm_count == 0:
        print("⚠  WARNING: param counts not set — complexity matching cannot be enforced.")
        print("   Set BatteryConfig.m2_policy_layer_param_count and lsm_policy_layer_param_count before publication run.")
        return

    ok, msg = check_complexity_match(cfg.m2_policy_layer_param_count, lsm_count, cfg.complexity_tolerance)
    print(f"\n── Complexity Matching (P2.2) ──────────────────")
    print(f"  {msg}")
    if not ok:
        raise ValueError(f"Complexity mismatch: {msg}. Reduce LSM capacity before running battery.")


# ──────────────────────────────────────────────────────────────
# OVERRIDE+SCORE_WIN gate (P2.5)
# ──────────────────────────────────────────────────────────────

def check_tier_a_gate(telemetry: TelemetryEmitter) -> Dict[str, Any]:
    """
    §M2.7.6: OVERRIDE + SCORE_WIN fraction > 0.70 required for Tier A.
    Call with telemetry from a completed Regime 1 run.
    """
    frac = telemetry.precedence_fraction()
    print(f"\n── Tier A Gate (P2.5) ──────────────────────────")
    print(f"  OVERRIDE:         {frac['override']:.3f}")
    print(f"  SCORE_WIN:        {frac['score_win']:.3f}")
    print(f"  WEIGHTED_ARB:     {frac['weighted_arb']:.3f}")
    print(f"  Strong fraction:  {frac['strong_fraction']:.3f}  (target > 0.70)")
    print(f"  Gate: {'PASS ✓' if frac['tier_a_gate_pass'] else 'FAIL ✗ — cannot publish Tier A mechanism claim'}")
    return frac


# ──────────────────────────────────────────────────────────────
# Full battery result
# ──────────────────────────────────────────────────────────────

@dataclass
class BatteryResult:
    topology:              LatentVsM2TopologySuiteResult
    counterfactual:        CounterfactualSuiteResult
    social_signal:         SocialSignalSuiteResult
    topology_win:          bool
    counterfactual_win:    bool
    social_signal_win:     bool
    battery_win:           bool     # all three
    partial_win:           bool     # exactly two
    tier_a_gate:           Optional[Dict[str, Any]] = None

    def summary(self) -> str:
        lines = ["\n" + "═" * 60]
        lines.append("  BATTERY RESULT")
        lines.append("═" * 60)
        lines.append(f"  Topology win:        {'YES ✓' if self.topology_win else 'NO ✗'}")
        lines.append(f"  Counterfactual win:  {'YES ✓' if self.counterfactual_win else 'NO ✗'}")
        lines.append(f"  Social signal win:   {'YES ✓' if self.social_signal_win else 'NO ✗'}")
        lines.append("  " + "-" * 56)
        if self.battery_win:
            lines.append("  BATTERY WIN — Tier A evidence set complete.")
        elif self.partial_win:
            lines.append("  PARTIAL WIN — 2/3 suites. Weakened Tier A claim. See downgrade tree.")
        else:
            wins = sum([self.topology_win, self.counterfactual_win, self.social_signal_win])
            lines.append(f"  BATTERY LOSS ({wins}/3). Review downgrade tree. Do not submit Tier A.")
        if self.tier_a_gate:
            g = self.tier_a_gate
            lines.append(f"\n  Tier A gate (OVERRIDE+SCORE_WIN): {g['strong_fraction']:.3f}  {'PASS ✓' if g['tier_a_gate_pass'] else 'FAIL ✗'}")
        lines.append("═" * 60)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────

def run_full_battery(
    m2_wrapper:  AgentWrapper,
    lsm_wrapper: AgentWrapper,
    cfg:         BatteryConfig = None,
    telemetry:   Optional[TelemetryEmitter] = None,
    lsm_runtime: Optional[LSMRuntime] = None,
    regime1_records = None,    # List[TickRecord] from Regime 1; used as real CF probe obs
) -> BatteryResult:
    """
    Run the full three-suite battery.
    Topology suite runs first — its calibration permutation feeds the other suites.

    Args:
        m2_wrapper:      M2 agent wrapper
        lsm_wrapper:     LSM agent wrapper
        cfg:             battery configuration
        telemetry:       TelemetryEmitter from Regime 1 run (for Tier A gate check)
        lsm_runtime:     LSMRuntime (for auto param count)
        regime1_records: TickRecord list from Regime 1. If provided, obs from these
                         records are used as counterfactual probe contexts instead of
                         random Gaussian vectors. Real obs are semantically diverse
                         (varying rd, urgency, scarcity) and enable meaningful
                         within-state coherence and between-state margin comparisons.
    """
    if cfg is None:
        cfg = BatteryConfig()

    # P2.2: Complexity check
    run_complexity_check(cfg, lsm_runtime)

    schedule = CatastropheSchedule(
        warmup_ticks=cfg.warmup_ticks,
        ramp_ticks=cfg.ramp_ticks,
        peak_ticks=cfg.peak_ticks,
        recovery_ticks=cfg.recovery_ticks,
        rd_peak=cfg.rd_peak,
    )

    # Suite 1: Topology
    print("\n" + "─" * 60)
    print("  Suite 1: Topology")
    print("─" * 60)
    topology_result = test_latent_vs_m2_topology_suite(
        m2_wrapper=m2_wrapper,
        lsm_wrapper=lsm_wrapper,
        num_actions=cfg.num_actions,
        calibration_seeds=cfg.calibration_seeds,
        test_seeds=cfg.topology_test_seeds,
        calibration_steps=cfg.calibration_steps,
        calibration_rd=cfg.calibration_rd,
        schedule=schedule,
    )

    # Suite 2: Counterfactual
    print("\n" + "─" * 60)
    print("  Suite 2: Counterfactual")
    print("─" * 60)
    # Build probe contexts from real Regime 1 obs when available
    # Random Gaussian probes cannot show M2 semantic advantage because they lack
    # the structured variation (rd, urgency, scarcity) that M2 families respond to.
    cf_probe_obs = None
    if regime1_records is not None and len(regime1_records) > 0:
        from counterfactual_suite import ProbeContext
        rng_cf = __import__("random").Random(cfg.seed)
        # Filter records that have obs_raw stored (wired in run_seed)
        records_with_obs = [r for r in regime1_records if getattr(r, "obs_raw", None)]
        if records_with_obs:
            sample = rng_cf.sample(records_with_obs,
                                   min(cfg.n_cf_contexts * 4, len(records_with_obs)))
            cf_probe_obs = []
            for r in sample[:cfg.n_cf_contexts]:
                cf_probe_obs.append(ProbeContext(
                    obs=r.obs_raw, rd=r.regression_depth,
                    goal_stack_type="NEUTRAL", seed=rng_cf.randint(0, 10000)
                ))

    cf_result = test_latent_vs_m2_counterfactual_suite(
        m2_wrapper=m2_wrapper,
        lsm_wrapper=lsm_wrapper,
        n_contexts=cfg.n_cf_contexts,
        n_repeats=cfg.n_cf_repeats,
        seed=cfg.seed,
        probe_contexts=cf_probe_obs,   # None → generates random contexts
    )

    # Suite 3: Social Signal
    print("\n" + "─" * 60)
    print("  Suite 3: Social Signal")
    print("─" * 60)
    rollout_cfg = RolloutConfig(
        num_episodes=cfg.rollout_episodes,
        episode_ticks=cfg.rollout_ticks,
    )
    ss_result = test_latent_vs_m2_social_signal_suite(
        m2_wrapper=m2_wrapper,
        lsm_wrapper=lsm_wrapper,
        rollout_cfg=rollout_cfg,
        num_actions=cfg.num_actions,
        seed=cfg.seed,
    )

    # P2.5: Tier A gate
    tier_a = None
    if telemetry is not None:
        tier_a = check_tier_a_gate(telemetry)

    # Verdict
    tw = topology_result.metrics.topology_win
    cw = cf_result.counterfactual_win
    sw = ss_result.social_signal_win
    wins = sum([tw, cw, sw])

    result = BatteryResult(
        topology=topology_result,
        counterfactual=cf_result,
        social_signal=ss_result,
        topology_win=tw,
        counterfactual_win=cw,
        social_signal_win=sw,
        battery_win=(wins == 3),
        partial_win=(wins == 2),
        tier_a_gate=tier_a,
    )
    print(result.summary())
    return result


# ──────────────────────────────────────────────────────────────
# Convenience: build default wrappers from LSM config
# ──────────────────────────────────────────────────────────────

def build_default_battery_wrappers(
    num_actions: int = 128,
    obs_dim:     int = 64,
    seed:        int = 0,
) -> tuple:
    """
    Build M2 and LSM wrappers with real M2MinimalPolicy injected.
    The M2 wrapper uses biologically-grounded scoring; LSM uses the 4-equation latent model.
    """
    from m2_policy import build_m2_agent, build_lsm_agent
    from yaml_validator import generate_reference_config
    from telemetry import TelemetryEmitter

    yaml_cfg = generate_reference_config()
    tel = TelemetryEmitter()

    lsm_cfg = LSMConfig(num_actions=num_actions, obs_dim=obs_dim, seed=seed)
    lsm_rt  = LSMRuntime(lsm_cfg)
    lsm_mdl = LSMModel(lsm_rt)

    m2  = build_m2_agent(num_actions=num_actions, yaml_config=yaml_cfg,
                          agent_id="battery_m2", telemetry=tel, seed=seed)
    lsm = LSMAgentWrapper(num_actions=num_actions, lsm_model=lsm_mdl)

    return m2, lsm, lsm_rt


if __name__ == "__main__":
    print("Building default wrappers (stub agents — replace with real sim engine)...")
    m2, lsm, lsm_rt = build_default_battery_wrappers(num_actions=20)

    cfg = BatteryConfig(
        num_actions=20,
        calibration_seeds=list(range(3)),
        topology_test_seeds=list(range(5)),
        n_cf_contexts=10,
        rollout_episodes=5,
        rollout_ticks=40,
    )

    result = run_full_battery(m2, lsm, cfg=cfg, lsm_runtime=lsm_rt)
