# Pre-Registration: CMA / M2 Six Discriminating Observables

**Status:** PRE-REGISTERED — submit to OSF before running Regime 1 battery  
**Date:** [fill before submission]  
**Registration URL:** [fill after OSF submission]  
**Spec version:** M2 v1.18 / CMA v4.1  
**Amendment reference:** A.92 (Grok R1), §M2.12.11  

---

## Purpose

This document pre-registers the six discriminating observables that constitute
the primary evidence set for the Tier A publication claim. Observables must be
registered **before** any Regime 1 data is collected. Post-hoc labelling is
prohibited. Any deviation from these targets is a finding, not an adjustment.

If results deviate from targets, the downgrade tree (§8 of the roadmap document)
governs what can be claimed.

---

## Pre-Registered Observables

### Observable 1 — Strategic Individuality

**Measure:** CV(tactic_class) computed over per-tick stream from Regime 1 PEACETIME arc.

**Target:**
- M2: CV(tactic_class) > 0.25
- Flat U(a) baseline: CV(tactic_class) ≈ 0 (uniform distribution — no strategic specialisation)
- Delta M2 vs flat: > 0.25

**Falsification condition:** If M2 CV(tactic_class) < 0.10, the family layer is not
producing tactic specialisation. Tier A claim fails. Do not re-run with different
thresholds — this is the pre-registered target.

**Data source:** `TickRecord.tactic_class` from telemetry.py per-tick stream.

**Code:** `baseline_suite.tactic_class_cv()`

---

### Observable 2 — Costly Switching

**Measure:** (a) Budget dip on family switch tick; (b) 10% trailing performance degradation
in the 5 ticks following a switch vs the 5 ticks preceding.

**Target:**
- Switch ticks show `switch_cost_magnitude > 0` in TickRecord
- Mean performance on [t+1, t+5] after switch ≤ 0.90 × mean performance on [t-5, t-1]
- BDI baseline (no switch cost): no degradation pattern

**Falsification condition:** If trailing performance is > 0.95 × pre-switch performance,
the switch cost is not behaviourally significant. The architectural claim is weakened.

**Data source:** `TickRecord.switch_cost_paid`, `TickRecord.switch_cost_magnitude`,
performance metric (reward or goal_progress) from sim engine.

---

### Observable 3 — Stress Collapse Order

**Measure:** Spearman rank correlation between observed accessible_families_count
collapse order and biological prior.

**Biological prior (fixed, not adjustable):**
REPAIR collapses first → EXPLORE → SEEK_HELP → DECEIVE → DOMINATE → WITHDRAW → DEFEND last.

**Target:**
- M2: Spearman ρ ≥ 0.85 vs biological prior
- LSM: Spearman ρ < 0.60 (no ordering constraint → near-random)
- Uniform competence baseline: ρ ≈ 0

**Falsification condition:** If M2 ρ < 0.70, the biological A/Q threshold ordering is
not producing the expected collapse sequence. Recalibrate thresholds or accept that
the ordering is emergent rather than architectural.

**Data source:** `TickRecord.accessible_families` from per-tick stream.
Topology suite: `topology_suite.spearman_corr()`.

---

### Observable 4 — Social Legibility

**Measure:** Observer inference accuracy lift from public signal.

**Protocol:**
- Train three observers per agent (ctx_only, ctx_prev, ctx_prev_signal)
- Key metric: lift = acc(ctx_prev_signal) − acc(ctx_prev)
- M2 vs LSM lift advantage: delta_lift_advantage ≥ 0.05 (5pp)

**Target:**
- M2 lift > LSM lift by ≥ 5pp
- M2 acc(ctx_prev_signal) ≥ 0.55 (above chance 1/7 ≈ 0.14)
- LSM pseudo-signal provides < 5pp advantage (latent cluster not architecturally legible)

**Falsification condition:** If delta_lift_advantage < 0, LSM signal is MORE legible
than M2 signal. This is a fundamental failure of the social legibility claim.

**Data source:** Social signal suite: `social_signal_suite.test_latent_vs_m2_social_signal_suite()`.

---

### Observable 5 — Character Formation (Temporal Stability)

**Measure:** Pearson r of family transition matrix across Regime 1 episodes.

**Protocol:**
- Fit transition matrix T_N at generation N (early episodes)
- Fit transition matrix T_{N+3} at generation N+3 (later episodes, same agent)
- Pearson r between flattened T_N and T_{N+3}

**Target:**
- M2: Pearson r ≥ 0.70 (stable character — transition probabilities persist)
- Flat U(a): Pearson r ≈ 0 (no persistent character — each episode independent)
- LSM: intermediate (persistent latent state but no semantic anchoring)

**Falsification condition:** If M2 r < 0.50, phenotype drift is too aggressive or
phenotype initialisation is not stable. The character formation claim fails.

**Data source:** Per-episode `TickRecord.active_policy_family` sequences.
Compute transition matrix from family sequence. Pearson r between matrices.

---

### Observable 6 — Cultural Transmission (DEFERRED — Regime 3)

**Measure:** KL divergence between student tactic distribution and teacher prior vs
content-null control.

**Target:**
- KL(student || teacher) − KL(student || content_null) ≥ 0.15
- Four-condition anti-false-positive protocol (§9.6) required before claim

**Status:** Tier C claim. Requires Regime 3 (multi-generation teaching chain).
**This is a second-paper claim. Do NOT attempt to collect during Phase 3.**

Pre-registered here for completeness only. Collection before Regime 3 is invalid.

---

## Analysis Plan

### Primary analysis
Run Regime 1 PEACETIME arc (≥ 32 agents, ≥ 3 seeds) and collect Observables 1–5.
Run LSM battery (topology + counterfactual + social signal suites).

### Planned comparisons
| Comparison | Test | Pre-registered threshold |
|---|---|---|
| M2 CV vs flat U(a) | One-sided t-test | p < 0.05, delta > 0.25 |
| M2 Spearman vs LSM Spearman | One-sided | M2 > LSM AND M2 ≥ 0.85 |
| Battery topology win | JS > 0.10 | Pre-specified in §M2.12.11 |
| Social signal lift advantage | Difference in lifts | ≥ 5pp |
| Character stability (Obs 5) | Pearson r | r ≥ 0.70, n_agents ≥ 16 |

### Stopping rules
- All six discriminating observables must be collected before drawing conclusions
  (except Obs 6 which is Regime 3 only)
- If cold-start gate fails (world_model_error ≥ 0.30 at tick 150): STOP.
  Fix replay buffer and rerun before proceeding to Regime 2

### Deviations
Any deviation from these targets must be:
1. Reported in the paper with the pre-registered target and actual result
2. Classified as a finding (if unexpected) or a specification failure (if systematic)
3. Never silently re-labelled as a "different metric" in the results section

---

## Excluded Analyses

The following were considered and explicitly excluded from primary analysis:

- **Reward maximisation**: Not a primary metric. M2 may sacrifice expected reward for
  strategic persistence. Reward is a secondary outcome measure, not a falsification target.
- **Wall-clock performance**: Runtime efficiency is not a validity criterion for Tier A.
- **Population-level welfare**: Individual agent behaviour is the measurement unit here.
  Population-level dynamics are Tier B / Regime 2 territory.
- **Observable 3 with non-biological prior**: The biological prior is fixed at the values
  above. Alternative orderings cannot be substituted if results are unsatisfactory.

---

## Counterfactual Probe Parameters (FROZEN 2026-03-11)

These parameters were not in the original pre-registration. They are added here
before any publication run. They are locked from this point.

| Parameter | Value | Rationale |
|---|---|---|
| `obs_noise_std` | 0.15 | Gaussian obs perturbation per repeat — breaks determinism |
| `cf_temperature` | 0.25 | Softmax temperature — prevents point-mass distributions |
| `obs_channel_amplitude` | 0.35 | Per-action obs-varying logit weight |
| `n_repeats` | 3 | Repeats per context per forced state |
| `probe_source` | `regime1_obs_raw` when available; else `generate_probe_contexts(seed)` | |

**Rationale for values:**
- `obs_channel_amplitude = 0.35` was chosen before seeing publication data.
  It was tuned during the integrity test session (2026-03-11) to produce nontrivial
  discrimination between M2 and LSM under a structurally correct test design.
  The 0.60 initial value was degenerate (drowned the family semantic signal).
  The 0.20 value was also degenerate (insufficient per-action variation).
  0.35 is the pre-registered value. It may NOT be changed after the publication run.
- Temperature 0.25 is a standard low-temperature softmax that preserves rank order
  while creating a non-trivial distribution shape. It is not tuned to outcome.

**Disclosure:**
The amplitude value (0.35) was selected during an integrity testing session before
any publication data was collected. The selection criterion was structural correctness
of the test (nontrivial per-action discrimination), not optimisation toward a
hypothesis-confirming result. This must be disclosed in the paper.

---

## Verdict Semantics (FROZEN 2026-03-11)

The five status labels used in verdict output are:

| Label | Meaning |
|---|---|
| `PASS` | Measured, threshold met, run config sufficient |
| `FAIL` | Measured, threshold NOT met, run config sufficient — genuine negative finding |
| `BLOCKED_BY_REGIME_LENGTH` | Structurally impossible to collect in the run config chosen |
| `NOT_TRIGGERED` | Event could in principle have occurred; did not in this run |
| `INVALIDATED_BY_BUG` | Value from prior run contaminated by known implementation bug |

**BLOCKED vs NOT_TRIGGERED distinction:**
- `BLOCKED`: The run config makes collection structurally impossible.
  E.g. Obs 3 with 80-tick episodes — shocks at t=150/300 cannot fire.
- `NOT_TRIGGERED`: The event *could* have occurred but did not.
  E.g. FMB Dim 2 onset delay when zero failure events were observed in a valid full run.

This distinction matters for peer review. A BLOCKED metric is not a negative finding.
A NOT_TRIGGERED metric is informative (the event did not occur) but not falsifying
without further investigation.

**Verdict aggregation rule (frozen — see verdict_contract.md Part 3):**
The downgrade tree is first-match, evaluated in order:
1. FULL_TIER_A: battery_win AND tier_a_gate AND obs1_pass AND obs3_pass
2. DOWNGRADE_1: battery_win AND NOT tier_a_gate (correlational claim only)
3. DOWNGRADE_2: NOT battery_win AND fmb_ready (benchmark contribution)
4. DOWNGRADE_3_WITH_SIGNAL: NOT battery_win AND any non-blocked PASS exists
5. DOWNGRADE_3: no publishable claim

BLOCKED metrics are treated as ABSENT in aggregation (do not promote, do not penalise).
NOT_TRIGGERED metrics are treated as informative absence (not a PASS, not a FAIL).

---

## Registration Checklist

Before submitting to OSF:

- [ ] All 6 observable targets recorded in this document (done above)
- [ ] Analysis plan section complete
- [ ] Stopping rules documented
- [ ] Battery config (BatteryConfig) frozen and committed to repo
- [ ] yaml_validator.py passes on current config
- [ ] causal_graph.md committed to repo
- [ ] Motif 5 (NARRATIVE) wired in goal_stack update logic
- [ ] This document committed to repo at: `docs/preregistration.md`
- [ ] OSF registration submitted with DOI recorded here: [DOI]

**Sign-off:** Register this document at OSF before running any Regime 1 episode.
Regression to pre-registration is not valid.