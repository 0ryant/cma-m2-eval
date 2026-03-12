# CMA / M2 Verdict Contract — FROZEN
## Version 1.0 — 2026-03-11 — LOCKED BEFORE PUBLICATION RUN

**Status:** FROZEN. No changes to this document or to the code that implements it
after the first 500-tick publication run begins. Any post-run change is epistemically
dirty and must be disclosed as a deviation in the paper.

---

## Part 1 — Metric Definitions and Thresholds

All thresholds are fixed. None may be adjusted after data collection begins.

### Observable 1 — Strategic Individuality
- **Measure:** `mean(cv_tactic_class)` across seeds, from Regime 1 PEACETIME arc
- **Threshold:** > 0.25
- **Code:** `baseline_suite.tactic_class_cv(records)`
- **Required regime length:** Any — observable is valid in all run modes

### Observable 2 — Costly Switching
- **Measure (a):** `mean(switch_frequency)` — fraction of ticks with `switch_cost_paid=True`
- **Measure (b):** `mean(post_switch_degradation)` — trailing perf drop ≤ −0.10
- **Threshold (a):** > 0.01 (switching occurs with nontrivial frequency)
- **Threshold (b):** ≤ −0.10 (10% trailing degradation window)
- **Code:** `baseline_suite.switch_frequency()`, `regime1_runner.compute_post_switch_degradation()`
- **Required regime length:** Any

### Observable 3 — Stress Collapse Order
- **Measure:** Spearman ρ between observed family collapse order and biological prior
- **Biological prior (fixed, not adjustable):** REPAIR → EXPLORE → SEEK_HELP → DECEIVE → DOMINATE → WITHDRAW → DEFEND
- **Threshold:** M2 ρ ≥ 0.85; LSM ρ < 0.60
- **Code:** `topology_suite.spearman_corr()`
- **Required regime length:** MUST include at least one stress shock (ticks ≥ 160 with shock at tick 150)
- **Status in quick mode (80t):** `BLOCKED_BY_REGIME_LENGTH` — stress shocks at t=150/300 never fire

### Observable 4 — Social Legibility
- **Measure:** `lift_advantage = m2_lift − lsm_lift` where `lift = acc(+signal) − acc(ctx_prev)`
- **Threshold:** ≥ 0.05 (5pp M2 lift advantage over LSM)
- **Code:** `social_signal_suite.test_latent_vs_m2_social_signal_suite()`
- **Required regime length:** Any — uses rollout episodes, not Regime 1 arc directly

### Observable 5 — Character Stability
- **Measure:** Pearson r of agent transition matrices between episode N and N+3
- **Threshold:** M2 ≥ 0.70; Flat U(a) ≈ 0
- **Code:** `regime1_runner.compute_character_stability()`
- **Required regime length:** Minimum 3 episodes × ticks_per_episode ticks total; 3 seeds required

### Tier A Gate (P2.5)
- **Measure:** `(OVERRIDE + SCORE_WIN) / total_events` from PRECEDENCE_TAG stream
- **Threshold:** > 0.70
- **Code:** `run_lsm_battery.check_tier_a_gate(telemetry)`
- **Required regime length:** Any — but note: quick mode (no stress) will produce
  near-100% SCORE_WIN, which is *not* a meaningful Tier A signal. The Tier A gate
  is only informative under stress conditions where OVERRIDE pressure exists.
  **Quick mode value should be treated as `NOT_TRIGGERED`, not PASS.**

### Battery — Topology Win
- **Measure:** Spearman vs bio prior (M2 ≥ 0.85) AND JS(M2, LSM) > 0.10 AND M2 volatility < LSM volatility
- **State entropy sub-condition:** M2 state entropy > 1.0 nats (requires sufficient stress-driven transitions)
- **Code:** `topology_suite.compute_topology_metrics()`
- **Required regime length:** MUST include stress shocks. State entropy condition requires
  sustained multi-state cycling under rd ≥ 0.40.
  **Quick mode (80t) state entropy = `BLOCKED_BY_REGIME_LENGTH`**

### Battery — Counterfactual Win
- **Measure:** M2 `mean_within_coherence > LSM mean_within_coherence` AND
  M2 `mean_between_margin > LSM mean_between_margin`
- **Probe parameters (FROZEN):**
  - `obs_noise_std = 0.15` (Gaussian perturbation per repeat)
  - `temperature = 0.25` (softmax temperature for distribution shaping)
  - `obs_channel_amplitude = 0.35` (per-action obs-varying logit weight)
  - `n_repeats = 3` per context per state
  - `probe_source = regime1_obs_raw` when Regime 1 records have `obs_raw` populated;
    otherwise `generate_probe_contexts(seed=battery_cfg.seed)`
- **Code:** `counterfactual_suite.probe_agent()`
- **Required regime length:** Any

### Battery — Social Signal Win
- **Measure:** M2 lift advantage ≥ 0.05 (same as Observable 4)
- **Required regime length:** Any

### FMB Suite
- **Dim 1:** KL(empirical collapse dist || uniform) > 0.30
- **Dim 2:** M2 failure-onset delay vs flat ≥ +5 ticks
- **Dim 3:** Fraction failures resolved within 30 ticks: M2 ≥ 0.60
- **Dim 4:** Pearson r of failure onset times across agents: M2 r < 0.30
- **Failure event definition (FROZEN):** `BASELINE_LOCK_IN` = ≥ 20 consecutive BASELINE
  ticks during period where `rd ≥ 0.40`
- **Code:** `fmb_suite.detect_failure_events()`, `fmb_suite.run_fmb_suite()`
- **Required regime length:** MUST include stress phase with rd ≥ 0.40 sustained ≥ 20 ticks.
  Dim 1, 3, 4 are `BLOCKED_BY_REGIME_LENGTH` in quick mode (shocks at t=150/300 never fire).
  Dim 2 can return a value (1000t advantage = no failures detected = `NOT_TRIGGERED`,
  not a real advantage signal).

---

## Part 2 — Status Label Rules

Every metric in the verdict gets exactly one of these five labels. Rules are exhaustive
and mutually exclusive. Apply the first matching rule.

### PASS
- Metric was measured
- Threshold was met
- Run configuration satisfies the required regime length for this metric

### FAIL
- Metric was measured
- Threshold was NOT met
- Run configuration satisfies the required regime length for this metric
- The failure is a genuine negative finding about the architecture

### BLOCKED_BY_REGIME_LENGTH
- The run configuration makes it structurally impossible to collect this metric
- The stress event, episode count, or tick count is definitionally insufficient
- Examples:
  - Obs 3 (Spearman) with ticks_per_episode < 160 (shock at t=150 never fires)
  - Topology entropy with ticks_per_episode < 160
  - FMB Dims 1/3/4 with ticks_per_episode < 160
  - Obs 5 (character stability) with num_seeds < 3 or episodes < 3 per agent
  - Tier A gate as an informative signal in quick mode (no stress → no OVERRIDE pressure)
- **A BLOCKED metric is NOT a negative finding. It is unmeasured.**

### NOT_TRIGGERED
- The run configuration could in principle have produced the event
- The event did not occur in this run (contingent absence)
- Examples:
  - FMB Dim 2 "advantage" = 1000t when zero failure events detected
    (failures could have occurred; they didn't — not a real advantage signal)
  - Switch frequency = 0 in a valid full run with stress (switches could have occurred)
  - OVERRIDE events = 0 in a full run (override triggers could have fired)
- **A NOT_TRIGGERED metric is informative but not falsifying without further investigation**

### INVALIDATED_BY_BUG
- A known implementation bug in a previous run produced a value for this metric
- That value must not be carried forward as evidence
- Current bugs known to have produced invalid values:
  - switch_frequency = 0.0000 in all sessions before 2026-03-11 (switch telemetry disconnected)
  - precedence_strong_fraction = 0.000 in all sessions before 2026-03-11 (telemetry emitter not shared)
  - Tier A gate = FAIL in all sessions before 2026-03-11 (same root cause as above)
  - Counterfactual coherence = 1.000/1.000 (M2=LSM) before 2026-03-11 (logit degeneracy)
- **These values must never be cited as evidence in the paper**

---

## Part 3 — Verdict Aggregation Rule

This is the downgrade tree. It is locked. The inputs, conditions, and output labels
are fixed. No new conditions may be added after the publication run.
```
INPUTS (all boolean):
  battery_win     = topology_win AND counterfactual_win AND social_signal_win
  tier_a_gate     = (OVERRIDE + SCORE_WIN) / total > 0.70  [informative under stress only]
  obs1_pass       = CV(tactic_class) > 0.25
  obs3_pass       = Spearman ρ ≥ 0.85  [requires stress; else BLOCKED]
  obs5_pass       = Pearson r ≥ 0.70   [requires 3 seeds/episodes; else BLOCKED]
  fmb_ready       = all 4 FMB dims pass [requires stress; else BLOCKED for dims 1/3/4]

RULE (evaluate top-to-bottom, first match wins):
  IF battery_win AND tier_a_gate AND obs1_pass AND obs3_pass
    → FULL_TIER_A

  IF battery_win AND NOT tier_a_gate
    → DOWNGRADE_1
    (battery win without mechanism gate — correlational claim only)

  IF NOT battery_win AND fmb_ready
    → DOWNGRADE_2
    (FMB benchmark contribution; no full architecture claim)

  IF NOT battery_win AND NOT fmb_ready AND (obs1_pass OR obs3_pass OR obs5_pass)
    → DOWNGRADE_3_WITH_SIGNAL
    (partial signals exist; characterise what was found; return to Phase 1 before Tier A)

  ELSE
    → DOWNGRADE_3
    (no publishable claim; return to Phase 1)
```

**Note on BLOCKED metrics in aggregation:**
- A BLOCKED metric is treated as ABSENT (not as FAIL) in the aggregation rule
- Example: if obs3_pass cannot be evaluated (BLOCKED_BY_REGIME_LENGTH), the FULL_TIER_A
  condition becomes unevaluable — do not promote to FULL_TIER_A and do not penalise
  to DOWNGRADE_3. Report as "requires full publication run"

---

## Part 4 — Run Configuration Contract

The following parameters are fixed for the publication run. Any deviation must be
documented as a deviation, not silently applied.
```
num_agents          = 32
num_seeds           = 3
ticks_per_episode   = 500
stress_shock_ticks  = [150, 300]
stress_shock_dur    = 8
rd_ceiling          = 0.40  (PEACETIME)
n_cf_contexts       = 20
n_cf_repeats        = 3
calibration_seeds   = 8
topology_seeds      = 16
rollout_episodes    = 20
rollout_ticks       = 100
obs_noise_std       = 0.15
cf_temperature      = 0.25
obs_channel_amp     = 0.35
failure_event_def   = BASELINE_LOCK_IN: ≥20 consecutive BASELINE ticks, rd ≥ 0.40
```

**Quick mode (--quick) is NOT a valid configuration for publication data.**
Quick mode is explicitly integrity-only (see Part 5).

---

## Part 5 — Quick Mode Classification

`--quick` mode parameters: 4 agents, 2 seeds, 80 ticks.

Quick mode is valid ONLY for:
- Pre-flight gate verification
- Telemetry wire integrity
- Module import smoke test
- Parameter parity check
- Counterfactual probe sanity (non-zero discrimination)
- Switch telemetry sanity (nonzero frequency)
- Precedence telemetry sanity (nonzero fraction)

Quick mode is INVALID for:
- Topology win/loss determination
- FMB dimension pass/fail determination
- Observable 3 (Spearman collapse order)
- Observable 5 (character stability)
- Tier A gate as an informative signal
- Any claim in the downgrade tree

Any metric from a quick run that falls under "INVALID" above must be labelled
`BLOCKED_BY_REGIME_LENGTH` in the verdict output, regardless of whether the
numeric value looks positive or negative.

---

## Part 6 — Deviation Protocol

If any of the above must be changed after the publication run begins:

1. Record the deviation in `deviations.md` (create if absent) with:
   - date
   - what changed
   - why
   - which metrics are affected
   - whether the change was pre-run or post-run (post-run = must be disclosed)

2. Post-run changes to thresholds, metric definitions, or aggregation rules are:
   - permissible only if the change is disclosed in the paper
   - never permissible if the change moves a FAIL to a PASS

3. Post-run changes to probe parameters (obs_noise_std, temperature, amplitude) are:
   - not permissible for the primary analysis
   - permissible for a clearly-labelled supplementary sensitivity analysis only

---

*This document must be committed to the repository before the first 500-tick run begins.*
*After that point, it is read-only.*