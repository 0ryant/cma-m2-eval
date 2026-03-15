# CMA/M2 Pre-Registration Completion Report

**OSF Registration:** https://osf.io/gesyh (DOI: 10.17605/OSF.IO.GESYH)  
**Registered:** 2026-03-12 (before publication run)  
**Completion date:** 2026-03-12  
**Source code:** https://github.com/0ryant/cma-m2-eval  
**Paper:** Submitted to WOA 2026 (27th Workshop "From Objects to Agents")

---

## Purpose

This document records the outcome of each pre-registered observable against its registered target and falsification condition. It is posted to the OSF project to close the registration and enable independent verification.

---

## Observable Outcomes

### Observable 1 — Strategic Individuality (CV of tactic class)

| | |
|---|---|
| **Registered target** | M2 CV(tactic_class) > 0.25 |
| **Falsification** | CV < 0.10 |
| **Result** | CV = 2.118 |
| **Verdict** | **PASS** |

### Observable 2 — Costly Switching

| | |
|---|---|
| **Registered target (2a)** | switch_cost_magnitude > 0 on switch ticks |
| **Registered target (2b)** | Mean post-switch performance ≤ 0.90 × pre-switch |
| **Falsification** | Post-switch performance > 0.95 × pre-switch |
| **Result (2a)** | Switch freq = 0.007/agent-tick |
| **Result (2b)** | Post-switch degradation = 0.033 (underpowered; ~30 wrong-family events vs ~100 needed) |
| **Verdict** | **UNDERPOWERED** — power analysis indicates ≥106 agents required at 900 ticks |

### Observable 3 — Stress Collapse Order (PRIMARY)

| | |
|---|---|
| **Registered target** | M2 Spearman ρ ≥ 0.85 vs biological prior |
| **Falsification** | ρ < 0.70 |
| **Result** | ρ = 0.878 ± 0.009 (10 seeds, 95% CI [0.872, 0.884]) |
| **Min per-seed** | 0.862 (seed 4) |
| **Verdict** | **PASS** — all 10 seeds above target; falsification bound cleared by 0.162 |

**Note on measurement correction:** The original run (20260312_024959) produced a failing Obs 3 due to three implementation bugs in the Spearman pipeline: (1) temporal misalignment (one-tick rd lag), (2) invalid NaN aggregation in per-agent averaging, (3) sequential tie assignment instead of midrank. These were corrected to match the registered statistical object (population-level Spearman with midrank ties). No observables, thresholds, W matrix, sensitivities, or architectural parameters were changed. Both runs are archived with exact code diffs at https://github.com/0ryant/cma-m2-eval.

**Collapse-ordering ablation (post-registration extension):** Seven alternative threshold orderings (reversed, uniform, 5 random permutations) were tested under the same catastrophe schedule. Only the registered biological ordering achieves ρ ≥ 0.85 (actually ρ = 1.000). Alternatives: mean ρ = 0.194.

### Observable 4 — Social Signal Lift

| | |
|---|---|
| **Registered target** | M2 lift > LSM lift by ≥ 5pp |
| **Falsification** | delta_lift_advantage < 0 (LSM more legible than M2) |
| **Result** | M2 lift = +0.252, LSM lift = +0.628, delta = −0.363 |
| **Verdict** | **FAIL (FALSIFIED)** — LSM signal is more legible than M2 |

**Interpretation:** This is the paper's most important secondary finding. It reveals a trade-off: hard-gated typed families produce crisp internal state transitions but abrupt, hard-to-predict external signals. The LSM's softmax produces graded transitions that external observers can track. Addressing this through a graded anticipatory signalling channel is the design objective for M3.

### Observable 5 — Character Stability

| | |
|---|---|
| **Registered target** | M2 Pearson r ≥ 0.70 |
| **Falsification** | r < 0.50 |
| **Result (aggregate)** | r = 0.897 (mean across 10 seeds) |
| **Result (per-seed)** | σ = 0.106; bimodal distribution. Seed 3: r = 0.686 (below 0.70 target) |
| **Verdict** | **PASS (AGGREGATE) / FRAGILE** — mean passes; one seed fails per-seed target |

**Note:** The bimodal pattern (6 seeds at 0.88–1.00, 4 seeds at 0.69–0.83) likely reflects interaction between Watts-Strogatz graph topology and synchronised stress exposure. This hypothesis is untested.

### Observable 6 — Counterfactual Coherence

| | |
|---|---|
| **Registered target** | M2 within-state coherence > LSM |
| **Result** | M2 = 0.999, LSM = 0.988, Δ = 0.011 |
| **Verdict** | **PASS (DIRECTIONAL)** — M2 > LSM, but near-ceiling and no per-seed CI available |

**Probe parameters (frozen addendum):** obs_noise_std = 0.15, cf_temperature = 0.25, obs_channel_amplitude = 0.35, n_repeats = 3. The amplitude (0.35) was selected during integrity testing before publication data. This is a partial pre-registration compliance limitation.

---

## Downgrade Verdict

Per the registered downgrade tree (verdict_contract.md Part 3):

| Condition | Met? |
|---|---|
| battery_win (2 of 3 suites) | YES (topology + counterfactual; social signal lost) |
| tier_a_gate (precedence > 0.70) | YES (0.989) |
| obs1_pass | YES (CV = 2.118) |
| obs3_pass | YES (ρ = 0.878) |

**Registered verdict:** FULL_TIER_A at n=32 registered scale.

**Publicly supported claim (narrower than internal label):** A threshold-gated controller with the registered authored collapse ordering produces seed-stable ordered accessibility collapse under stress at the registered 32-agent scale. This is an architectural-specificity result. Broader claims about typed-family superiority, semantic ontology beyond the threshold scaffold, or LM-deployment transfer are not supported.

---

## Scale Extension (Post-Registration)

| Scale | Obs 3 (ρ) | Obs 5 (r) | Topo suite | Battery |
|---|---|---|---|---|
| n=32 (registered) | 0.880 | 0.905 | ρ = 1.000 | WIN |
| n=64 (extension) | 0.848 | 0.881 | ρ = 1.000 | WIN |
| n=128 (extension) | 0.842 | 0.878 | ρ = 1.000 | WIN |

The regime-1 population-level ρ softens below 0.85 at larger populations. The topology-suite ρ (isolated controller) remains 1.000 at all scales. The 0.85 target was registered for n=32 only.

---

## Summary

| Observable | Target | Result | Verdict |
|---|---|---|---|
| Obs 1 (tactic CV) | > 0.25 | 2.118 | PASS |
| Obs 2b (switch-cost) | ≤ 0.90× | underpowered | UNRESOLVED |
| Obs 3 (collapse ρ) | ≥ 0.85 | 0.878 | **PASS** |
| Obs 4 (social signal) | ≥ +5pp | −36.3pp | **FAIL** |
| Obs 5 (character r) | ≥ 0.70 | 0.897 (aggregate) | FRAGILE PASS |
| Obs 6 (coherence) | M2 > LSM | Δ = 0.011 | WEAK PASS |

**Three findings:**
1. Ordered collapse topology — seed-stable, ablation-confirmed.
2. Internal structure / external legibility trade-off — discovered, not resolved.
3. Evaluation methodology — pre-registration + hostile baseline + transparent audit as replicable template.

---

*This completion report is posted to OSF to close the pre-registration loop. The full paper, source code, raw data, and code diffs are at https://github.com/0ryant/cma-m2-eval.*
