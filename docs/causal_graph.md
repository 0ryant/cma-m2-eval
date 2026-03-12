# Signed Causal Graph — CMA v4.1 §36.11

**Status:** committed pre-Regime 1  
**Version:** CMA v4.1 / M2 v1.18  
**Amendment:** A.92 (Grok R1)  

This file must be committed to the repository before any Regime run begins.  
It is the authoritative list of signed feedback motifs, their telemetry hooks, and their bounding features.  
Any motif listed as STUB must be promoted to WIRED or explicitly deferred with justification before that Regime run proceeds.

---

## Precedence-Debt Gate

**Publication rule (§M2.7.6):** PRECEDENCE_TAG events must be wired and `(OVERRIDE + SCORE_WIN) / total > 0.70` in Regime 1 data before any Tier A mechanism claim can be made. WEIGHTED_ARB-dominant runs are correlational evidence only.

**Telemetry required before Regime 1:** `PRECEDENCE_TAG` event wired in `telemetry.py` and emitted on every action.

---

## Five Signed Feedback Motifs

### Motif 1 — Fear Amplification: DEFEND Lock-In

| Field | Value |
|---|---|
| **Direction** | Positive (amplifying) |
| **Mechanism** | High rd → DEFEND activation → threat-focused tactic set → high risk actions fail → rd increases → further DEFEND reinforcement |
| **Bounding feature** | DEFEND refractory minimum: 15 ticks (§M2.5.4). `fear_decay_rate` floor: 0.05/tick (§7.5 stability controls) |
| **Telemetry fields** | `feedback_loop_gain_FEAR`, `DEFEND_lock_risk` (per §36.11) |
| **Event emitted** | `FEEDBACK_LOOP_ALERT` with `motif="FEAR_AMPLIFICATION"` when `gain_estimate > 1.0` |
| **Status** | **WIRED** — §M2.7.5, §7.5 stability controls, telemetry.py `FeedbackLoopAlertEvent` |
| **Wire location** | `telemetry.py → FeedbackLoopAlertEvent`, sim engine DEFEND state tracker |

---

### Motif 2 — Shame–Concealment Spiral

| Field | Value |
|---|---|
| **Direction** | Positive (amplifying) |
| **Mechanism** | Failed DECEIVE tactics → shame increase → higher `latent_deceive_prior` activation → more DECEIVE → more failure exposure → shame further increases |
| **Bounding feature** | DECEIVE hard cap: 25 ticks (§M2.5.4c). `shame_ceiling`: 0.85. `latent_deceive_prior_floor`: 0.10 (prevents shame collapsing prior to zero) |
| **Telemetry fields** | `feedback_loop_gain_SHAME` |
| **Event emitted** | `FEEDBACK_LOOP_ALERT` with `motif="SHAME_CONCEALMENT"` |
| **Status** | **WIRED** — §M2.7.5, `yaml_validator.py` Check 5 enforces all bounding values |
| **Wire location** | sim engine shame tracker, DECEIVE persistence counter |

---

### Motif 3 — Trust Collapse → Social Isolation

| Field | Value |
|---|---|
| **Direction** | Positive (amplifying, bounded) |
| **Mechanism** | Trust loss → social network contraction → fewer cooperative opportunities → fewer trust-building interactions → further trust decay |
| **Bounding feature** | Trust delta cap: ±0.15/tick (§36.11). Prevents single-tick collapse. |
| **Telemetry fields** | `feedback_oscillation_detected` |
| **Event emitted** | `FEEDBACK_LOOP_ALERT` with `motif="TRUST_COLLAPSE"` when oscillation detected |
| **Status** | **STUB** — wire before Regime 2 |
| **Deferral justification** | Regime 1 PEACETIME trust network is small (8–16 agents). Trust collapse motif requires multi-agent social network with sufficient density to exhibit the spiral. Regime 2 population density is the correct test bed. |
| **Wire before:** | Regime 2 first run |

---

### Motif 4 — Arousal Suppression Burnout

| Field | Value |
|---|---|
| **Direction** | Positive (amplifying, bounded) |
| **Mechanism** | Low arousal (A < 0.20) → VADCS_amplifier suppresses DOMINATE access → agent cannot discharge energy through DOMINATE → arousal continues to build under suppression → eventual DOMINATE burst when suppression lifts |
| **Bounding feature** | `VADCS_amplifier` threshold: A < 0.20 triggers suppression. DOMINATE accessibility suppressed. `burnout_signature` flag raised when arousal accumulation exceeds phenotype norm. |
| **Telemetry fields** | `burnout_signature` (boolean flag), arousal accumulation counter |
| **Event emitted** | `FEEDBACK_LOOP_ALERT` with `motif="AROUSAL_BURNOUT"` when `burnout_signature=True` |
| **Status** | **WIRED** — §M2.4.7, VADCS module |
| **Wire location** | VADCS module arousal tracker, DOMINATE accessibility gate |

---

### Motif 5 — Narrative Fragmentation → Goal Instability

| Field | Value |
|---|---|
| **Direction** | Positive (amplifying, bounded) |
| **Mechanism** | Narrative coherence loss → goal stack update instability → inconsistent tactic selection → further narrative fragmentation from conflicting outcome signals |
| **Bounding feature** | `narrative_coherence_score` gate ≥ 0.50 on any `goal_stack` update. Updates below threshold are queued, not applied, until coherence recovers. |
| **Telemetry fields** | `narrative_fragmentation_score` |
| **Event emitted** | `DEPRESSIVE_LOCK_IN` (narrative_coherence_ratio < 0.70) — see `telemetry.py` |
| **Status** | **WIRED** — `narrative_gate.py` `GatedGoalStack` blocks goal stack updates when `narrative_coherence_score < 0.50`. `validate_motif5_wired()` confirms gate fires. Integrated: 2026-03-11. |
| **Wire location** | `narrative_gate.py` → `GatedGoalStack.propose_update()` + `tick_step()`. Inject into sim engine via `GatedGoalStack` replacing raw goal list. |
| **Deferral justification** | `narrative_coherence_score` field must be computed by the narrative module before the gate can be applied. The narrative module is a Regime 1 dependency. Wire alongside narrative module initialisation, before first Regime 1 run. |
| **Wire before:** | Regime 1 first run |

---

## Commit Checklist

Before Regime 1:

- [ ] Motif 1 (FEAR): WIRED — confirm `feedback_loop_gain_FEAR` emitted in DEFEND state
- [ ] Motif 2 (SHAME): WIRED — confirm `feedback_loop_gain_SHAME` emitted after failed DECEIVE
- [ ] Motif 3 (TRUST): STUB — confirmed deferred to Regime 2, justification above accepted
- [ ] Motif 4 (AROUSAL): WIRED — confirm `burnout_signature` flag in VADCS module
- [x] Motif 5 (NARRATIVE): **WIRED** — `narrative_gate.py` `GatedGoalStack` active, `validate_motif5_wired()` passes
- [ ] PRECEDENCE_TAG wired in telemetry.py and sim engine loop
- [ ] yaml_validator.py passes all 5 checks on current config
- [ ] causal_graph.md committed to repo at path: `docs/causal_graph.md`

---

## Stability Governance Gate

A Regime 1 run may proceed only when:

1. All WIRED motifs have passing unit tests (telemetry event fires in controlled scenario)
2. Motif 5 (NARRATIVE) is promoted from STUB to WIRED
3. Motif 3 (TRUST) deferral is acknowledged by sign-off on this document
4. `yaml_validator.py --strict` returns exit code 0 on the active config
5. This file is committed at `docs/causal_graph.md` in the same commit as the config

> **Sign-off:** Commit this file with a note: `causal_graph: pre-Regime-1 sign-off [date]`
