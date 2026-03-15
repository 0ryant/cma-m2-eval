# Experiment Verdict [20260312_024959]


------------------------------------------------------------------
  EXPERIMENT VERDICT
------------------------------------------------------------------
  Metric                                  Value   Status
  ────────────────────────────────────────────────────────────
  Obs 1 — CV(tactic_class)               1.4195   ✓ PASS
  Obs 2a — Switch freq                   0.0500   ✓ PASS
  Obs 2b — Post-switch degradation       0.0011   ✗ FAIL
  Obs 3 — Spearman collapse ρ            0.5677   ✗ FAIL
  Obs 4 — Social lift advantage         -0.3367   ✗ FAIL
  Obs 5 — Pearson r char stability       0.8606   ✓ PASS
  Tier A gate (OVERRIDE+SCORE_WIN)       0.4277   ✗ FAIL
  Battery — Topology win                 0.0000   ✗ FAIL
  Battery — Counterfactual win           1.0000   ✓ PASS
  Battery — Social signal win            0.0000   ✗ FAIL
  FMB Dim 1 — Collapse KL                0.0000   ✗ FAIL
  FMB Dim 2 — Onset delay advantage    1000.0000   ○ NOT TRIGGERED
       → no failure events detected — onset delay cannot be measured
  FMB Dim 3 — Recovery tractability      1.0000   ○ NOT TRIGGERED
       → no failure events detected in this run
  FMB Dim 4 — Contagion individuality    0.0000   ○ NOT TRIGGERED
       → no failure events detected in this run

  Downgrade level:    DOWNGRADE_3_WITH_SIGNAL
  Recommended action: Partial signals detected. Characterise what was found. Return to Phase 1 before attempting Tier A.

  ⚠  Some metrics passed — architecture expresses intended mechanisms. Not yet sufficient for publication claim.
------------------------------------------------------------------

## Observable Snapshots

- Seed 0: CV=1.4042  Spearman=0.5590  PearsonR=0.8456  Gate=PASS
- Seed 1: CV=1.4282  Spearman=0.5836  PearsonR=0.8373  Gate=PASS
- Seed 2: CV=1.4261  Spearman=0.5605  PearsonR=0.8988  Gate=PASS
