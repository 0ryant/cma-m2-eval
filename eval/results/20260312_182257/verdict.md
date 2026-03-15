# Experiment Verdict [20260312_182257]


------------------------------------------------------------------
  EXPERIMENT VERDICT
------------------------------------------------------------------
  Metric                                  Value   Status
  ────────────────────────────────────────────────────────────
  Obs 1 — CV(tactic_class)               2.3265   ✓ PASS
  Obs 2a — Switch freq                   0.0044   ✗ FAIL
  Obs 2b — Post-switch degradation       0.0054   ✗ FAIL
  Obs 3 — Spearman collapse ρ            0.8422   ✗ FAIL
  Obs 4 — Social lift advantage         -0.3633   ✗ FAIL
  Obs 5 — Pearson r char stability       0.8782   ✓ PASS
  Tier A gate (OVERRIDE+SCORE_WIN)       0.9958   ✓ PASS
  Battery — Topology win                 1.0000   ✓ PASS
  Battery — Counterfactual win           1.0000   ✓ PASS
  Battery — Social signal win            0.0000   ✗ FAIL
  FMB Dim 1 — Collapse KL                   n/a   ○ NOT TRIGGERED
       → metric not collected in this run
  FMB Dim 2 — Onset delay advantage         n/a   ○ NOT TRIGGERED
       → metric not collected in this run
  FMB Dim 3 — Recovery tractability         n/a   ○ NOT TRIGGERED
       → metric not collected in this run
  FMB Dim 4 — Contagion individuality       n/a   ○ NOT TRIGGERED
       → metric not collected in this run

  Downgrade level:    DOWNGRADE_3_WITH_SIGNAL
  Recommended action: Partial signals detected. Characterise what was found. Return to Phase 1 before attempting Tier A.

  ⚠  Some metrics passed — architecture expresses intended mechanisms. Not yet sufficient for publication claim.
------------------------------------------------------------------

## Observable Snapshots

- Seed 0: CV=2.3609  Spearman=0.8221  PearsonR=0.9880  Gate=PASS
- Seed 1: CV=2.4114  Spearman=0.8443  PearsonR=0.6520  Gate=PASS
- Seed 2: CV=2.2072  Spearman=0.8602  PearsonR=0.9947  Gate=PASS
