# Experiment Verdict [20260312_182210]


------------------------------------------------------------------
  EXPERIMENT VERDICT
------------------------------------------------------------------
  Metric                                  Value   Status
  ────────────────────────────────────────────────────────────
  Obs 1 — CV(tactic_class)               2.3106   ✓ PASS
  Obs 2a — Switch freq                   0.0058   ✗ FAIL
  Obs 2b — Post-switch degradation       0.0187   ✗ FAIL
  Obs 3 — Spearman collapse ρ            0.8481   ✗ FAIL
  Obs 4 — Social lift advantage         -0.3633   ✗ FAIL
  Obs 5 — Pearson r char stability       0.8805   ✓ PASS
  Tier A gate (OVERRIDE+SCORE_WIN)       0.9954   ✓ PASS
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

- Seed 0: CV=2.4497  Spearman=0.8264  PearsonR=0.9927  Gate=PASS
- Seed 1: CV=2.3263  Spearman=0.8583  PearsonR=0.6550  Gate=PASS
- Seed 2: CV=2.1558  Spearman=0.8596  PearsonR=0.9940  Gate=PASS
