# Experiment Verdict [20260312_182146]


------------------------------------------------------------------
  EXPERIMENT VERDICT
------------------------------------------------------------------
  Metric                                  Value   Status
  ────────────────────────────────────────────────────────────
  Obs 1 — CV(tactic_class)               2.1176   ✓ PASS
  Obs 2a — Switch freq                   0.0068   ✗ FAIL
  Obs 2b — Post-switch degradation       0.0331   ✗ FAIL
  Obs 3 — Spearman collapse ρ            0.8801   ✓ PASS
  Obs 4 — Social lift advantage         -0.3633   ✗ FAIL
  Obs 5 — Pearson r char stability       0.9054   ✓ PASS
  Tier A gate (OVERRIDE+SCORE_WIN)       0.9876   ✓ PASS
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

  Downgrade level:    FULL_TIER_A
  Recommended action: Submit to NeurIPS/ICLR/Nature MI. Include downgrade tree in supplementary.
------------------------------------------------------------------

## Observable Snapshots

- Seed 0: CV=2.0729  Spearman=0.8755  PearsonR=0.9890  Gate=PASS
- Seed 1: CV=2.2714  Spearman=0.8736  PearsonR=0.8461  Gate=PASS
- Seed 2: CV=2.0085  Spearman=0.8912  PearsonR=0.8810  Gate=PASS
