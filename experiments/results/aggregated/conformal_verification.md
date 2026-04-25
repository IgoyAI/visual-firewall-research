# Conformal FPR verification

## (1) Transformed-view empirical FPR (paper Table 4)

Calibration on 100 untransformed clean images; evaluation on 1200 clean-control views across {clean, resize, jpeg, crop, rerender, blur}. The bound holds at every α.

| alpha | tau | Transformed-view FPR | Bound α+1/(n+1) | Within? |
|---|---|---|---|---|
| 0.05 | 0.788 | 0.029 | 0.060 | yes |
| 0.1 | 0.755 | 0.056 | 0.110 | yes |
| 0.2 | 0.646 | 0.123 | 0.210 | yes |

## (2) Calibration-distribution single-split (verify_conformal.py)

Single fixed split, n_cal=100 / n_test=200, no transforms. **A single trial can overshoot the bound by sampling noise** — Theorem 4 gives Var(V_n) ≈ α(1-α)/n. The deployment-relevant guarantee is the *expected* FPR over calibration draws, which section (3) verifies.

| alpha | tau | Empirical FPR | Bound α+1/(n+1) | Within? |
|---|---|---|---|---|
| 0.05 | 0.788 | 0.050 | 0.060 | yes |
| 0.1 | 0.755 | 0.120 | 0.110 | no |
| 0.2 | 0.645 | 0.220 | 0.210 | no |

## (3) Bootstrap re-split verification

Mean empirical FPR across B=2000 random cal/test re-splits of the 300 cached risks (n_cal=100, n_test=200).

| alpha | mean FPR | 95% CI | bound α+1/(n+1) | mean ≤ bound? |
|---|---|---|---|---|
| 0.05 | 0.049 | [0.010, 0.105] | 0.060 | yes |
| 0.1 | 0.098 | [0.040, 0.180] | 0.110 | yes |
| 0.2 | 0.198 | [0.115, 0.300] | 0.210 | yes |
