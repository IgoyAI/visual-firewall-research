"""Bootstrap re-split verification of the conformal FPR bound.

Loads the calibration + test risks already computed by verify_conformal.py,
pools them, and performs B random cal/test re-splits to report the mean
empirical FPR with a 95% percentile CI against the bound α + 1/(n_cal+1).

This corrects the single-split reading of verify_conformal.py: the bound
holds on the *expected* FPR over calibration draws; a single test split
can overshoot by finite-sample noise.

Writes back into experiments/results/conformal_verification/results.json
under key 'trial_results', which the aggregator renders.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results" / "conformal_verification"


def main() -> None:
    path = RESULTS / "results.json"
    data = json.loads(path.read_text())

    pool = np.asarray(list(data["cal_risks"]) + list(data["test_risks"]), dtype=float)
    n_pool = len(pool)
    n_cal = len(data["cal_risks"])          # keep the same 100/200 split sizes
    n_test = len(data["test_risks"])
    assert n_cal + n_test == n_pool

    alphas = [r["alpha"] for r in data["results"]]
    n_bootstrap = 2000
    rng = np.random.default_rng(7)

    # Pre-draw B permutations of the pool. First n_cal indices = calibration,
    # remaining n_test indices = test.
    out_rows = []
    for alpha in alphas:
        fprs = np.empty(n_bootstrap, dtype=float)
        for b in range(n_bootstrap):
            perm = rng.permutation(n_pool)
            cal = pool[perm[:n_cal]]
            test = pool[perm[n_cal:]]
            cal_sorted = np.sort(cal)
            k = int(math.ceil((n_cal + 1) * (1.0 - alpha))) - 1
            k = max(0, min(n_cal - 1, k))
            tau = cal_sorted[k]
            fprs[b] = float((test >= tau).mean())
        mean_fpr = float(fprs.mean())
        lo, hi = (float(x) for x in np.quantile(fprs, [0.025, 0.975]))
        bound = alpha + 1.0 / (n_cal + 1)
        out_rows.append({
            "alpha": alpha,
            "mean_fpr": round(mean_fpr, 4),
            "fpr_lo": round(lo, 4),
            "fpr_hi": round(hi, 4),
            "bound": round(bound, 4),
            "mean_within_bound": mean_fpr <= bound,
            "frac_trials_within_bound": round(float((fprs <= bound).mean()), 4),
        })

    data["trial_results"] = {
        "n_bootstrap": n_bootstrap,
        "n_pool": n_pool,
        "n_cal": n_cal,
        "n_test": n_test,
        "rows": out_rows,
    }

    path.write_text(json.dumps(data, indent=2))

    print(f"Pool={n_pool} (cal={n_cal}, test={n_test}), B={n_bootstrap}\n")
    print(f'{"alpha":>7s} {"meanFPR":>9s} {"95% CI":>18s} {"bound":>8s} {"within":>7s} {"frac<=bnd":>10s}')
    print("-" * 62)
    for r in out_rows:
        ok = "YES" if r["mean_within_bound"] else "NO"
        ci = f"[{r['fpr_lo']:.4f},{r['fpr_hi']:.4f}]"
        print(f'{r["alpha"]:>7.2f} {r["mean_fpr"]:>9.4f} {ci:>18s} {r["bound"]:>8.4f} {ok:>7s} {r["frac_trials_within_bound"]:>10.4f}')
    print(f"\nWrote trial_results to {path}")


if __name__ == "__main__":
    main()
