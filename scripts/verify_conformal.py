"""Empirically verify the conformal calibration FPR bound.

Splits MM-SafetyBench into calibration and test sets of clean images,
calibrates the threshold on the calibration set at target FPR alpha,
then measures the empirical FPR on the held-out test set. Compares
against the theoretical bound alpha + 1/(n+1).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defenses.cmc_firewall import CMCFirewall, CMCFirewallConfig
from src.defenses.ocr_firewall import FirewallConfig, run_ocr
from src.defenses.semantic_firewall import _group_words_into_spans, _span_text
from src.eval.benchmark_loader import load_benchmark_manifest, load_image
from src.eval.gpu_encoder import discover_local_siglip_snapshot


def main() -> None:
    alphas = [0.05, 0.10, 0.20]
    n_cal = 100
    n_test = 200

    siglip = discover_local_siglip_snapshot()
    ocr_cfg = FirewallConfig(ocr_backend="easyocr", easyocr_gpu=True, easyocr_cuda_device=0)
    fw = CMCFirewall(CMCFirewallConfig(
        siglip_path=str(siglip), device="cuda:1",
        lambda_inconsistency=0.5,
        ocr_config=ocr_cfg,
    ))

    samples = load_benchmark_manifest(
        Path("data/mm_safetybench/sd_all_shuffled.parquet"), limit=n_cal + n_test,
    )
    cal_samples = samples[:n_cal]
    test_samples = samples[n_cal:n_cal + n_test]

    def max_risk(img) -> float:
        words = run_ocr(img, ocr_cfg)
        spans = _group_words_into_spans(words)
        candidates = []
        for span in spans:
            t = _span_text(span)
            if len(t.split()) >= 2:
                candidates.append(t)
        for w in (2, 3):
            for i in range(len(spans) - w + 1):
                merged = []
                for j in range(w):
                    merged.extend(spans[i + j])
                t = _span_text(merged)
                if len(t.split()) >= 2:
                    candidates.append(t)
        if not candidates:
            return 0.0
        s_inst, s_inc = fw.score_components(img, candidates)
        return max(fw.combined_risk(si, sc) for si, sc in zip(s_inst, s_inc))

    # Compute risks on calibration and test clean images
    print("Computing calibration risks...")
    cal_risks: list[float] = []
    cal_images = []
    for i, s in enumerate(cal_samples):
        try:
            img = load_image(s)
        except Exception:
            continue
        cal_images.append(img)
        cal_risks.append(max_risk(img))
        if (i + 1) % 20 == 0:
            print(f"  cal {i+1}/{len(cal_samples)}")

    print("Computing test risks...")
    test_risks: list[float] = []
    for i, s in enumerate(test_samples):
        try:
            img = load_image(s)
        except Exception:
            continue
        test_risks.append(max_risk(img))
        if (i + 1) % 40 == 0:
            print(f"  test {i+1}/{len(test_samples)}")

    # For each target FPR alpha, calibrate and measure empirical FPR
    import math

    results = []
    cal_risks_sorted = sorted(cal_risks)
    n = len(cal_risks)
    for alpha in alphas:
        k = int(math.ceil((n + 1) * (1.0 - alpha))) - 1
        k = max(0, min(n - 1, k))
        tau = cal_risks_sorted[k]
        empirical_fpr = sum(1 for r in test_risks if r >= tau) / len(test_risks)
        theoretical_bound = alpha + 1.0 / (n + 1)
        results.append({
            "alpha": alpha,
            "tau": round(tau, 4),
            "empirical_fpr": round(empirical_fpr, 4),
            "theoretical_bound": round(theoretical_bound, 4),
            "within_bound": empirical_fpr <= theoretical_bound,
            "n_cal": n,
            "n_test": len(test_risks),
        })

    print()
    print(f'{"alpha":>7s} {"tau":>7s} {"emp_FPR":>9s} {"bound":>8s} {"within":>7s}')
    print("-" * 45)
    for r in results:
        ok = "YES" if r["within_bound"] else "NO"
        print(f'{r["alpha"]:>7.2f} {r["tau"]:>7.4f} {r["empirical_fpr"]:>9.4f} {r["theoretical_bound"]:>8.4f} {ok:>7s}')

    out_dir = Path("experiments/results/conformal_verification")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({
        "cal_risks": cal_risks,
        "test_risks": test_risks,
        "results": results,
    }, indent=2))
    print(f"\nSaved results to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
