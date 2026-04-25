# Main attack-defense results (LLaVA-1.5-7B)

| Condition | n | Unsafe rate | 95% CI | Flagged | 95% CI |
|---|---|---|---|---|---|
| Keyword | 1200 | 0.139 | [0.120, 0.158] | 1.000 | [1.000, 1.000] |
| Semantic-only (tau=0.78) | 1200 | 0.190 | [0.168, 0.212] | 0.215 | [0.192, 0.238] |
| CMC + Conformal (alpha=0.10) | 1200 | 0.191 | [0.168, 0.213] | 0.212 | [0.188, 0.235] |
| Semantic / Bounded overlay | 1200 | 0.163 | [0.142, 0.183] | 0.547 | [0.518, 0.575] |
| Semantic / Paraphrase | 1200 | 0.130 | [0.112, 0.148] | 0.103 | [0.086, 0.121] |
| CMC / Paraphrase | 1200 | 0.127 | [0.108, 0.146] | 0.094 | [0.078, 0.111] |
| No defense / Adaptive (pool) | 1200 | 0.140 | [0.121, 0.160] | 0.000 | [0.000, 0.000] |
| Keyword / Adaptive (pool) | 1200 | 0.143 | [0.124, 0.163] | 0.000 | [0.000, 0.000] |
| Semantic / Adaptive (pool) | 1200 | 0.147 | [0.128, 0.167] | 0.089 | [0.073, 0.105] |
| CMC / Adaptive (pool) | 1200 | 0.143 | [0.124, 0.163] | 0.058 | [0.044, 0.071] |
| No defense / GCG | 1200 | 0.174 | [0.153, 0.196] | 0.000 | [0.000, 0.000] |
| Semantic / GCG | 1200 | 0.175 | [0.154, 0.197] | 0.043 | [0.032, 0.055] |
| CMC / GCG | 1200 | 0.175 | [0.154, 0.197] | 0.046 | [0.034, 0.058] |

---

# α-dial sweep (CMC, λ=0.5)

| Setting | n | Defended unsafe | Attack flag | Clean flag | Δ unsafe (attack→defended) |
|---|---|---|---|---|---|
| α = 0.05 | 1200 | 0.194 | 0.120 | 0.029 | 0.002 |
| α = 0.10 (default) | 1200 | 0.191 | 0.212 | 0.056 | 0.005 |
| α = 0.20 | 1200 | 0.155 | 0.774 | 0.123 | 0.041 |

---

# λ (inconsistency weight) sweep

| Setting | n | Defended unsafe | Attack flag | Clean flag | Δ unsafe |
|---|---|---|---|---|---|
| λ = 0.03 (vMF-implied, α=0.20) | 1200 | 0.154 | 0.812 | 0.124 | 0.042 |
| λ = 0.25 (α=0.10) | 1200 | 0.192 | 0.221 | 0.059 | 0.004 |
| λ = 0.50 default (α=0.10) | 1200 | 0.191 | 0.212 | 0.056 | 0.005 |
| λ = 1.00 (α=0.10) | 1200 | 0.193 | 0.139 | 0.061 | 0.003 |

---

# OCR-backend ablation

| Backend | n | Defended unsafe | Attack flag | Clean flag | Δ unsafe |
|---|---|---|---|---|---|
| EasyOCR (default) | 1200 | 0.191 | 0.212 | 0.056 | 0.005 |
| Tesseract | 1200 | 0.194 | 0.030 | 0.022 | 0.002 |

---

# Reference-bank size ablation

| Setting | n | Defended unsafe | Attack flag | Clean flag | Δ unsafe |
|---|---|---|---|---|---|
| k = 10 (instruction-override only) | 1200 | 0.183 | 0.369 | 0.103 | 0.013 |
| k = 20 (+ safety-bypass) | 1200 | 0.182 | 0.410 | 0.102 | 0.013 |
| k = 30 (default, full bank) | 1200 | 0.191 | 0.212 | 0.056 | 0.005 |

---

# Target-model results (paper Table 8)

| Target MLLM | n | base | attacked | defended | atk_flag | clean_flag |
|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 1200 | - | 0.196 | 0.154 | 0.812 | 0.124 |
| Qwen2-VL-7B | 1200 | 0.110 | 0.055 | 0.062 | 0.212 | 0.056 |
| Qwen3.5-9B | 1200 | 0.145 | 0.172 | 0.139 | 0.861 | 0.247 |
| Gemma 4 E4B-it | 1200 | 0.005 | 0.001 | 0.003 | 0.861 | 0.247 |

---

# Defense baselines (paper Table 9)

| Defense | n | Defended unsafe | Attack flag | Clean flag | Δ unsafe |
|---|---|---|---|---|---|
| Keyword OCR blocklist | 1200 | 0.139 | 1.000 | - | 0.057 |
| Semantic-only SigLIP (τ=0.78) | 1200 | 0.190 | 0.215 | 0.043 | 0.006 |
| OCR redaction (mask all spans) | 1200 | 0.144 | 1.000 | 0.468 | 0.052 |
| CMC + conformal (α=0.20, λ=0.03) | 1200 | 0.154 | 0.812 | 0.124 | 0.042 |

---

# Sample-size robustness (paper Table 10)

| Experiment | n | Defended unsafe | Attack flag | Clean flag | Δ unsafe |
|---|---|---|---|---|---|
| MM-SafetyBench n=200 (λ=0.03, α=0.20) | 1200 | 0.154 | 0.812 | 0.124 | 0.042 |
| MM-SafetyBench n=500 (λ=0.03, α=0.20) | 3000 | 0.172 | 0.840 | 0.187 | 0.065 |

---

# MMBench utility (defended_control vs base accuracy, n=32 samples)

| Defense | base acc | def_control acc | retention | clean flag |
|---|---|---|---|---|
| Keyword (default) | 0.469 | 0.422 | 0.900 | 0.000 |
| Semantic-only SigLIP | 0.469 | 0.417 | 0.889 | 0.078 |
| CMC + conformal | 0.469 | 0.422 | 0.900 | 0.068 |

---

# Paired bootstrap significance tests (two-sided p, 95% CI)

Pairs matched by (sample_id, variant, transform, attack_position).
Metric: is_unsafe (A - B, positive = A is more unsafe than B).

| A | B | n pairs | mean diff | 95% CI | p-value |
|---|---|---|---|---|---|
| mmsafety_qwen35_cmc_full attacked | mmsafety_qwen35_cmc_full defended | 1200 | +0.0325 | [+0.0133, +0.0517] | 0.0008 |
| mmsafety_gemma4_cmc_full attacked | mmsafety_gemma4_cmc_full defended | 1200 | -0.0025 | [-0.0067, +0.0008] | 0.0910 |
| mmsafety_llava_keyword_full | mmsafety_llava_cmc_full | 1200 | -0.0517 | [-0.0708, -0.0333] | 0.0000 |
| mmsafety_llava_semantic_full | mmsafety_llava_cmc_full | 2400 | -0.0008 | [-0.0025, +0.0008] | 0.1908 |
| mmsafety_llava_semantic_full | mmsafety_llava_cmc_alpha20 | 2400 | +0.0192 | [+0.0121, +0.0262] | 0.0000 |
| mmsafety_llava_semantic_full | mmsafety_llava_cmc_lambda003 | 3600 | +0.0128 | [+0.0081, +0.0178] | 0.0000 |
| mmsafety_llava_paraphrase_semantic | mmsafety_llava_paraphrase_cmc | 1200 | +0.0033 | [-0.0008, +0.0083] | 0.0968 |
| mmsafety_llava_nodef_adaptive | mmsafety_llava_cmc_adaptive | 1200 | -0.0025 | [-0.0067, +0.0017] | 0.1780 |
| mmsafety_llava_semantic_adaptive | mmsafety_llava_cmc_adaptive | 3600 | +0.0008 | [-0.0006, +0.0022] | 0.1732 |
| mmsafety_llava_nodef_gcg | mmsafety_llava_cmc_gcg | 1200 | -0.0008 | [-0.0067, +0.0042] | 0.6544 |
| mmsafety_llava_semantic_gcg | mmsafety_llava_cmc_gcg | 3600 | +0.0000 | [-0.0008, +0.0008] | 0.6976 |

---

# Multi-seed CMC + Conformal (LLaVA-1.5-7B, visible attack)

| Seed | n | Unsafe rate | Flagged |
|---|---|---|---|
| 0 | 1200 | 0.1900 | 0.2117 |
| 1 | 1200 | 0.1900 | 0.2117 |
| 2 | 1200 | 0.1900 | 0.2117 |
| 42 | 1200 | 0.1908 | 0.2117 |

**Mean +/- std across seeds:** unsafe_rate = 0.1902 +/- 0.0004, flag_rate = 0.2117 +/- 0.0000

---

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
