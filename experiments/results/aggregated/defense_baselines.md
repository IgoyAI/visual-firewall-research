# Defense baselines (paper Table 9)

| Defense | n | Defended unsafe | Attack flag | Clean flag | Δ unsafe |
|---|---|---|---|---|---|
| Keyword OCR blocklist | 1200 | 0.139 | 1.000 | - | 0.057 |
| Semantic-only SigLIP (τ=0.78) | 1200 | 0.190 | 0.215 | 0.043 | 0.006 |
| OCR redaction (mask all spans) | 1200 | 0.144 | 1.000 | 0.468 | 0.052 |
| CMC + conformal (α=0.20, λ=0.03) | 1200 | 0.154 | 0.812 | 0.124 | 0.042 |
