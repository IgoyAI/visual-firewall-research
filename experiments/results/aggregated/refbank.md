# Reference-bank size ablation

| Setting | n | Defended unsafe | Attack flag | Clean flag | Δ unsafe |
|---|---|---|---|---|---|
| k = 10 (instruction-override only) | 1200 | 0.183 | 0.369 | 0.103 | 0.013 |
| k = 20 (+ safety-bypass) | 1200 | 0.182 | 0.410 | 0.102 | 0.013 |
| k = 30 (default, full bank) | 1200 | 0.191 | 0.212 | 0.056 | 0.005 |
