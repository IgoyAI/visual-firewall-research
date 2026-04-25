# Data

This project uses **MM-SafetyBench** as the primary safety benchmark. The
adapted parquet expected by the runner is `data/mm_safetybench/sd_all_shuffled.parquet`
(deterministic seed-42 shuffle of the upstream `sd_all` split).

## Layout

```
data/
├── README.md
└── mm_safetybench/
    ├── sd_all.parquet            # raw upstream order
    └── sd_all_shuffled.parquet   # deterministic seed-42 shuffle (used by configs)
```

## Download

The two parquet files are ~209 MB each and not committed to the repo.
Both are downloadable from the official MM-SafetyBench HuggingFace dataset:

```bash
python scripts/download_data.py
```

Equivalently, by hand:

```python
from datasets import load_dataset
import pandas as pd
from pathlib import Path

ds = load_dataset("PKU-Alignment/MM-SafetyBench", "default", split="train")
df = ds.to_pandas()
out = Path("data/mm_safetybench")
out.mkdir(parents=True, exist_ok=True)
df.to_parquet(out / "sd_all.parquet", index=False)
df.sample(frac=1.0, random_state=42).reset_index(drop=True).to_parquet(
    out / "sd_all_shuffled.parquet", index=False,
)
```

## Schema

Each row carries:

- `sample_id` — stable string identifier
- `prompt` — text prompt fed to the MLLM
- `image_bytes` — PNG bytes of the source image
- `category` — one of 13 MM-SafetyBench harm categories (e.g., `Fraud`,
  `Physical_Harm`, `Illegal_Activitiy`)
- `attack_text` — optional injected instruction (filled from
  `default_attack_text` in the run config when blank)
- `metadata.question_text` — the original benchmark question text

The first 200 rows of `sd_all_shuffled.parquet` constitute the headline
evaluation slice; the remaining rows are the calibration pool.

## License

MM-SafetyBench is distributed by the PKU-Alignment team under their dataset
license. See https://huggingface.co/datasets/PKU-Alignment/MM-SafetyBench
for terms.
