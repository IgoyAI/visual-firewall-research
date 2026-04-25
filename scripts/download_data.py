"""Download MM-SafetyBench and produce the shuffled parquet used by the runner."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "data" / "mm_safetybench"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "sd_all.parquet"
    shuffled_path = out_dir / "sd_all_shuffled.parquet"

    if shuffled_path.exists() and raw_path.exists():
        print(f"already present:\n  {raw_path}\n  {shuffled_path}")
        return

    print("Downloading PKU-Alignment/MM-SafetyBench (≈420 MB after parquet conversion)...")
    ds = load_dataset("PKU-Alignment/MM-SafetyBench", "default", split="train")
    df = ds.to_pandas()
    print(f"  loaded {len(df):,} rows")

    df.to_parquet(raw_path, index=False)
    print(f"  wrote {raw_path}")

    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    shuffled.to_parquet(shuffled_path, index=False)
    print(f"  wrote {shuffled_path} (deterministic seed=42 shuffle)")


if __name__ == "__main__":
    sys.exit(main())
