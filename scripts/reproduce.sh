#!/bin/bash
# Master reproduction script — runs every experiment behind a paper number.
#
# Total wall time on 2× H100 NVL: ~6–8 hours (sequential). Each individual
# config takes ~20–60 minutes depending on the target model and limit.
#
# All sub-queues are idempotent (skip if summary.json already exists), so this
# script can be re-run safely after partial failures.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Visual Firewall reproduction queue ==="
echo "(each sub-queue is idempotent; skips configs whose summary.json exists)"
echo

# Each queue script logs to logs/queueN.log and respects existing summaries.
bash scripts/run_neurips_queue.sh   # main + α-dial + λ-sweep + adaptive
bash scripts/run_neurips_queue2.sh  # GCG + multi-seed + OCR backend
bash scripts/run_neurips_queue3.sh  # GCG retries
bash scripts/run_neurips_queue4.sh  # λ=0.03 vMF-implied operating point
bash scripts/run_neurips_queue5.sh  # reference-bank size ablation
bash scripts/run_neurips_queue6.sh  # MMBench utility + n=500 + OCR-redact-all
bash scripts/run_neurips_queue6b.sh # MMBench reruns after queue6 (chained)
bash scripts/run_neurips_queue7.sh  # Qwen3.5-9B + Gemma 4 cross-model

echo
echo "=== Aggregating results ==="
python scripts/aggregate_neurips_tables.py > /dev/null

echo
echo "=== Reproduction complete ==="
echo "Aggregated tables in: experiments/results/aggregated/"
echo "Per-experiment summaries: experiments/results/<run>/summary.json"
