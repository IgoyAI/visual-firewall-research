#!/bin/bash
# Run remaining experiments sequentially after Qwen2-VL completes.
# Waits for summary.json, then runs each experiment in order.

set -e
cd /home/e/e1507650/visual-firewall-research

echo "Waiting for Qwen2-VL experiment to finish..."
until [ -f experiments/results/mmsafety_qwen2vl_semantic_full/summary.json ]; do
  sleep 60
done
echo "Qwen2-VL done. Starting queue."

echo "=== 1/4: Conformal FPR verification ==="
python3 scripts/verify_conformal.py 2>&1 | tail -30

echo "=== 2/4: CMC + conformal on visible attack ==="
python3 scripts/run_real_experiment.py --config configs/mmsafety_llava_cmc_full.json 2>&1 | tail -20

echo "=== 3/4: Semantic on paraphrase attack ==="
python3 scripts/run_real_experiment.py --config configs/mmsafety_llava_paraphrase_semantic.json 2>&1 | tail -20

echo "=== 4/4: CMC on paraphrase attack ==="
python3 scripts/run_real_experiment.py --config configs/mmsafety_llava_paraphrase_cmc.json 2>&1 | tail -20

echo "All experiments complete."
