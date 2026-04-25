#!/bin/bash
# Queue 6b: rerun the MMBench utility configs that failed in queue6 due to
# missing predict_batch on the SigLIP MCQ backend (now patched in
# src/eval/model_backends.py). Waits until queue6 has logged "queue6 done"
# before starting, so it doesn't fight the LLaVA runs for GPU.

set -u
cd /home/e/e1507650/visual-firewall-research

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

QUEUE_LOG=logs/queue6b.log
mkdir -p logs experiments/results

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_LOG"
}

run_cfg() {
    local name="$1"
    local cfg="configs/${name}.json"
    local out="experiments/results/${name}/summary.json"
    local log="logs/${name}.log"
    if [ ! -f "$cfg" ]; then log "SKIP ${name}: config missing"; return; fi
    if [ -f "$out" ]; then log "SKIP ${name}: already has summary.json"; return; fi
    log "START ${name}"
    local t0=$(date +%s)
    python3 scripts/run_real_experiment.py --config "$cfg" > "$log" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ]; then log "DONE  ${name}  (${dt}s)"
    else log "FAIL  ${name}  (rc=${rc}, ${dt}s) — see ${log}"; fi
}

log "==== queue6b waiting on queue6 ===="
# Wait for queue6 to finish before claiming GPUs
until grep -q "queue6 done" logs/queue6.log 2>/dev/null; do sleep 30; done
log "==== queue6 done detected; queue6b start — pid $$ ===="

# Clean previous failed result dirs (they may contain partial state)
rm -rf experiments/results/mmbench_subset_semantic experiments/results/mmbench_subset_cmc 2>/dev/null

run_cfg mmbench_subset_semantic
run_cfg mmbench_subset_cmc

log "==== queue6b done ===="
