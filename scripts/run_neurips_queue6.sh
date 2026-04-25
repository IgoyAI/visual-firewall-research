#!/bin/bash
# Queue 6: post-audit experiment top-up.
#  - MMBench utility for semantic + CMC defenses (keyword already done)
#  - Larger MM-SafetyBench split (n=500) at λ=0.03, α=0.20
#  - OCR-redact-all baseline (CMC with τ=-10 → mask every OCR span)

set -u
cd /home/e/e1507650/visual-firewall-research

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

QUEUE_LOG=logs/queue6.log
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

log "==== queue6 (post-audit top-up) start — pid $$ ===="
# Quick MMBench utility (n=32 each, ~5 min)
run_cfg mmbench_subset_semantic
run_cfg mmbench_subset_cmc
# OCR-redact-all baseline (n=200 MM-SafetyBench, ~30 min)
run_cfg mmsafety_llava_ocrall
# Larger MM-SafetyBench at headline operating point (n=500, ~75 min)
run_cfg mmsafety_llava_cmc_n500
log "==== queue6 done ===="
