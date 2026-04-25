#!/bin/bash
# Queue 5: reference-bank size ablation.
# refbank50 is intentionally skipped — the default bank only has 30 phrases,
# so refbank50 reduces to the full bank (equivalent to mmsafety_llava_cmc_full).
#
# Memory mitigations:
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True reduces fragmentation
#     on the MIG-sliced H100 (each slice = 47.5 GiB, hosts judge + SigLIP).
#   - configs lowered to batch_size=32 (was 64) to leave headroom for KV cache.

set -u
cd /home/e/e1507650/visual-firewall-research

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

QUEUE_LOG=logs/queue5.log
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

log "==== queue5 (refbank ablation, retry with expandable_segments + batch=32) start — pid $$ ===="
run_cfg mmsafety_llava_cmc_refbank10
run_cfg mmsafety_llava_cmc_refbank20
log "==== queue5 done ===="
