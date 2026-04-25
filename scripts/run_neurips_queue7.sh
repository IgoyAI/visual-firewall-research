#!/bin/bash
# Queue 7: cross-model validation on Qwen3.5-9B and Gemma 4 E4B-it.
# Waits for both HF downloads to finish, then runs each model with CMC at the
# headline operating point (α=0.20, λ=0.03) on MM-SafetyBench n=200.

set -u
cd /home/e/e1507650/visual-firewall-research

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

QUEUE_LOG=logs/queue7.log
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

# Wait for both model downloads to complete (no .incomplete files).
wait_for_download() {
    local name="$1"
    local hub_dir="/home/e/e1507650/.cache/huggingface/hub/${name}"
    log "Waiting on download of ${name}"
    until [ -d "$hub_dir" ] && [ -z "$(find "$hub_dir" -name '*.incomplete' -print -quit 2>/dev/null)" ] && [ -d "$hub_dir/snapshots" ]; do
        sleep 30
    done
    # Extra: ensure model.safetensors files exist and have nonzero size in latest snapshot
    local snap=$(ls "$hub_dir/snapshots/" | head -1)
    until ls "$hub_dir/snapshots/$snap"/*.safetensors 2>/dev/null | head -1 > /dev/null && \
          [ -z "$(find "$hub_dir/blobs" -name '*.lock' -print -quit 2>/dev/null)" ]; do
        sleep 30
    done
    log "Download of ${name} appears complete"
}

log "==== queue7 (cross-model validation) start — pid $$ ===="
wait_for_download "models--Qwen--Qwen3.5-9B"
wait_for_download "models--google--gemma-4-E4B-it"

run_cfg mmsafety_qwen35_cmc_full
run_cfg mmsafety_gemma4_cmc_full

log "==== queue7 done ===="
