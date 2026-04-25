#!/bin/bash
# Continuation queue: runs the NeurIPS-caliber additions that the first
# queue (pid 2371999) could not pick up because the script file was
# unlinked on NFS after bash opened it.
#
# Waits for the first queue's bash wrapper to exit, then runs in order:
# adaptive attacker evaluations, theorem verification, multi-seed CMC,
# OCR-backend ablation, GCG training, GCG evaluations, aggregation.

set -u
cd /home/e/e1507650/visual-firewall-research

QUEUE_LOG=logs/queue2.log
mkdir -p logs experiments/results artifacts

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
    [ $rc -eq 0 ] && log "DONE  ${name}  (${dt}s)" || log "FAIL  ${name}  (rc=${rc}, ${dt}s) — see ${log}"
}

run_script() {
    local name="$1"; shift
    local log="logs/${name}.log"
    log "START ${name}"
    local t0=$(date +%s)
    "$@" > "$log" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))
    [ $rc -eq 0 ] && log "DONE  ${name}  (${dt}s)" || log "FAIL  ${name}  (rc=${rc}, ${dt}s) — see ${log}"
}

log "==== NeurIPS queue2 start — pid $$ ===="

# Wait for first queue to finish (bash wrapper PID 2371999)
FIRST_QUEUE_PID=2371999
if kill -0 "$FIRST_QUEUE_PID" 2>/dev/null; then
    log "Waiting for first queue (pid $FIRST_QUEUE_PID) to exit..."
    # Loop until PID disappears. Check every 30s.
    while kill -0 "$FIRST_QUEUE_PID" 2>/dev/null; do
        sleep 30
    done
    log "First queue exited."
else
    log "First queue already exited."
fi

log "GPUs: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr '\n' ' ')"

# GCG UNIVERSAL SUFFIX TRAINING (once; produces artifacts/gcg_suffix.json)
if [ ! -f artifacts/gcg_suffix.json ]; then
    run_script train_gcg python3 scripts/train_gcg.py \
        --suffix_len 15 --n_iter 150 --train_images 50 --topk 64 --n_candidates 128
else
    log "SKIP train_gcg: artifacts/gcg_suffix.json already exists"
fi

# ADAPTIVE WHITE-BOX ATTACKER (pool-based)
run_cfg mmsafety_llava_nodef_adaptive
run_cfg mmsafety_llava_keyword_adaptive
run_cfg mmsafety_llava_semantic_adaptive
run_cfg mmsafety_llava_cmc_adaptive

# GCG ATTACK (token-level universal suffix)
run_cfg mmsafety_llava_nodef_gcg
run_cfg mmsafety_llava_semantic_gcg
run_cfg mmsafety_llava_cmc_gcg

# THEOREM VERIFICATION (Levy, Lipschitz, DKW, vMF)
run_script theorem_verification python3 scripts/verify_theorems.py

# MULTI-SEED CMC (mean +/- std for main result)
run_cfg mmsafety_llava_cmc_seed0
run_cfg mmsafety_llava_cmc_seed1
run_cfg mmsafety_llava_cmc_seed2

# OCR BACKEND ABLATION
run_cfg mmsafety_llava_cmc_ocr_tesseract

# FINAL AGGREGATION (reads all records.json, produces paper tables)
run_script aggregate_tables python3 scripts/aggregate_neurips_tables.py

log "==== NeurIPS queue2 done ===="
log "Tables at experiments/results/aggregated/"
