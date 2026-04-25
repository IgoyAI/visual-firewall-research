#!/bin/bash
# Sequential queue of NeurIPS-blocker experiments.
# Idempotent: skips experiments whose summary.json already exists.
# Launches immediately — assumes GPUs are free or will be free.

set -u
cd /home/e/e1507650/visual-firewall-research

QUEUE_LOG=logs/queue.log
mkdir -p logs experiments/results

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_LOG"
}

run_cfg() {
    local name="$1"
    local cfg="configs/${name}.json"
    local out="experiments/results/${name}/summary.json"
    local log="logs/${name}.log"

    if [ ! -f "$cfg" ]; then
        log "SKIP ${name}: config missing"
        return
    fi
    if [ -f "$out" ]; then
        log "SKIP ${name}: already has summary.json"
        return
    fi
    log "START ${name}"
    local t0=$(date +%s)
    python3 scripts/run_real_experiment.py --config "$cfg" > "$log" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ]; then
        log "DONE  ${name}  (${dt}s)"
    else
        log "FAIL  ${name}  (rc=${rc}, ${dt}s) — see ${log}"
    fi
}

run_script() {
    local name="$1"
    shift
    local log="logs/${name}.log"
    log "START ${name}"
    local t0=$(date +%s)
    "$@" > "$log" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ]; then
        log "DONE  ${name}  (${dt}s)"
    else
        log "FAIL  ${name}  (rc=${rc}, ${dt}s) — see ${log}"
    fi
}

log "==== NeurIPS queue start — pid $$ ===="
log "GPUs: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr '\n' ' ')"

# CRITICAL PRIORITY (paper's main results row)
run_cfg mmsafety_llava_cmc_full
run_cfg mmsafety_qwen2vl_cmc_full

# CONFORMAL VERIFICATION (Table 4)
run_script conformal_verification python3 scripts/verify_conformal.py

# PARAPHRASE ATTACK FAMILY (adaptive story — Table 2)
run_cfg mmsafety_llava_paraphrase_cmc
run_cfg mmsafety_llava_paraphrase_semantic

# APPENDIX: Qwen2-VL semantic (for cross-model appendix)
run_cfg mmsafety_qwen2vl_semantic_full

# ABLATIONS: α sweep (appendix)
run_cfg mmsafety_llava_cmc_alpha05
run_cfg mmsafety_llava_cmc_alpha20

# ABLATIONS: λ sweep (appendix)
run_cfg mmsafety_llava_cmc_lambda025
run_cfg mmsafety_llava_cmc_lambda100

# ADAPTIVE WHITE-BOX ATTACKER (NeurIPS-caliber: reviewers' #1 ask)
# Each defense condition evaluated against a per-image attacker that
# selects the evasion text from a pool by minimizing r(s, x).
run_cfg mmsafety_llava_nodef_adaptive
run_cfg mmsafety_llava_keyword_adaptive
run_cfg mmsafety_llava_semantic_adaptive
run_cfg mmsafety_llava_cmc_adaptive

# THEOREM VERIFICATION (empirical support for Thms 1-4 + vMF fit)
run_script theorem_verification python3 scripts/verify_theorems.py

# MULTI-SEED CMC (mean+/-std for main result)
run_cfg mmsafety_llava_cmc_seed0
run_cfg mmsafety_llava_cmc_seed1
run_cfg mmsafety_llava_cmc_seed2

# OCR BACKEND ABLATION (tesseract vs easyocr)
run_cfg mmsafety_llava_cmc_ocr_tesseract

# GCG UNIVERSAL ADVERSARIAL SUFFIX (token-level gradient-based attacker)
# Train once, then evaluate against all three defense conditions. This is
# the strongest adaptive attack in the paper.
if [ ! -f artifacts/gcg_suffix.json ]; then
    run_script train_gcg python3 scripts/train_gcg.py \
        --suffix_len 15 --n_iter 150 --train_images 50 --topk 64 --n_candidates 128
fi
run_cfg mmsafety_llava_nodef_gcg
run_cfg mmsafety_llava_semantic_gcg
run_cfg mmsafety_llava_cmc_gcg

# FINAL AGGREGATION: produce paper-ready tables with bootstrap CIs
run_script aggregate_tables python3 scripts/aggregate_neurips_tables.py

log "==== NeurIPS queue done ===="
log "Tables at experiments/results/aggregated/"
