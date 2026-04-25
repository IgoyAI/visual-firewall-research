#!/bin/bash
# Queue 3: runs the three GCG evaluation configs after queue 2 exits, then
# re-runs the final aggregator to include GCG rows.
# The GCG artifact at artifacts/gcg_suffix.json was trained separately.

set -u
cd /home/e/e1507650/visual-firewall-research

QUEUE_LOG=logs/queue3.log
mkdir -p logs experiments/results

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_LOG"; }

run_cfg() {
    local name="$1"
    local cfg="configs/${name}.json"
    local out="experiments/results/${name}/summary.json"
    local log="logs/${name}.log"
    if [ -f "$out" ]; then log "SKIP ${name}: already done"; return; fi
    log "START ${name}"
    local t0=$(date +%s)
    python3 scripts/run_real_experiment.py --config "$cfg" > "$log" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))
    [ $rc -eq 0 ] && log "DONE  ${name}  (${dt}s)" || log "FAIL  ${name}  (rc=${rc}, ${dt}s)"
}

run_script() {
    local name="$1"; shift
    log "START ${name}"
    local t0=$(date +%s)
    "$@" > "logs/${name}.log" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))
    [ $rc -eq 0 ] && log "DONE  ${name}  (${dt}s)" || log "FAIL  ${name}  (rc=${rc}, ${dt}s)"
}

log "==== queue3 start (pid $$) ===="

# Wait for queue2 to exit.
Q2_PID=2394764
if kill -0 "$Q2_PID" 2>/dev/null; then
    log "Waiting for queue2 (pid $Q2_PID) to exit..."
    while kill -0 "$Q2_PID" 2>/dev/null; do sleep 30; done
    log "queue2 exited."
fi

# Remove stale FAILED records so run_cfg does not skip.
for name in mmsafety_llava_nodef_gcg mmsafety_llava_semantic_gcg mmsafety_llava_cmc_gcg; do
    out="experiments/results/${name}/summary.json"
    if [ ! -f "$out" ] && [ -d "experiments/results/${name}" ]; then
        log "Cleaning stale dir for ${name}"
    fi
done

# Run the three GCG eval configs
run_cfg mmsafety_llava_nodef_gcg
run_cfg mmsafety_llava_semantic_gcg
run_cfg mmsafety_llava_cmc_gcg

# Re-run aggregator so GCG rows appear in ALL_TABLES.md
run_script aggregate_tables_v2 python3 scripts/aggregate_neurips_tables.py

log "==== queue3 done ===="
