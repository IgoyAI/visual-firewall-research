#!/bin/bash
# Queue 4: runs the lambda=0.03 experiment (vMF-implied optimum) after
# queue3 exits. This closes the "vMF says lambda should be ~0.03 but we
# use lambda=0.5" gap the paper identifies.

set -u
cd /home/e/e1507650/visual-firewall-research

QUEUE_LOG=logs/queue4.log
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

log "==== queue4 start (pid $$) ===="

# Wait for queue3 to exit.
Q3_PID=2514125
if kill -0 "$Q3_PID" 2>/dev/null; then
    log "Waiting for queue3 (pid $Q3_PID) to exit..."
    while kill -0 "$Q3_PID" 2>/dev/null; do sleep 30; done
    log "queue3 exited."
fi

# Run lambda=0.03 experiment (vMF-implied optimum) at alpha=0.20
run_cfg mmsafety_llava_cmc_lambda003

# Re-run aggregator one last time to include lambda003 row
run_script aggregate_tables_v3 python3 scripts/aggregate_neurips_tables.py

log "==== queue4 done ===="
