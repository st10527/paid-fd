#!/bin/bash
# =============================================================================
# PAID-FD Phase 1 Complete Runner
# =============================================================================
# Usage (on GPU server):
#   nohup bash scripts/run_phase1_all.sh 2>&1 &
#
# Or with tmux/screen:
#   tmux new -s phase1
#   bash scripts/run_phase1_all.sh
#   (Ctrl+B, D to detach)
#
# Monitor progress:
#   tail -f results/logs/phase1_run.log
#   grep "Phase 1\." results/logs/phase1_run.log | tail -20
# =============================================================================

set -e  # Exit on error

# ── Config ─────────────────────────────────────────────────────────────────
SEED=42
ROUNDS=100
DEVICE="cuda"
LOGDIR="results/logs"
LOGFILE="${LOGDIR}/phase1_run_$(date +%Y%m%d_%H%M%S).log"
SCRIPT="scripts/run_all_experiments.py"

# ── Setup ──────────────────────────────────────────────────────────────────
mkdir -p "$LOGDIR"

echo "============================================================" | tee "$LOGFILE"
echo " PAID-FD Phase 1 Complete Experiment Run"                      | tee -a "$LOGFILE"
echo " Started: $(date)"                                             | tee -a "$LOGFILE"
echo " Seed: $SEED | Rounds: $ROUNDS | Device: $DEVICE"             | tee -a "$LOGFILE"
echo " Log: $LOGFILE"                                                | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"

# ── Helper ─────────────────────────────────────────────────────────────────
run_phase() {
    local phase="$1"
    local start_time=$(date +%s)
    echo "" | tee -a "$LOGFILE"
    echo "──────────────────────────────────────────────────────────" | tee -a "$LOGFILE"
    echo "[$(date '+%H:%M:%S')] Starting Phase $phase ..." | tee -a "$LOGFILE"
    echo "──────────────────────────────────────────────────────────" | tee -a "$LOGFILE"

    python3 "$SCRIPT" \
        --phase "$phase" \
        --seeds "$SEED" \
        --rounds "$ROUNDS" \
        --device "$DEVICE" \
        --skip-existing \
        2>&1 | tee -a "$LOGFILE"

    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    local minutes=$(( elapsed / 60 ))
    local seconds=$(( elapsed % 60 ))

    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✅ Phase $phase completed in ${minutes}m ${seconds}s" | tee -a "$LOGFILE"
    else
        echo "[$(date '+%H:%M:%S')] ❌ Phase $phase FAILED (exit=$exit_code) after ${minutes}m ${seconds}s" | tee -a "$LOGFILE"
        echo "Continuing to next phase..." | tee -a "$LOGFILE"
    fi
    return $exit_code
}

# ── Run Phase 1 ───────────────────────────────────────────────────────────
TOTAL_START=$(date +%s)

# Phase 1.1 must run first (1.2-1.4 depend on its best_gamma)
run_phase "1.1" || true

# Phase 1.2-1.4 are independent of each other, run sequentially on single GPU
run_phase "1.2" || true
run_phase "1.3" || true
run_phase "1.4" || true

# ── Summary ────────────────────────────────────────────────────────────────
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))
TOTAL_HOURS=$(( TOTAL_ELAPSED / 3600 ))
TOTAL_MINS=$(( (TOTAL_ELAPSED % 3600) / 60 ))

echo "" | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"
echo " Phase 1 Complete!"                                            | tee -a "$LOGFILE"
echo " Finished: $(date)"                                            | tee -a "$LOGFILE"
echo " Total time: ${TOTAL_HOURS}h ${TOTAL_MINS}m"                  | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"

# List result files
echo "" | tee -a "$LOGFILE"
echo "Result files:" | tee -a "$LOGFILE"
ls -la results/experiments/phase1_*seed${SEED}*.json 2>/dev/null | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

echo "Done. Check results with:"
echo "  python3 scripts/analyze_fix17.py"
