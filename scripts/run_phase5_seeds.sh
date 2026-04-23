#!/usr/bin/env bash
# ============================================================
# Phase 5 Pipeline Ablation — Multi-Seed Full Run
# Seeds 123 and 456 (seed 42 already exists)
#
# Prerequisites:
#   1. Smoke test passed: python scripts/_smoke_test_phase5.py --device cuda
#   2. s42 files exist in results/experiments/tmc/
#
# Usage (on aelab-2):
#   cd /path/to/paid_fd
#   bash scripts/run_phase5_seeds.sh cuda:0 2>&1 | tee logs/phase5_multiseed.log
# ============================================================

set -euo pipefail

DEVICE="${1:-cuda:0}"
RUNNER="python scripts/run_tmc_experiment.py"
LOGDIR="results/logs"
mkdir -p "$LOGDIR"

echo "============================================================"
echo "  Phase 5 Multi-Seed Run  (seeds 123, 456)"
echo "  Device : $DEVICE"
echo "  Started: $(date)"
echo "============================================================"

total_runs=6
run_idx=0

for SEED in 123 456; do
    for TASK_ID in 0 1 2; do
        run_idx=$((run_idx + 1))
        LABEL="phase5_s${SEED}_task${TASK_ID}"
        LOG="$LOGDIR/${LABEL}.log"

        # Determine ablation name for display
        case $TASK_ID in
            0) ABLATION="noEMA" ;;
            1) ABLATION="noMixedLoss" ;;
            2) ABLATION="noPersistent" ;;
        esac

        echo ""
        echo "──────────────────────────────────────────────────────────"
        echo "  Run $run_idx/$total_runs : seed=$SEED  ablation=$ABLATION"
        echo "  Log: $LOG"
        echo "──────────────────────────────────────────────────────────"

        $RUNNER \
            --phase 5 \
            --task-id "$TASK_ID" \
            --seed "$SEED" \
            --device "$DEVICE" \
            --rounds 100 \
            2>&1 | tee "$LOG"

        echo "  Done: $(date)"
    done
done

echo ""
echo "============================================================"
echo "  All 6 runs complete.  $(date)"
echo "  Run analysis:"
echo "    python scripts/_analyze_phase5.py"
echo "============================================================"
