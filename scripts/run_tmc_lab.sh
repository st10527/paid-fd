#!/bin/bash
# ============================================================
# Run TMC experiments on lab GPU (aelab-3 / aelab-2)
# ============================================================
# Sequential execution on a single GPU. Use nohup for stability.
#
# Usage:
#   # Phase 1 (all 33 runs, ~8.5 hrs)
#   nohup bash scripts/run_tmc_lab.sh 1 0 32 cuda:0 > results/logs/phase1_lab.log 2>&1 &
#
#   # Phase 1 partial (just Exp A, task 0-8)
#   nohup bash scripts/run_tmc_lab.sh 1 0 8 cuda:0 > results/logs/phase1_expA.log 2>&1 &
#
#   # Phase 2 on GPU 1
#   nohup bash scripts/run_tmc_lab.sh 2 0 8 cuda:1 > results/logs/phase2_lab.log 2>&1 &
#
# Arguments:
#   $1 = phase (1/2/3)
#   $2 = start task-id (inclusive)
#   $3 = end task-id (inclusive)
#   $4 = device (default: cuda:0)
# ============================================================

set -e

PHASE=${1:-1}
START=${2:-0}
END=${3:-32}
DEVICE=${4:-cuda:0}

echo "=========================================="
echo "  TMC Lab Runner — Phase $PHASE"
echo "  Tasks: $START to $END on $DEVICE"
echo "  Start: $(date)"
echo "=========================================="

mkdir -p results/experiments/tmc
mkdir -p results/logs

FAILED=0
DONE=0
SKIPPED=0

for TASK_ID in $(seq $START $END); do
    echo ""
    echo "-------- Task $TASK_ID / $END --------"
    
    python -u scripts/run_tmc_experiment.py \
        --phase $PHASE \
        --task-id $TASK_ID \
        --device $DEVICE \
        --rounds 100 || {
        echo "[ERROR] Task $TASK_ID failed!"
        FAILED=$((FAILED + 1))
        continue
    }
    
    DONE=$((DONE + 1))
done

echo ""
echo "=========================================="
echo "  Lab Runner Complete"
echo "  Done: $DONE, Failed: $FAILED"
echo "  End: $(date)"
echo "=========================================="
