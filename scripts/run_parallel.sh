#!/bin/bash
# =============================================================================
# PAID-FD Parallel Experiment Launcher
# =============================================================================
# Runs all 7 phases with 3 seeds in parallel on a single GPU.
#
# Strategy:
#   1. Phase 1.1 runs first (sequential, all seeds) — needed by later phases
#   2. After Phase 1.1, launches Phases 2-7 in 3 parallel streams (one per seed)
#   3. After all done, merges per-seed results into combined files
#
# Usage:
#   chmod +x scripts/run_parallel.sh
#   ./scripts/run_parallel.sh              # defaults: cuda:0, 100 rounds
#   ./scripts/run_parallel.sh cuda:1 50    # custom device & rounds
# =============================================================================

set -e

DEVICE="${1:-cuda:0}"
ROUNDS="${2:-100}"
SEEDS=(42 123 456)
SCRIPT="scripts/run_all_experiments.py"

echo "============================================================"
echo "  PAID-FD Parallel Experiment Runner"
echo "  Device: $DEVICE | Rounds: $ROUNDS | Seeds: ${SEEDS[*]}"
echo "  Started: $(date)"
echo "============================================================"

# ------------------------------------------------------------------
# Step 1: Phase 1.1 (must complete first — all later phases need best_gamma)
# ------------------------------------------------------------------
echo ""
echo ">>> Step 1/3: Running Phase 1.1 (all seeds, sequential)..."
python $SCRIPT --phase 1.1 --seeds ${SEEDS[*]} --device $DEVICE --rounds $ROUNDS
echo ">>> Phase 1.1 done at $(date)"

# ------------------------------------------------------------------
# Step 2: Phases 1.2 + 2-7 — launch 3 parallel streams (one per seed)
# ------------------------------------------------------------------
echo ""
echo ">>> Step 2/3: Launching Phases 1.2-7 in parallel (3 seeds)..."

PIDS=()
LOGS_DIR="results/logs"
mkdir -p "$LOGS_DIR"

for SEED in "${SEEDS[@]}"; do
    LOGFILE="$LOGS_DIR/parallel_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
    echo "  Launching seed=$SEED → $LOGFILE"
    
    python $SCRIPT --all --seed $SEED --device $DEVICE --rounds $ROUNDS \
        --skip-existing \
        > "$LOGFILE" 2>&1 &
    
    PIDS+=($!)
    
    # Stagger launches by 5 seconds to avoid GPU memory contention at startup
    sleep 5
done

echo ""
echo "  PIDs: ${PIDS[*]}"
echo "  Monitor: tail -f $LOGS_DIR/parallel_seed*.log"
echo ""

# Wait for all to finish
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    SEED=${SEEDS[$i]}
    if wait $PID; then
        echo "  ✅ Seed $SEED (PID $PID) completed successfully"
    else
        echo "  ❌ Seed $SEED (PID $PID) failed!"
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "⚠️  $FAILED seed(s) failed. Check logs in $LOGS_DIR/"
    echo "  You can re-run failed seeds individually:"
    echo "  python $SCRIPT --all --seed <SEED> --device $DEVICE --rounds $ROUNDS --skip-existing"
fi

# ------------------------------------------------------------------
# Step 3: Merge per-seed results
# ------------------------------------------------------------------
echo ""
echo ">>> Step 3/3: Merging per-seed results..."
python $SCRIPT --merge --all --seeds ${SEEDS[*]}

echo ""
echo "============================================================"
echo "  All experiments completed at $(date)"
echo "  Results in: results/experiments/"
echo "============================================================"
echo ""
echo "Next: python scripts/plot_all_figures.py"
