#!/usr/bin/env bash
# ============================================================
# Phase 5 Multi-Seed Full Run — nohup-friendly entry point
#
# On aelab-2:
#   git pull
#   chmod +x scripts/run_phase5_nohup.sh
#   bash scripts/run_phase5_nohup.sh [cuda:0]
#
# All output goes to logs/phase5_nohup_<timestamp>.log
# You can safely close the SSH session after launching.
# Monitor progress:
#   tail -f logs/phase5_nohup_*.log
# ============================================================

set -euo pipefail

DEVICE="${1:-cuda:0}"
TIMESTAMP=$(date +%Y%m%d_%H%M)
LOGDIR="results/logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/phase5_nohup_${TIMESTAMP}.log"

echo "============================================================" | tee "$LOGFILE"
echo "  Phase 5 Multi-Seed Full Run" | tee -a "$LOGFILE"
echo "  Device  : $DEVICE" | tee -a "$LOGFILE"
echo "  Log     : $LOGFILE" | tee -a "$LOGFILE"
echo "  Started : $(date)" | tee -a "$LOGFILE"
echo "  PID     : $$" | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"

RUNNER="python scripts/run_tmc_experiment.py"

total_runs=6
run_idx=0

for SEED in 123 456; do
    for TASK_ID in 0 1 2; do
        run_idx=$((run_idx + 1))

        case $TASK_ID in
            0) ABLATION="noEMA" ;;
            1) ABLATION="noMixedLoss" ;;
            2) ABLATION="noPersistent" ;;
        esac

        LABEL="expI_${ABLATION}_s${SEED}"
        OUTFILE="results/experiments/tmc/${LABEL}.json"

        echo "" | tee -a "$LOGFILE"
        echo "──────────────────────────────────────────────────" | tee -a "$LOGFILE"
        echo "  Run $run_idx/$total_runs : seed=$SEED  ablation=$ABLATION" | tee -a "$LOGFILE"
        echo "  Output : $OUTFILE" | tee -a "$LOGFILE"
        echo "  Time   : $(date)" | tee -a "$LOGFILE"
        echo "──────────────────────────────────────────────────" | tee -a "$LOGFILE"

        # Skip if already done
        if [ -f "$OUTFILE" ]; then
            echo "  [SKIP] Already exists." | tee -a "$LOGFILE"
            continue
        fi

        $RUNNER \
            --phase 5 \
            --task-id "$TASK_ID" \
            --seed "$SEED" \
            --device "$DEVICE" \
            --rounds 100 \
            2>&1 | tee -a "$LOGFILE"

        echo "  Finished: $(date)" | tee -a "$LOGFILE"
    done
done

echo "" | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"
echo "  ALL 6 RUNS COMPLETE — $(date)" | tee -a "$LOGFILE"
echo "  Running analysis..." | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"

python scripts/_analyze_phase5.py 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "  Analysis written to results/analysis/" | tee -a "$LOGFILE"
echo "  Done." | tee -a "$LOGFILE"
