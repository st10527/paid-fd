#!/bin/bash
# Phase 5: Pipeline Internal Ablation (3 runs on aelab-2)
# Expected: ~1hr each on RTX 5070 Ti, total ~3hrs

cd ~/paid_fd
source .venv/bin/activate

echo "========== Phase 5: Pipeline Internal Ablation =========="
echo "Start: $(date)"

for i in 0 1 2; do
    echo ""
    echo "--- Task $i / 2 ---"
    python scripts/run_tmc_experiment.py --phase 5 --task-id $i --device cuda:0
    echo "Finished task $i at $(date)"
done

echo ""
echo "========== Phase 5 Complete: $(date) =========="
echo "Results:"
ls -la results/experiments/tmc/expI_*.json
