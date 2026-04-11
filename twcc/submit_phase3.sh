#!/bin/bash
# ============================================================
# TMC Phase 3: Non-IID α Sweep (12 runs)
# ============================================================
# Exp E: α={0.1, 1.0} × γ={3, 10} × 3 seeds
#        (α=0.5 already done in v10.1)
#
# α=0.1: extreme Non-IID (most challenging)
# α=1.0: mild Non-IID (easier)
#
# Est. time: ~15 min/run × 12 = ~3 hrs
#
# Usage:
#   sbatch twcc/submit_phase3.sh               # 全部 12 個
#   sbatch --array=0-5  twcc/submit_phase3.sh   # 只跑 α=0.1
#   sbatch --array=6-11 twcc/submit_phase3.sh   # 只跑 α=1.0
# ============================================================

#SBATCH --job-name=tmc-phase3
#SBATCH --account=PROJECT_ID             # ← 改成你的計畫 ID
#SBATCH --partition=gp1d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-11
#SBATCH --output=results/logs/phase3_%A_%a.log
#SBATCH --error=results/logs/phase3_%A_%a.err
#SBATCH --mail-type=END,FAIL

set -e

module purge 2>/dev/null || true
module load miniconda3
source activate paid-fd 2>/dev/null || conda activate paid-fd

cd "$SLURM_SUBMIT_DIR" || cd ~/paid_fd

mkdir -p results/experiments/tmc
mkdir -p results/logs

echo "=========================================="
echo "  Phase 3 (α sweep) — Task ${SLURM_ARRAY_TASK_ID}/11"
echo "  Node: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "  Time: $(date)"
echo "=========================================="

python scripts/run_tmc_experiment.py \
    --phase 3 \
    --task-id ${SLURM_ARRAY_TASK_ID} \
    --device cuda:0 \
    --rounds 100

echo "Done: Phase 3 Task ${SLURM_ARRAY_TASK_ID} at $(date)"
