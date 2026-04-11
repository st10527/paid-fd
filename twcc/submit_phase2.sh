#!/bin/bash
# ============================================================
# TMC Phase 2: CIFAR-10 Cross-Dataset Validation (9 runs)
# ============================================================
# Exp D: Privacy-preserving methods on CIFAR-10
#        PAID-FD + Fixed-ε + CSRA × 3 seeds
#
# Est. time: ~10 min/run × 9 = ~1.5 hrs
# CIFAR-10 (10 classes) is faster than CIFAR-100
#
# Usage:
#   sbatch twcc/submit_phase2.sh               # 全部 9 個
#   sbatch --array=0-2 twcc/submit_phase2.sh   # 只跑 PAID-FD
#   sbatch --array=3-5 twcc/submit_phase2.sh   # 只跑 Fixed-ε
#   sbatch --array=6-8 twcc/submit_phase2.sh   # 只跑 CSRA
# ============================================================

#SBATCH --job-name=tmc-phase2
#SBATCH --account=PROJECT_ID             # ← 改成你的計畫 ID
#SBATCH --partition=gp1d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00                  # CIFAR-10 更快
#SBATCH --array=0-8
#SBATCH --output=results/logs/phase2_%A_%a.log
#SBATCH --error=results/logs/phase2_%A_%a.err
#SBATCH --mail-type=END,FAIL

set -e

module purge 2>/dev/null || true
module load miniconda3
source activate paid-fd 2>/dev/null || conda activate paid-fd

cd "$SLURM_SUBMIT_DIR" || cd ~/paid_fd

mkdir -p results/experiments/tmc
mkdir -p results/logs

echo "=========================================="
echo "  Phase 2 (CIFAR-10) — Task ${SLURM_ARRAY_TASK_ID}/8"
echo "  Node: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "  Time: $(date)"
echo "=========================================="

python scripts/run_tmc_experiment.py \
    --phase 2 \
    --task-id ${SLURM_ARRAY_TASK_ID} \
    --device cuda:0 \
    --rounds 100

echo "Done: Phase 2 Task ${SLURM_ARRAY_TASK_ID} at $(date)"
