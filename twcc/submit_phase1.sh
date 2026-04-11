#!/bin/bash
# ============================================================
# TMC Phase 1: Core Experiments (33 runs)
# ============================================================
# Exp A:  Privacy-preserving comparison (9)   — Fixed-ε, CSRA
# Exp A': No-privacy reference (3)            — FedAvg, FedMD, FedGMKD
# Exp B:  N sweep / scalability (12)          — N={20,80} × γ={3,10}
# Exp C:  Ablation study (9)                  — BLUE off, full-part, oracle
#
# Est. time: ~15 min/run × 33 = ~8.3 hrs (parallelized on TWCC)
# Each job uses 1 GPU, SLURM schedules in parallel
#
# Usage:
#   sbatch twcc/submit_phase1.sh               # 全部 33 個
#   sbatch --array=0-8  twcc/submit_phase1.sh   # 只跑 Exp A
#   sbatch --array=9-11 twcc/submit_phase1.sh   # 只跑 Exp A'
#   sbatch --array=12-23 twcc/submit_phase1.sh  # 只跑 Exp B
#   sbatch --array=24-32 twcc/submit_phase1.sh  # 只跑 Exp C
#
# Monitor:
#   squeue -u $USER
#   sacct -j <JOBID> --format=JobID%20,State,Elapsed,MaxRSS
#   ls results/experiments/tmc/exp*.json | wc -l  # 看完成幾個
# ============================================================

#SBATCH --job-name=tmc-phase1
#SBATCH --account=PROJECT_ID             # ← 改成你的計畫 ID
#SBATCH --partition=gp1d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00                  # 每個 run ~15min, 留 buffer
#SBATCH --array=0-32
#SBATCH --output=results/logs/phase1_%A_%a.log
#SBATCH --error=results/logs/phase1_%A_%a.err
#SBATCH --mail-type=END,FAIL

set -e

# ---- Environment ----
module purge 2>/dev/null || true
module load miniconda3
source activate paid-fd 2>/dev/null || conda activate paid-fd

cd "$SLURM_SUBMIT_DIR" || cd ~/paid_fd

# ---- Create output dirs ----
mkdir -p results/experiments/tmc
mkdir -p results/logs

# ---- Run ----
echo "=========================================="
echo "  Phase 1 — Task ${SLURM_ARRAY_TASK_ID}/32"
echo "  Node: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "  Time: $(date)"
echo "=========================================="

python scripts/run_tmc_experiment.py \
    --phase 1 \
    --task-id ${SLURM_ARRAY_TASK_ID} \
    --device cuda:0 \
    --rounds 100

echo "Done: Phase 1 Task ${SLURM_ARRAY_TASK_ID} at $(date)"
