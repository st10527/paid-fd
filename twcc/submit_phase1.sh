#!/bin/bash
# ============================================================
# TMC Phase 1: Core Experiments (33 runs)
# ============================================================
# 💰 計費安全：這是 sbatch 批次作業，跑完自動結束，不會持續扣款
#    --time=06:00:00 硬上限 → 超時強制 kill → 不可能忘記關
#
# Exp A:  Privacy-preserving comparison (9)   — Fixed-ε, CSRA
# Exp A': No-privacy reference (3)            — FedAvg, FedMD, FedGMKD
# Exp B:  N sweep / scalability (12)          — N={20,80} × γ={3,10}
# Exp C:  Ablation study (9)                  — BLUE off, full-part, oracle
#
# V100 實測：5 rounds = 13 min → 100 rounds ≈ 3.5 hr (N=50), ~5.5 hr (N=80)
# 硬上限 6 小時，留 buffer
#
# Usage:
#   sbatch twcc/submit_phase1.sh               # 全部 33 個
#   sbatch --array=0-8  twcc/submit_phase1.sh   # 只跑 Exp A
#   sbatch --array=9-11 twcc/submit_phase1.sh   # 只跑 Exp A'
#   sbatch --array=12-23 twcc/submit_phase1.sh  # 只跑 Exp B
#   sbatch --array=24-32 twcc/submit_phase1.sh  # 只跑 Exp C
#
# 跑完後確認安全:
#   bash twcc/check_billing_safe.sh
# ============================================================

#SBATCH --job-name=tmc-phase1
#SBATCH --account=ACD114197             # ← 改成你的計畫 ID
#SBATCH --partition=gp1d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00                  # V100 實測: N=50 ~3.5hr, N=80 ~5.5hr, 硬上限 6hr
#SBATCH --array=0-32
#SBATCH --output=results/logs/phase1_%A_%a.log
#SBATCH --error=results/logs/phase1_%A_%a.err
#SBATCH --mail-type=END,FAIL

set -e

# ---- Environment ----
module purge 2>/dev/null || true
module load miniconda3
source activate paid-fd 2>/dev/null || conda activate paid-fd

cd "$SLURM_SUBMIT_DIR" || cd /work/$USER/paid-fd

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
echo "💰 此 job 已結束，GPU 已自動釋放，不會繼續扣款"
