#!/bin/bash
# ============================================================
# TMC Phase 4: Reviewer-Defense Experiments (12 runs)
# ============================================================
# 💰 計費安全：sbatch 批次作業，跑完自動結束
#    --time=04:00:00 硬上限 → 超時 kill
#
# Exp F: Fair Fixed-ε (3 runs)     — Same pipeline, fixed epsilon
# Exp G: ε-sweep (6 runs)          — Privacy-utility curve
# Exp H: Hetero-λ BLUE (3 runs)    — BLUE validation
#
# V100 估計：每個 run ~2-3.5 hr (N=50, 100 rounds)
# 12 runs 全部 < 20，不用分批
#
# 提交方式：
#   sbatch twcc/submit_phase4.sh                    # 全部 12 runs
#   sbatch --array=0-2  twcc/submit_phase4.sh       # 只跑 Exp F (Fair Fixed-ε)
#   sbatch --array=3-8  twcc/submit_phase4.sh       # 只跑 Exp G (ε-sweep)
#   sbatch --array=9-11 twcc/submit_phase4.sh       # 只跑 Exp H (Hetero-λ)
#
# 跑完後確認安全:
#   bash twcc/check_billing_safe.sh
# ============================================================

#SBATCH --job-name=tmc-phase4
#SBATCH --account=ACD114197
#SBATCH --partition=gp1d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-11
#SBATCH --output=results/logs/phase4_%A_%a.log
#SBATCH --error=results/logs/phase4_%A_%a.err
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
echo "  Phase 4 — Task ${SLURM_ARRAY_TASK_ID}/11"
echo "  Node: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "  Time: $(date)"
echo "=========================================="

python scripts/run_tmc_experiment.py \
    --phase 4 \
    --task-id ${SLURM_ARRAY_TASK_ID} \
    --device cuda:0 \
    --rounds 100

echo "Done: Phase 4 Task ${SLURM_ARRAY_TASK_ID} at $(date)"
echo "💰 此 job 已結束，GPU 已自動釋放，不會繼續扣款"
