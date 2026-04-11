#!/bin/bash
#SBATCH --job-name=paid-fd-sweep         # 工作名稱
#SBATCH --account=ACD114197             # ← 改成你的計畫 ID (例: GOV112345)
#SBATCH --partition=gp1d                  # gtest(30min測試)/gp1d(1天)/gp2d(2天)/gp4d(4天)
#SBATCH --nodes=1                         # 每個 task 1 個節點
#SBATCH --ntasks=1                        # 每個 task 1 個任務
#SBATCH --cpus-per-task=4                 # 4 CPU cores (比例: 1 GPU = 4 CPU = 90GB RAM)
#SBATCH --gres=gpu:1                      # 1 張 V100 GPU per task
#SBATCH --time=04:00:00                   # 每個 task 最大 4 小時
#SBATCH --array=0-19                      # ← 20 個 unique configs
#SBATCH --output=results/logs/slurm_%A_%a.log
#SBATCH --error=results/logs/slurm_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=167320@o365.tku.edu.tw

# ============================================================
# PAID-FD v10.1 Array Job — 3-seed + Lambda Sweep
# ============================================================
# 同時跑所有 20 個 unique (gamma, seed, lambda_mult) 組合
# SLURM 會根據可用 GPU 自動排隊，平行執行
#
# Usage:
#   sbatch twcc/submit_array.sh           # 跑全部 20 個
#   sbatch --array=0-3 twcc/submit_array.sh  # 只跑前 4 個
#
# 查看狀態:
#   squeue -u $USER
#   sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS
# ============================================================

set -e

# ---- 20 個 Unique Configs (gamma seed lambda_mult) ----
# 3-seed: gamma × seed, lambda_mult=1.0
# Lambda: gamma × lambda_mult, seed=42
# 已去除重複的 4 組 (g*_s42_lm1.0)

CONFIGS=(
    # === 3-seed robustness (12 runs, 其中 4 個與 lambda lm1.0 重複) ===
    "3  42  1.0"    # 0  = g3_s42  = lm1.0_g3
    "3  123 1.0"    # 1  = g3_s123
    "3  456 1.0"    # 2  = g3_s456
    "5  42  1.0"    # 3  = g5_s42  = lm1.0_g5
    "5  123 1.0"    # 4  = g5_s123
    "5  456 1.0"    # 5  = g5_s456
    "7  42  1.0"    # 6  = g7_s42  = lm1.0_g7
    "7  123 1.0"    # 7  = g7_s123
    "7  456 1.0"    # 8  = g7_s456
    "10 42  1.0"    # 9  = g10_s42 = lm1.0_g10
    "10 123 1.0"    # 10 = g10_s123
    "10 456 1.0"    # 11 = g10_s456
    # === Lambda sweep (只有 lm=0.5 和 lm=2.0，lm=1.0 已在上面) ===
    "3  42  0.5"    # 12 = lm0.5_g3
    "5  42  0.5"    # 13 = lm0.5_g5
    "7  42  0.5"    # 14 = lm0.5_g7
    "10 42  0.5"    # 15 = lm0.5_g10
    "3  42  2.0"    # 16 = lm2.0_g3
    "5  42  2.0"    # 17 = lm2.0_g5
    "7  42  2.0"    # 18 = lm2.0_g7
    "10 42  2.0"    # 19 = lm2.0_g10
)

# ---- 取得當前 task 的 config ----
IDX=${SLURM_ARRAY_TASK_ID}
CFG=${CONFIGS[$IDX]}
read -r GAMMA SEED LAMBDA_MULT <<< "$CFG"

echo "============================================================"
echo "  Array Task: ${SLURM_ARRAY_TASK_ID} / ${SLURM_ARRAY_TASK_COUNT}"
echo "  Job ID: ${SLURM_JOB_ID}, Array Job: ${SLURM_ARRAY_JOB_ID}"
echo "  Config: gamma=${GAMMA} seed=${SEED} lambda_mult=${LAMBDA_MULT}"
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Date: $(date)"
echo "============================================================"

# ---- 環境 ----
module purge 2>/dev/null || true
module load miniconda3 2>/dev/null || module load anaconda3 2>/dev/null || true
source activate paid-fd 2>/dev/null || conda activate paid-fd

# ---- 切換到專案目錄 ----
cd ${WORK:-$HOME}/paid-fd
mkdir -p results/logs results/experiments

# ---- 執行（run_v10_1_single.py 會自動 skip 已有結果的 config）----
python -u scripts/run_v10_1_single.py \
    --gamma ${GAMMA} \
    --seed ${SEED} \
    --lambda-mult ${LAMBDA_MULT} \
    --device cuda \
    --rounds 100

echo ""
echo "Task ${SLURM_ARRAY_TASK_ID} completed at: $(date)"
