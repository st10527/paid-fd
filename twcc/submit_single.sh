#!/bin/bash
#SBATCH --job-name=paid-fd-single       # 工作名稱
#SBATCH --account=PROJECT_ID             # ← 改成你的計畫 ID (例: GOV112345)
#SBATCH --partition=gp1d                 # gtest(30min測試)/gp1d(1天)/gp2d(2天)/gp4d(4天)
#SBATCH --nodes=1                        # 1 個節點
#SBATCH --ntasks=1                       # 1 個任務
#SBATCH --cpus-per-task=4                # 4 CPU cores (比例: 1 GPU = 4 CPU = 90GB RAM)
#SBATCH --gres=gpu:1                     # 1 張 V100 GPU
#SBATCH --time=04:00:00                  # 最大 4 小時（1 run 約 2.5hr）
#SBATCH --output=results/logs/slurm_%j.log   # stdout log
#SBATCH --error=results/logs/slurm_%j.err    # stderr log
#SBATCH --mail-type=END,FAIL             # 完成或失敗時寄信
# #SBATCH --mail-user=你的email@xxx.com  # ← 取消註解並填你的 email

# ============================================================
# PAID-FD Single Run on TWCC
# ============================================================
# 跑一個 gamma/seed/lambda_mult 組合
#
# Usage:
#   sbatch twcc/submit_single.sh 5 42 1.0
#   sbatch twcc/submit_single.sh 3 123 2.0
# ============================================================

set -e

# ---- 參數 ----
GAMMA=${1:-5}
SEED=${2:-42}
LAMBDA_MULT=${3:-1.0}

echo "============================================================"
echo "  PAID-FD Single Run"
echo "  gamma=${GAMMA} seed=${SEED} lambda_mult=${LAMBDA_MULT}"
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Date: $(date)"
echo "============================================================"

# ---- 環境 ----
module purge 2>/dev/null || true
module load miniconda3 2>/dev/null || module load anaconda3 2>/dev/null || true
source activate paid-fd 2>/dev/null || conda activate paid-fd

# ---- 切換到專案目錄 ----
# TWCC: /home 空間有限，建議用 /work
cd ${WORK:-$HOME}/paid-fd

# ---- 確保資料夾存在 ----
mkdir -p results/logs results/experiments

# ---- 執行 ----
python -u scripts/run_v10_1_single.py \
    --gamma ${GAMMA} \
    --seed ${SEED} \
    --lambda-mult ${LAMBDA_MULT} \
    --device cuda \
    --rounds 100

echo ""
echo "Completed at: $(date)"
