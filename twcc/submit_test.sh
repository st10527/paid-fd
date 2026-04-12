#!/bin/bash
#SBATCH --job-name=paid-fd-test          # 工作名稱
#SBATCH --account=ACD114197             # ← 改成你的計畫 ID (例: GOV112345)
#SBATCH --partition=gtest                 # 測試用 queue（最長30分鐘，不浪費額度）
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00                   # 只要 30 分鐘（跑 5 rounds 測試）
#SBATCH --output=results/logs/slurm_test_%j.log
#SBATCH --error=results/logs/slurm_test_%j.err

# ============================================================
# TWCC Quick Test — 驗證環境和程式碼是否正常
# ============================================================
# 只跑 5 rounds 確認一切能動，不浪費額度
#
# Usage:
#   sbatch twcc/submit_test.sh
# ============================================================

set -e

echo "============================================================"
echo "  PAID-FD TWCC Environment Test"
echo "  Node: $(hostname)"
echo "  Date: $(date)"
echo "============================================================"

# ---- 環境 ----
module purge 2>/dev/null || true
module load miniconda3 2>/dev/null || module load anaconda3 2>/dev/null || true
source activate paid-fd 2>/dev/null || conda activate paid-fd

# ---- 驗證基本環境 ----
echo ""
echo "[1/4] Python + PyTorch check..."
python -c "
import torch
print(f'  Python: {__import__(\"sys\").version}')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "[2/4] Project imports check..."
cd /work/$USER/paid-fd
python -c "
from scripts.run_all_experiments import run_single_experiment, save_json
from src.data.datasets import load_cifar100_safe_split
from src.methods.paid_fd import PAIDFD
print('  All imports OK!')
"

echo ""
echo "[3/4] Data download check..."
python -c "
from src.data.datasets import load_cifar100_safe_split
train, pub, test = load_cifar100_safe_split(root='./data', n_public=20000, seed=42)
print(f'  Train: {len(train)}, Public: {len(pub)}, Test: {len(test)}')
print('  Data OK!')
"

echo ""
echo "[4/4] Quick 5-round test run..."
mkdir -p results/experiments
python -u scripts/run_v10_1_single.py \
    --gamma 5 \
    --seed 42 \
    --lambda-mult 1.0 \
    --device cuda \
    --rounds 5 \
    --outdir results/experiments/test

echo ""
echo "============================================================"
echo "  ALL TESTS PASSED!"
echo "  Environment is ready for full sweep."
echo "  Next: sbatch twcc/submit_array.sh"
echo "============================================================"
echo "Completed at: $(date)"
