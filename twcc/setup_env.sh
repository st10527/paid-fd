#!/bin/bash
# ============================================================
# TWCC Environment Setup for PAID-FD
# ============================================================
# 在 TWCC 計算節點上執行，建立 conda 環境
# 
# Usage:
#   bash twcc/setup_env.sh
# ============================================================

set -e

ENV_NAME="paid-fd"
PYTHON_VERSION="3.10"

echo "============================================================"
echo "  PAID-FD Environment Setup for TWCC (台灣杉二號)"
echo "============================================================"

# ---- 1. 載入 module（TWCC 標準做法：用 module load，不要 conda init）----
echo "[1/5] Loading modules..."
module purge 2>/dev/null || true
module load miniconda3
# 注意：TWCC 的 miniconda module 會自動設定環境變數
# 不需要跑 conda init，也不要修改 ~/.bashrc

# ---- 2. 建立 conda 環境 ----
echo "[2/5] Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})..."
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "  Environment already exists, skipping creation."
else
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

# ---- 3. 啟動環境 ----
echo "[3/5] Activating environment..."
source activate ${ENV_NAME} 2>/dev/null || conda activate ${ENV_NAME}

# ---- 4. 安裝 PyTorch（TWCC GPU = NVIDIA，用 CUDA 12.1）----
echo "[4/5] Installing PyTorch with CUDA..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ---- 5. 安裝專案依賴 ----
echo "[5/5] Installing project dependencies..."
pip install numpy scipy scikit-learn pandas pyyaml omegaconf \
            tensorboard matplotlib seaborn tqdm python-dateutil

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "  To activate: conda activate ${ENV_NAME}"
echo "============================================================"

# 驗證
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
"
