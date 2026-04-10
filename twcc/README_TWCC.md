# TWCC 台灣杉二號使用指南 — PAID-FD 實驗
> 給第一次使用 TWCC HPC 的你，step-by-step 從 0 到跑完全部實驗

---

## 📋 你需要準備的東西

| 項目 | 說明 | 在哪裡取得 |
|------|------|-----------|
| **主機帳號** | 登入 TWCC 的帳號 | [TWCC 網站](https://www.twcc.ai/) 註冊時設定 |
| **主機密碼** | 登入密碼 | 同上 |
| **OTP 認證碼** | 雙因素驗證，每次登入都需要 | 手機裝 **IDExpert** App，或選 Email OTP |
| **計畫 ID** | 例如 `GOV112345` | TWCC 會員中心 → 計畫管理 |
| **Git 存取** | 能從 TWCC pull 你的 repo | GitHub SSH key 或 HTTPS |

---

## 🚀 Step 0: 在本機設定計畫 ID（只做一次）

在你的 Mac 終端機執行：

```bash
cd ~/Desktop/tku_research\ paper/2026_TMC/paid_fd

# 把 PROJECT_ID 改成你的計畫 ID，email 填你的
python twcc/configure.py GOV112345 --email your@tku.edu.tw

# commit + push
git add twcc/ scripts/run_v10_1_single.py scripts/aggregate_twcc_results.py
git commit -m "Configure TWCC project ID and scripts"
git push
```

---

## 🔐 Step 1: SSH 登入 TWCC

macOS 直接開 Terminal：

```bash
ssh <你的主機帳號>@twnia2-cli.nchc.org.tw
```

> ⚠️ **登入流程**：
> 1. 輸入主機帳號後按 Enter
> 2. 選擇 Login method：
>    - **1** = Mobile App OTP（開 IDExpert App 看驗證碼）
>    - **2** = Mobile App Push（指紋/臉部辨識）
>    - **3** = Email OTP（驗證碼寄到信箱）
> 3. 輸入 **主機密碼**
> 4. 輸入 **OTP 認證碼**
> 
> ❗ 連續 3 次登入失敗會被鎖 15 分鐘

---

## 📂 Step 2: 了解 TWCC 儲存空間

```
/home/<帳號>/     ← 容量小，放 config、小腳本
/work/<帳號>/     ← 容量大，放程式碼、資料、結果（建議用這裡）
```

```bash
# 檢查你的空間
echo "Home: $(du -sh $HOME 2>/dev/null | cut -f1)"
echo "Work: $(du -sh /work/$USER 2>/dev/null | cut -f1)"

# 我們把專案放在 /work
cd /work/$USER    # 或 cd $WORK（如果環境變數有設定）
```

---

## 📦 Step 3: 下載專案程式碼

```bash
cd /work/$USER

# 方法 A：用 HTTPS（不需要設定 SSH key）
git clone https://github.com/st10527/paid-fd.git

# 方法 B：用 SSH（需要先在 TWCC 上產生 SSH key）
# ssh-keygen -t ed25519 -C "twcc"
# cat ~/.ssh/id_ed25519.pub  # 把這個加到 GitHub Settings > SSH Keys
# git clone git@github.com:st10527/paid-fd.git

cd paid-fd
```

---

## 🐍 Step 4: 建立 Python 環境

```bash
# 載入 TWCC 的 miniconda（不需要 conda init！）
module purge
module load miniconda3

# 建立虛擬環境
conda create -n paid-fd python=3.10 -y

# 啟動
conda activate paid-fd

# 安裝 PyTorch（TWCC 的 V100 用 CUDA 12.1）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安裝其他依賴
pip install numpy scipy scikit-learn pandas pyyaml omegaconf \
            tensorboard matplotlib seaborn tqdm python-dateutil

# 驗證
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

> 💡 **或直接跑自動設定腳本**：
> ```bash
> bash twcc/setup_env.sh
> ```

---

## 🧪 Step 5: 快速測試（重要！不要跳過）

用 `gtest` queue 跑 5 rounds，確認一切能動：

```bash
cd /work/$USER/paid-fd
mkdir -p results/logs results/experiments

# 提交測試 job
sbatch twcc/submit_test.sh
```

會顯示：`Submitted batch job 12345`

```bash
# 查看 job 狀態
squeue -u $USER

# 等它跑完後看 log（通常 5-10 分鐘）
cat results/logs/slurm_test_12345.log

# 確認看到 "ALL TESTS PASSED!" 才繼續！
```

> ⚠️ **如果失敗了**：
> - `ModuleNotFoundError` → 環境沒裝好，重新跑 Step 4
> - `CUDA not available` → 確認有用 `--gres=gpu:1`
> - `FileNotFoundError` → 確認 `cd` 到正確目錄

---

## 🎯 Step 6: 提交正式實驗

### 方法 A：Array Job（推薦 — 平行跑 20 個）

```bash
# 提交所有 20 個 unique configs
sbatch twcc/submit_array.sh

# 查看狀態
squeue -u $USER

# 輸出範例：
#   JOBID  PARTITION  NAME           ST  TIME  NODES
#   12345_0  gp1d     paid-fd-sweep  R   0:05  1       ← 正在跑
#   12345_1  gp1d     paid-fd-sweep  R   0:05  1
#   12345_2  gp1d     paid-fd-sweep  PD  0:00  1       ← 排隊中
#   ...
```

> ⚡ **TWCC 限制**：每個用戶最多 20 個 job 同時跑。
> 20 個 config 剛好 = 可能全部同時跑 = **~4 小時全部完成！**
> （vs lab server 要跑 ~50 小時）

### 方法 B：只跑部分

```bash
# 只跑 3-seed（index 0-11）
sbatch --array=0-11 twcc/submit_array.sh

# 只跑 lambda sweep（index 12-19）
sbatch --array=12-19 twcc/submit_array.sh

# 只跑一個特定的
sbatch twcc/submit_single.sh 5 42 1.0
```

---

## 📊 Step 7: 監控進度

```bash
# 查看所有 job 狀態
squeue -u $USER

# 查看今天跑過的 job（含已完成）
sacct -X --format=JobID,JobName,State,Elapsed,MaxRSS

# 看某個 job 的 log（即時）
tail -f results/logs/slurm_12345_3.log

# 看已完成的結果
ls -la results/experiments/v10_1_g*.json

# 取消某個 job
scancel 12345       # 取消整個 array
scancel 12345_3     # 只取消第 3 個 task
```

---

## 📥 Step 8: 下載結果

全部跑完後：

```bash
# 在 TWCC 上先聚合結果
module purge && module load miniconda3 && conda activate paid-fd
cd /work/$USER/paid-fd

python scripts/aggregate_twcc_results.py
```

然後在你的 **Mac** 上：

```bash
# 方法 A：用 scp 下載
scp -r <帳號>@twnia2-cli.nchc.org.tw:/work/<帳號>/paid-fd/results/experiments/v10_1_*.json \
    ~/Desktop/tku_research\ paper/2026_TMC/paid_fd/results/experiments/

# 方法 B：在 TWCC 上 git push，本機 git pull
# (在 TWCC 上)
cd /work/$USER/paid-fd
git add results/experiments/v10_1_*.json
git commit -m "TWCC results: 3-seed + lambda sweep"
git push

# (在你的 Mac 上)
cd ~/Desktop/tku_research\ paper/2026_TMC/paid_fd
git pull
```

---

## 💰 費用估算

| Queue | 每 GPU·小時 | 1 run (2.5hr) | 20 runs |
|-------|------------|---------------|---------|
| gtest | 免費(測試) | $0 | - |
| gp1d  | ~9 元/GPU·hr | ~22.5 元 | ~450 元 |

> 20 runs × 2.5hr × 1 GPU = **50 GPU·小時 ≈ 450 元**
> 如果平行跑，wall time 只要 2.5-4 小時

---

## ⚡ 速查表

| 動作 | 指令 |
|------|------|
| 登入 | `ssh <帳號>@twnia2-cli.nchc.org.tw` |
| 載入 conda | `module purge && module load miniconda3` |
| 啟動環境 | `conda activate paid-fd` |
| 到專案 | `cd /work/$USER/paid-fd` |
| 測試 | `sbatch twcc/submit_test.sh` |
| 跑全部 | `sbatch twcc/submit_array.sh` |
| 看狀態 | `squeue -u $USER` |
| 看 log | `tail -f results/logs/slurm_XXX.log` |
| 取消 | `scancel <JOB_ID>` |
| 登出 | `exit` |

---

## ❓ 常見問題

### Q: OTP 是什麼？怎麼拿？
A: One-Time Password，雙因素驗證。最簡單的方式是登入時選 **方法 3 (Email OTP)**，驗證碼會寄到你的信箱。

### Q: 計畫 ID 在哪裡找？
A: 登入 [TWCC 會員中心](https://www.twcc.ai/) → 左側「計畫管理」→ 你的計畫 ID（格式如 `GOV112345`）

### Q: conda activate 失敗？
A: 確認有先跑 `module load miniconda3`。TWCC 不需要 `conda init`。

### Q: Job 一直在排隊 (PD)？
A: 正常，等其他人的 GPU 釋放。可以用 `squeue -u $USER` 看 `ST` 欄位，`R`=Running、`PD`=Pending。

### Q: 結果檔案在哪？
A: `results/experiments/v10_1_g<gamma>_s<seed>_lm<lambda>.json`

### Q: 可以跑超過 24 小時嗎？
A: 改用 `gp2d`（48hr）或 `gp4d`（96hr）partition。單個 run 2.5hr 的話 `gp1d` 就夠了。

---

## 📁 檔案結構

```
twcc/
├── README_TWCC.md          ← 你正在看的這份指南
├── configure.py            ← 自動填入計畫 ID 的腳本
├── setup_env.sh            ← 環境安裝腳本
├── submit_test.sh          ← 測試用 SLURM job（gtest, 5 rounds）
├── submit_single.sh        ← 單一實驗 SLURM job
└── submit_array.sh         ← 批次實驗 SLURM array job（20 configs）

scripts/
├── run_v10_1_single.py     ← 單一 (gamma, seed, lambda_mult) 實驗
└── aggregate_twcc_results.py ← 聚合所有結果 + 分析
```
