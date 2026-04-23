# PAID-FD 專案狀態

> 📌 **Living Document** — 這是最常更新的文件，記錄當前狀態與下一步  
> 最後更新：2026-04-01

---

## 快速概覽

| 項目 | 狀態 |
|------|------|
| 目標期刊 | IEEE Transactions on Mobile Computing (TMC) |
| 當前版本 | **v9.1**（distillation hyperparams sweep） |
| GPU Server | `aelab-3@aelab-3-MS-7D18`（NVIDIA RTX 5070 Ti） |
| GitHub | `git@github.com:st10527/paid-fd.git` |
| Latest Commit | `430fcf6` (plot_v9_1_figures.py) |
| 🔄 目前在跑 | v9.1 — 8 configs × 50 rounds on GPU |

---

## 當前等待：v9.1 GPU 結果

### 什麼在跑
```bash
nohup python scripts/run_phase0_v9_1_distill_fix.py > results/logs/v9_1_distill_fix.log 2>&1 &
```

### 結果出來後的動作
1. 從 GPU 拉結果 JSON
2. 顯示 summary table
3. 回答 Q1/Q2/Q3（見下方）
4. 執行 `python scripts/plot_v9_1_figures.py` 生成 7 張圖
5. 根據 outcome 決定下一步

### Decision Framework

| Q | 判斷 | 意義 |
|---|------|------|
| **Q1** | D2 (C=5,T=1) final ≥ R1 − 2%? | Distillation 不退化 |
| **Q2** | D8 < D2 < D7, gap > 3%? | γ-accuracy 假設成立 |
| **Q3** | D3 (C=5,T=3) 也不退化? | Soft label 敘事可保留 |

### Outcome 劇本

| Q1 | Q2 | Q3 | 行動 |
|----|----|----|------|
| ✅ | ✅ | ✅ | 🎉 用 C=5, T=3 → 進入 formal experiments |
| ✅ | ✅ | ❌ | 用 C=5, T=1（hard label story）→ formal experiments |
| ✅ | ❌ | — | γ 無差異 → 改走 efficiency story (Route B revisited) |
| ❌ | — | — | Fundamental rethink 🔴 |

---

## 版本歷史摘要

| 版本 | 核心問題 | 發現 |
|------|---------|------|
| v7 | Persistent models 掩蓋 γ | 60% 穩定但 gap < 1.2% |
| v8.0 | Adam state leak + no anchor | 全部崩潰 44%→7% |
| v8.1 | CE anchor masks γ | 穩定 45% 但 γ 無關 |
| v8.2 | Denoising 毀 dark knowledge | D8 smoking gun |
| v8.3 | Old solver → 小 ε → low SNR | γ 全 33-34% |
| v9.0 | C=2,T=3 → teacher near-uniform | Game 修好，distillation 仍退化 |
| **v9.1** | 🔄 Testing C=5,T=1 | 等結果中 |

> 📁 完整版本歷史：[docs/CHANGELOG.md](CHANGELOG.md)

---

## Pipeline（v9.1 版）

```
Pre-train(10ep, lr=0.1, SGD) → ~44% accuracy
  → [每 round]
  → 每個參與 device: fresh copy of server model
  → Local Train(5ep, lr=0.01, SGD+momentum=0.9)
  → Clip(C=5)                          ← v9.1 change
  → LDP Laplace(0, 2C/ε_i)
  → Upload noisy logits
  → Server: BLUE Aggregation(weights ∝ ε_i²)
  → Server: softmax(T=1)               ← v9.1 change
  → Server: KL distillation(fresh SGD, lr=0.001)
```

---

## 核心檔案清單

### 方法實現
| 文件 | 方法 | 行數 |
|------|------|------|
| `src/methods/paid_fd.py` | PAID-FD (v9) | ~407 |
| `src/methods/fixed_eps.py` | Fixed-ε | ~348 |
| `src/methods/fedmd.py` | FedMD (oracle) | ~301 |
| `src/methods/fedavg.py` | FedAvg | ~199 |
| `src/methods/fedgmkd.py` | FedGMKD | ~540 |
| `src/methods/csra.py` | CSRA | ~323 |

### Game
| 文件 | 內容 |
|------|------|
| `src/game/stackelberg.py` | Cubic solver (np.roots), ServerPricing, StackelbergSolver |
| `src/game/utility.py` | QualityFunction: g(ε), q(s,ε) |

### 實驗腳本
| 文件 | 用途 | 狀態 |
|------|------|------|
| `scripts/run_phase0_v9_1_distill_fix.py` | v9.1 C/T/CE sweep | 🔄 Running |
| `scripts/plot_v9_1_figures.py` | 7 figures for v9.1 | ✅ Ready |
| `scripts/run_all_experiments.py` | Unified runner | ✅ |
| `scripts/diagnose_v9_teacher_signal.py` | Monte Carlo signal analysis | ✅ Done |
| `scripts/verify_cubic_roots.py` | Two-root bug proof | ✅ Done |
| `scripts/verify_solver_fix.py` | End-to-end solver verification | ✅ Done |

---

## 文件結構說明

```
docs/
├── PROJECT_STATUS.md    ← 📌 你最常看的（當前狀態 + 下一步）
├── CHANGELOG.md         ← 版本更新時改（v7→v8→...→v9.1）
├── EXPERIMENT_LOG.md    ← 實驗結果出來時改（數據表格）
├── DESIGN_DECISIONS.md  ← 做設計決策時改（theory alignment + open issues）
└── archive/             ← 歷史文件（不需再看，除非要查舊資料）
    ├── 01_theory_code_alignment.md
    ├── 02_v7_to_v8_changelog.md
    ├── 03_experiment_plan.md
    ├── 04_discussion_issues.md
    ├── 05_phase01_diagnostic_report.md
    ├── 06_v82_results_analysis.md
    ├── project_status_v7.md
    ├── project_status.md
    ├── route_a_report.md
    ├── route_b_experiment_plan.md
    └── tmc_experiment_plan_final.md
```

---

## 備忘

- **GPU 指令格式**：`nohup python scripts/xxx.py > results/logs/xxx.log 2>&1 &`
- **遠端伺服器**：aelab-3（ssh 連線，commands 要用 nohup）
- CIFAR-100：100 classes, 30K private + 20K public + 10K test
- 50 devices, Dirichlet α=0.5, 3 device types (Jetson Nano 30%, RPi4 40%, RPi3 30%)
- 5 privacy levels (λ: 0.05, 0.2, 0.5, 1.0, 1.5)
