# PAID-FD TMC 論文：專案全貌狀態報告

> 更新日期: 2025-03-29  
> 目標期刊: IEEE Transactions on Mobile Computing (TMC)  
> GPU: NVIDIA RTX 5070 Ti (aelab-2 遠端伺服器)

---

## 一、系統架構概要

### 核心方法: PAID-FD (Privacy-Aware Incentive-Driven Federated Distillation)
- **Stackelberg Game**: Server 設定 price p → Devices 根據 privacy cost 決定參與/出力
- **Federated Distillation**: 傳遞 logits (100 floats, ~400 bytes) 而非 model parameters (11.2M × 4 bytes = ~44.8 MB)
- **LDP (Local Differential Privacy)**: Laplace noise 加在 logits 上，保護隱私
- **BLUE Aggregation**: 按 ε² 加權 (高隱私預算 = 低 noise = 高權重)
- **EMA Buffer**: 指數移動平均穩定 consensus logits

### Pipeline (Fix 17 最終版)
```
Pre-train(10ep, lr=0.1, SGD)
  → Local Train(5ep, lr=0.01, SGD+momentum=0.9)
  → Clip(C=2.0)
  → LDP Laplace(0, 2C/ε)
  → BLUE Aggregation(weights ∝ ε²)
  → EMA(momentum=0.9)
  → Softmax(T=3.0)
  → MixedLoss(0.7×KL + 0.3×CE)
  → Adam(lr=0.001)
```

### 實驗設定
| 參數 | 值 |
|------|-----|
| Model | ResNet-18 (~11.2M params, CIFAR adaptation) |
| Dataset | CIFAR-100 (100 classes) |
| Data Split | 40K private + 10K public + 10K test |
| Partition | Dirichlet α=0.5 (non-IID) |
| Devices | 50 台, 3 types (Jetson Nano 30%, RPi4 40%, RPi3 30%) |
| Privacy Levels | 5 levels (λ: 0.05, 0.2, 0.5, 1.0, 1.5) |
| Rounds | 100 |
| Game | γ=5, δ=0.01, budget=100 |

---

## 二、已完成工作 ✅

### A. 系統開發 (17 次修復迭代)
| Fix | 內容 | 結果 |
|-----|------|------|
| 1-3 | 基礎框架 + 初始訓練 | ~10% (random) |
| 4-6 | SGD, momentum, CIFAR adaptation | ~25% |
| 7-9 | Pre-training 突破 | ~45% |
| 10-12 | Noise handling, clipping | ~50% |
| 13-15 | BLUE aggregation, mixed loss | ~55% |
| 16-17 | EMA buffer, final tuning | **~60-61% stable** |

📄 **文件**: [docs/route_a_report.md](../docs/route_a_report.md)

### B. Route A → Route B 轉向
- **發現**: γ 不影響準確度 (γ=3 到 γ=15 都是 ~60%)
- **轉向**: γ 控制的是**效率** (參與率、成本、隱私預算)，不是準確度
- 📄 **文件**: [docs/route_b_experiment_plan.md](../docs/route_b_experiment_plan.md)

### C. GPU 實驗結果

#### C1. Phase 1.1: γ Sweep (seed 42, 100 rounds)
| γ | Final Accuracy | Participation | avg_ε |
|---|---------------|--------------|-------|
| 3 | 60.3% | Low | Low |
| 5 | 61.1% | Medium | Medium |
| 7 | 60.8% | Medium-High | Medium-High |
| 10 | 60.3% | High | High |
| 15 | 59.9% | Very High | Very High |

📁 `results/experiments/phase1_gamma_seed42.json`

#### C2. Phase 1.2: λ Sweep (seed 42, 100 rounds)
| λ_mult | Final Accuracy |
|--------|---------------|
| 0.3 | 61.2% |
| 0.5 | 61.0% |
| 1.0 | 61.1% |
| 2.0 | 60.0% |
| 3.0 | 59.5% |

📁 `results/experiments/phase1_lambda_seed42.json`

#### C3. Exp 1: 7-Method Comparison (3 seeds × 100 rounds) ⭐
| Method | Accuracy (mean±std) | Conv. Round | 備註 |
|--------|---------------------|-------------|------|
| **PAID-FD** | **60.04 ± 0.67%** | R15 | Game-determined ε≈0.55 |
| Fixed-ε-1.0 | 60.54 ± 0.33% | R15 | 固定 high privacy budget |
| Fixed-ε-0.5 | 60.38 ± 0.21% | R15 | 固定 medium privacy budget |
| FedGMKD | 47.33 ± 0.38% | R68 | Prototype-based, slow convergence |
| FedAvg | 46.07 ± 0.65% | R64 | Parameter averaging baseline |
| FedMD | 34.87 ± 0.95% | — | Accuracy **drifts down** (no EMA/BLUE) |
| CSRA | 1.00 ± 0.00% | — | Parameter-level DP destroys model |

📁 `results/experiments/routeB_exp1_merged_3seeds.json`

**11/11 Route B 目標全部通過** ✅

#### C4. Exp 6: Ablation Study (3 seeds × 100 rounds) ⭐
| Variant | Accuracy | Δ vs Full | 說明 |
|---------|----------|-----------|------|
| Full (PAID-FD) | 60.04% | — | Complete pipeline |
| No-EMA | 60.36% | +0.3% | EMA 穩定性，非準確度 |
| No-BLUE | 60.34% | +0.3% | BLUE 在同質模型下影響小 |
| No-CE | **13.93%** | **-46.1%** | CE anchor 是關鍵組件 |
| Bare-FD | **21.05%** | **-39.0%** | 去掉所有組件 |
| No-LDP | 60.51% | +0.5% | Oracle upper bound (no noise) |

📁 `results/experiments/routeB_exp6_merged_3seeds.json`

**Key Insight**: CE anchor 貢獻最大 (-46.1%)；BLUE/EMA 在同質模型 (全 ResNet-18) 下影響小

### D. 已生成圖表 (19 張 PDF)

| Figure | 內容 | 數據來源 |
|--------|------|---------|
| fig2a | γ convergence curves | Phase 1.1 γ sweep |
| fig2b | γ convergence annotated | Phase 1.1 γ sweep |
| fig3a | 7-method convergence | Exp 1 (3 seeds) |
| fig3b | Final accuracy bar chart | Exp 1 (3 seeds) |
| fig3c | Multi-metric 3-panel | Exp 1 (3 seeds) |
| fig3d | Accuracy-privacy tradeoff | Exp 1 (3 seeds) |
| fig4a | Accuracy vs cost | Phase 1.1 |
| fig4b | Pareto front | Phase 1.1 |
| fig4c | Privacy-cost tradeoff | Phase 1.1 |
| fig4d | Convergence speed | Phase 1.1 |
| fig5a | λ sensitivity | Phase 1.2 |
| fig5b | λ price-privacy | Phase 1.2 |
| fig6a | Ablation convergence | Exp 6 (3 seeds) |
| fig6b | Ablation accuracy bar | Exp 6 (3 seeds) |
| fig6c | Component contribution | Exp 6 (3 seeds) |
| fig7a | Server utility vs price | Game computation |
| fig7b | Participation vs price | Game computation |
| fig7c | Equilibrium comparison | Game computation |
| fig7d | Device decisions | Game computation |

📁 `results/figures/`

---

## 三、未完成工作 ❌ (TMC 差距分析)

### 🔴 P0: Must-Have (審稿人必問)

#### P0-A: 多數據集實驗
- **現狀**: 僅 CIFAR-100
- **需要**: 至少 +1 數據集 (CIFAR-10, FEMNIST, 或 Fashion-MNIST)
- **代碼準備度**: CIFAR-10 wrapper 已存在 (`src/data/datasets.py`)，可直接使用
- **GPU 時間**: ~20h (7 methods × 3 seeds × 100 rounds)
- **重要性**: TMC 幾乎 100% 會要求多數據集驗證

#### P0-B: non-IID 程度敏感度 (α sweep)
- **現狀**: 僅 α=0.5
- **需要**: α ∈ {0.1, 0.3, 0.5, 1.0} (甚至 IID)
- **代碼準備度**: Dirichlet partitioner 已支援任意 α，IID partitioner 也存在
- **GPU 時間**: ~16h (4α × 7 methods × 100 rounds, 可單 seed)
- **重要性**: non-IID 是 FL 論文核心變量

#### P0-C: 設備規模可擴展性 (N sweep)
- **現狀**: 僅 N=50
- **需要**: N ∈ {10, 20, 50, 100}
- **代碼準備度**: `n_devices` 參數可直接調整
- **GPU 時間**: ~20h (4N × 7 methods × 100 rounds)
- **重要性**: 可擴展性是 mobile computing 期刊核心關注

### 🟡 P1: Strongly Recommended (大幅提升論文深度)

#### P1-D: 模型異質性
- **現狀**: 所有 50 台設備都用 ResNet-18 (同質)
- **需要**: 不同設備類型用不同模型 (Jetson Nano→ResNet-18, RPi4→CNN_CIFAR, RPi3→SimpleCNN)
- **代碼準備度**: 
  - 模型已存在: ResNet-18 (~11M), CNN_CIFAR (~1.2M), SimpleCNN (~200K)
  - 需修改: `run_routeB_gpu.py` 中按設備分配模型
  - FD 天然支持異質模型 (只傳 logits)，**但 FedAvg 不行** (需要同質模型)
- **GPU 時間**: ~12h
- **重要性**: FD 相對 FL 的核心優勢之一；目前 ablation 中 BLUE/EMA 影響小正是因為模型同質

#### P1-E: 通信效率量化
- **現狀**: 有 fig4 系列 (Phase 1.1 數據)，但缺少**跨方法**通信量比較
- **需要**: 
  - PAID-FD vs FedAvg 每輪通信量對比 (logits 400B vs parameters 44.8MB)
  - Accuracy-per-KB metric 跨方法比較
- **代碼準備度**: 通信量可計算，不需 GPU
- **GPU 時間**: ~2h (可能需補跑部分 baseline)
- **重要性**: TMC 特別關注 mobile 場景下的通信效率

### 🟢 P2: Nice-to-Have (如有時間)

#### P2-F: Public Data Size 敏感度
- **需要**: public_samples ∈ {5K, 10K, 15K, 20K}
- **GPU 時間**: ~8h

#### P2-G: Budget Constraints
- **需要**: 不同 server budget 下的表現
- **GPU 時間**: ~6h

#### P2-H: Temperature 敏感度
- **需要**: T ∈ {1, 2, 3, 5, 10}
- **GPU 時間**: ~4h

---

## 四、代碼資產清單

### 方法實現 (`src/methods/`)
| 文件 | 行數 | 方法 | 狀態 |
|------|------|------|------|
| paid_fd.py | 609 | PAID-FD (含 ablation flags) | ✅ 完整 |
| fixed_eps.py | 348 | Fixed-ε variant | ✅ 完整 |
| fedavg.py | 199 | FedAvg | ✅ 完整 |
| fedmd.py | 301 | FedMD (oracle, no noise) | ✅ 完整 |
| fedgmkd.py | 540 | FedGMKD (prototype-based) | ✅ 完整 |
| csra.py | 323 | CSRA (parameter-level DP) | ✅ 完整 |
| base.py | — | Base class | ✅ 完整 |

### 模型 (`src/models/`)
| 模型 | 參數量 | 文件 | 使用狀態 |
|------|--------|------|---------|
| ResNet-18 | ~11.2M | resnet.py | ✅ 主力模型 |
| ResNet-34 | ~21.4M | resnet.py | ❌ 未使用 |
| CNN_CIFAR | ~1.2M | cnn.py | ❌ 可用於異質實驗 |
| SimpleCNN | ~200K | cnn.py | ❌ 可用於異質實驗 |
| LeNet5 | ~60K | cnn.py | ❌ 太小，可能不適合 |

### 數據 (`src/data/`)
| 數據集 | Wrapper | 使用狀態 |
|--------|---------|---------|
| CIFAR-100 | ✅ | ✅ 主力數據集 |
| CIFAR-10 | ✅ | ❌ 可直接使用 |
| STL-10 | ✅ (public data) | ❌ 未使用 |

### 腳本 (`scripts/`)
| 腳本 | 用途 | 狀態 |
|------|------|------|
| run_routeB_gpu.py | GPU 跑 Exp1 + Exp6 | ✅ |
| plot_exp1_comparison.py | Fig 3a-d | ✅ |
| plot_exp6_ablation.py | Fig 6a-c | ✅ |
| plot_exp4_convergence.py | Fig 2a-b | ✅ |
| plot_exp23_efficiency.py | Fig 4a-d, 5a-b | ✅ |
| plot_exp5_incentive.py | Fig 7a-d | ✅ |
| analyze_full_results.py | 3-seed merge + analysis | ✅ |

---

## 五、TMC 實驗規劃 (優先排序)

### 執行建議順序

```
Week 1: P0-A 多數據集 (CIFAR-10, ~20h GPU)
  → 修改 runner 支持 CIFAR-10
  → 跑 7 methods × 3 seeds × 100 rounds
  → 生成對比圖

Week 2: P0-B α sweep (~16h GPU) + P1-D 模型異質性 (~12h GPU)
  → α ∈ {0.1, 0.3, 0.5, 1.0}
  → 異質模型分配 (Nano→ResNet-18, RPi4→CNN, RPi3→SimpleCNN)

Week 3: P0-C N sweep (~20h GPU)
  → N ∈ {10, 20, 50, 100}
  → 需修改 device generation 支持不同 N

Week 4: P1-E 通信效率圖 + 收尾
  → 計算通信量、延遲
  → 補充圖表、整理 tables
```

### 總計 GPU 時間估算
| 實驗 | GPU 時間 | 優先級 |
|------|---------|--------|
| P0-A: CIFAR-10 | ~20h | 🔴 |
| P0-B: α sweep | ~16h | 🔴 |
| P0-C: N sweep | ~20h | 🔴 |
| P1-D: 模型異質性 | ~12h | 🟡 |
| P1-E: 通信效率 | ~2h | 🟡 |
| **Total** | **~70h (≈3 天)** | |

---

## 六、論文 Tables/Figures 完成度

### 預計論文圖表 (TMC 等級)

| # | 圖表 | 狀態 | 需要什麼 |
|---|------|------|---------|
| Fig 1 | System Architecture | ❌ | 手繪/TikZ |
| Fig 2 | Game Sequence Diagram | ❌ | 手繪/TikZ |
| Fig 3 | 7-Method Convergence + Bar | ✅ 有 | fig3a-d |
| Fig 4 | γ Efficiency Frontier | ✅ 有 | fig4a-d (from Phase 1.1) |
| Fig 5 | λ Sensitivity | ✅ 有 | fig5a-b (from Phase 1.2) |
| Fig 6 | Ablation Study | ✅ 有 | fig6a-c |
| Fig 7 | Game Mechanism | ✅ 有 | fig7a-d |
| Fig 8 | CIFAR-10 Results | ❌ 缺 | P0-A |
| Fig 9 | α Sensitivity | ❌ 缺 | P0-B |
| Fig 10 | N Scalability | ❌ 缺 | P0-C |
| Fig 11 | Model Heterogeneity | ❌ 缺 | P1-D |
| Fig 12 | Communication Efficiency | ❌ 缺 | P1-E |
| Table I | Notation | ❌ | 寫作時整理 |
| Table II | Method Comparison | ✅ 有數據 | Exp 1 數據完整 |
| Table III | Ablation Results | ✅ 有數據 | Exp 6 數據完整 |
| Table IV | Device Specs | ⚠️ 部分 | 需整理 |
| Table V | Hyperparameters | ❌ | 寫作時整理 |

---

## 七、已知問題與備註

1. **CSRA 1.0%**: 不是 bug，是 parameter-level DP 的必然結果 — 論文中當作 **反面教材** 使用
2. **FedMD accuracy drift**: FedMD 準確度隨 round 下降 (39%→35%)，因為沒有 EMA/BLUE — 凸顯 PAID-FD 設計的必要性
3. **BLUE/EMA ablation 影響小**: 因為模型同質 (全 ResNet-18)。P1-D 模型異質性實驗預期能顯示 BLUE 的真正價值
4. **Fixed-ε 略高於 PAID-FD**: 60.5% vs 60.0%，但 Fixed-ε 沒有隱私保護的理論保證，且成本更高 — 需在論文中論述 accuracy-efficiency tradeoff
5. **γ sweep 只有 1 seed**: Phase 1.1/1.2 僅 seed 42，如需 error bar 要補跑 2 seeds (~5h each)

---

*此文件由 project_status.md 自動生成，最後更新: 2025-03-29*
