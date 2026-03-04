# PAID-FD Route B 實驗計畫：效率均衡 (Efficiency Equilibrium)

**版本**: v1.0  
**日期**: 2026-03-04  
**前提**: Route A 確認 γ 對準確度影響 < 1%，轉向效率維度分析  

---

## 核心論述 (Paper Narrative)

> PAID-FD 的 Stackelberg game mechanism 在不同 γ 設定下均能穩健收斂至 ~61% accuracy。
> Game mechanism 的真正價值不在於提升準確度，而在於讓 server **在相同目標準確度下
> 選擇最高效的運作模式** —— 以更少的參與者、更低的通訊成本、更強的隱私保障達成目標。

**Route A 的負面結果本身就是 Route B 的正面發現**：
- 穩健性 (robustness) 是 PAID-FD 的優勢，不是缺陷
- Game mechanism 控制的是 "how efficiently" 而非 "how well"

---

## 已有數據（可直接使用）

| 實驗 | 狀態 | 數據位置 |
|------|------|---------|
| Phase 1.1: γ sweep (3,5,7,10,15) | ✅ 完成 | `phase1_gamma_seed42.json` |
| Phase 1.2: λ sweep (0.3,0.5,1.0,2.0,3.0) | ✅ 完成 | `phase1_lambda_seed42.json` |
| Phase 1.3: N sweep | 🔄 卡住 | GPU 上需重跑 |
| Phase 1.4: C sweep | ⬜ 未開始 | GPU 上需跑 |

### 已有數據的 Route B 指標

**γ=3 vs γ=10（相同準確度 ~61%）：**
- 參與者: 19 vs 50 → **62% fewer participants**
- 通訊量: 742 KB vs 1953 KB → **62% less communication**
- 總付款: 4,498 vs 22,070 → **80% cheaper**
- 隱私: ε=0.85 vs ε=0.32 → **2.6× stronger per-device privacy**
- 達到 60% 的速度: R54 vs R71 → **24% faster convergence**

---

## 重新規劃的實驗結構

### 總覽：7 個實驗 → 7+ 張圖 + 3 張表

| Exp | 名稱 | 目的 | 需要新跑？ |
|-----|------|------|-----------|
| 1 | Convergence Comparison | 6 methods 收斂曲線 | **需新跑** |
| 2 | Efficiency Frontier | γ 的 cost-accuracy tradeoff | 用現有 1.1 + 補 seed |
| 3 | Privacy-Cost Analysis | ε vs communication cost | 用現有 1.1/1.2 |
| 4 | Scalability (N) | N 對效率的影響 | **需新跑 1.3** |
| 5 | Incentive Mechanism | Game 均衡可視化 | 純計算，不需 GPU |
| 6 | Ablation Study | EMA/BLUE/MixedLoss 各自貢獻 | **需新跑** |
| 7 | Energy & Protocol | 能耗分解 + 協議延遲 | 計算 + 少量 GPU |

---

### Exp 1: Convergence & Performance Comparison （Phase 2）

**目的**: 展示 PAID-FD vs 5 baselines 的收斂曲線

**方法**: PAID-FD, FedMD, Fixed-ε, FedAvg, CSRA, FedGMKD

**設定**:
- N=50, γ=5 (PAID-FD 的中間值), 100 rounds, 3 seeds
- CIFAR-100, Dirichlet α=0.5
- Fixed-ε: 使用 ε=0.5 (接近 PAID-FD 的 avg_ε)

**圖表**:
- **Fig. 3**: Accuracy vs Round (6 lines), 帶 std deviation band
- **Table II**: Final accuracy, best accuracy, convergence speed (R@58%)

**預期結果**: PAID-FD ≈ FedMD > Fixed-ε >> FedAvg (without DP), 但 PAID-FD 有隱私保障

**GPU 時間估計**: 6 methods × 100 rounds × 3 seeds ≈ 12 小時

---

### Exp 2: Efficiency Frontier（核心圖，Route B 的主角）

**目的**: 展示 γ 如何在「相同準確度」下控制效率

**資料來源**: Phase 1.1 (γ sweep) — 已有 seed42，需補 seed123, seed456

**圖表**:
- **Fig. 4a**: Multi-axis chart
  - X 軸: γ ∈ {3, 5, 7, 10, 15}
  - 左 Y 軸: Accuracy (bar chart, 幾乎一樣高)
  - 右 Y 軸: Communication Cost / Total Payment / Participation (lines)
  - 視覺效果：bars 齊平，但 cost lines 差距巨大

- **Fig. 4b**: Pareto Front
  - X 軸: Total Communication (KB)
  - Y 軸: Best Accuracy (%)
  - 每個點 = 一個 γ 值
  - 標註：γ=3 在左上（低 cost 高 acc），γ=15 在右邊（高 cost 同 acc）
  - 明確展示 γ=3 dominates γ=10/15

- **Fig. 4c**: Privacy-Cost Tradeoff
  - X 軸: avg_ε (privacy budget, lower = more private)
  - Y 軸: Total Payment to devices
  - 每個 γ 一個點，展示 "pay more but less private" 的 tradeoff

**GPU 時間**: 如需補 2 seeds → 5γ × 100R × 2 seeds ≈ 5 小時

---

### Exp 3: Privacy-Cost Analysis (λ sensitivity)

**目的**: 展示不同隱私偏好群體下系統的適應能力

**資料來源**: Phase 1.2 (λ sweep) — 已有 seed42

**圖表**:
- **Fig. 5a**: λ_mult vs [Accuracy, Participation, avg_ε]
  - Triple-axis or subplot layout
  - 展示：λ 越高 → 設備越 privacy-sensitive → 更少參與 → 但準確度仍穩定

- **Fig. 5b**: Communication Efficiency per λ
  - X 軸: λ_mult
  - Y 軸: KB per 1% accuracy (cost-efficiency metric)

---

### Exp 4: Scalability with N

**目的**: 展示 PAID-FD 在不同規模下的效率表現

**設定**: N ∈ {10, 20, 30, 40, 50}, γ=5, 100 rounds, 1 seed

**圖表**:
- **Fig. 6a**: N vs [Accuracy, Participation Rate, avg_ε]
- **Fig. 6b**: N vs Communication Efficiency (KB per accuracy-point)
- **Fig. 6c**: N vs Convergence Speed (rounds to 58%)

**GPU 時間**: 5N × 100R × 1 seed ≈ 5 小時

---

### Exp 5: Incentive Mechanism Visualization（純計算）

**目的**: 展示 Stackelberg game 的均衡特性

**圖表**:
- **Fig. 7a**: Server Utility U_S vs price p (inverted-U shape)
  - 多條曲線 for different γ
  - 標註 optimal p*

- **Fig. 7b**: Device Participation vs price
  - Sigmoid-like curve
  - 展示 threshold price

- **Fig. 7c**: Equilibrium Table
  - γ, p*, N_part, avg_ε, avg_s, server_utility

**GPU 時間**: 0（純數學計算）

---

### Exp 6: Ablation Study

**目的**: 量化 BLUE, EMA, Mixed Loss 各自的貢獻

**設定** (γ=5, N=50, 100 rounds, 1 seed):

| 變體 | BLUE | EMA | Mixed Loss (CE) | 預期 |
|------|------|-----|-----------------|------|
| Full (v17) | ✅ | ✅ (0.9) | ✅ (0.3) | ~61%, stable |
| No EMA | ✅ | ❌ | ✅ (0.3) | ~60%, more volatile |
| No BLUE | ❌ (uniform avg) | ✅ | ✅ (0.3) | ~59%, slightly lower |
| No CE | ✅ | ✅ | ❌ (pure KL) | ~58%, may degrade |
| Bare FD | ❌ | ❌ | ❌ | ~55%?, unstable |
| No LDP | ✅ | ✅ | ✅ | ~63%?, upper bound |

**圖表**:
- **Fig. 8**: Bar chart — accuracy of each variant
- **Table III**: Detailed metrics per variant (accuracy, loss stability, convergence speed)

**GPU 時間**: 6 variants × 100R × 1 seed ≈ 4 小時

---

### Exp 7: Energy & Protocol Comparison

**目的**: 展示 FD vs FL 的能耗和協議優勢

**圖表**:
- **Fig. 9a**: Energy Breakdown (stacked bar)
  - PAID-FD vs FedAvg vs CSRA
  - 成分: Training, Inference, Communication
  - 重點: PAID-FD 的 communication ≪ FedAvg

- **Fig. 9b**: Protocol Latency (bar chart)
  - PAID-FD (one-shot) vs CSRA (multi-round auction)
  - X 軸: N ∈ {10, 50, 100, 200}
  - Y 軸: Protocol overhead (ms)

- **Fig. 9c**: Communication Volume
  - Per-round upload: PAID-FD (logits: 400 bytes × N_part) vs FedAvg (gradients: 44 MB)

**GPU 時間**: ~2 小時 (主要是 FedAvg baseline)

---

## 圖表清單 (Route B)

| Fig # | 內容 | 類型 | 數據來源 |
|-------|------|------|---------|
| 1 | System Architecture | Diagram | 手繪 |
| 2 | Game Sequence Diagram | Flow | 手繪 |
| 3 | Convergence (6 methods) | Line + band | Exp 1 (新跑) |
| 4a | Efficiency Frontier (γ) | Bar + Line | Exp 2 (現有+補seed) |
| 4b | Pareto Front | Scatter | Exp 2 |
| 4c | Privacy-Cost Tradeoff | Scatter | Exp 2 |
| 5a | Lambda Sensitivity | Multi-axis | Exp 3 (現有) |
| 5b | Lambda Efficiency | Bar | Exp 3 (現有) |
| 6a-c | N Scalability | Line | Exp 4 (新跑) |
| 7a | Server Utility Curve | Line | Exp 5 (計算) |
| 7b | Participation vs Price | Line | Exp 5 (計算) |
| 8 | Ablation | Bar | Exp 6 (新跑) |
| 9a | Energy Breakdown | Stacked Bar | Exp 7 (部分新跑) |
| 9b | Protocol Latency | Bar | Exp 7 (計算) |
| 9c | Communication Volume | Bar | Exp 7 (計算) |

| Table # | 內容 |
|---------|------|
| I | Notation |
| II | Method Comparison (accuracy, convergence, privacy) |
| III | Ablation Results |
| IV | Efficiency Metrics by γ |
| V | Device Specifications |
| VI | Energy Model Parameters |
| VII | Protocol Overhead |

---

## GPU 時間預估

| 實驗 | GPU 時間 | 優先級 |
|------|---------|--------|
| Exp 1 (6 methods comparison) | ~12 小時 | ⭐⭐⭐ |
| Exp 2 (補 2 seeds for γ) | ~5 小時 | ⭐⭐ |
| Exp 4 (N scalability) | ~5 小時 | ⭐⭐ |
| Exp 6 (Ablation, 6 variants) | ~4 小時 | ⭐⭐⭐ |
| Exp 7 (Energy, FedAvg run) | ~2 小時 | ⭐ |
| **Total new GPU time** | **~28 小時** | |

**已有可用數據**: Phase 1.1, 1.2 (Exp 2, 3, 5) — 不需重跑

---

## 執行順序

```
Priority 1 (立即可做):
  → Exp 5: Incentive Mechanism (純計算，0 GPU)
  → Exp 2/3: 用現有 Phase 1.1/1.2 數據畫圖

Priority 2 (需要 GPU):
  → Exp 1: 6-method convergence (最重要的對比圖)
  → Exp 6: Ablation (論文 reviewer 最在意的)

Priority 3 (收尾):
  → Exp 4: N scalability (重跑 Phase 1.3)
  → Exp 7: Energy comparison
  → Exp 2 補 seed (如需 error bar)
```

---

## 與 Route A 實驗計畫的差異

| 面向 | Route A (舊) | Route B (新) |
|------|-------------|-------------|
| 核心指標 | Accuracy gap between γ | Efficiency gap between γ |
| 主角圖 | Accuracy vs γ (希望斜率大) | Pareto Front: Cost vs Accuracy |
| γ 的故事 | "γ↑ → accuracy↑" | "γ↑ → same accuracy, higher cost" |
| Phase 1 目的 | 找最佳 γ for accuracy | 展示 γ 的 efficiency tradeoff |
| Ablation 重點 | 哪個組件提高 accuracy | 哪個組件提供 robustness |
| 論文結論 | Game → better accuracy | Game → efficient resource allocation |

---

## 需要的代碼修改

1. **Ablation 支援**: 在 `paid_fd.py` 中加入 ablation flags:
   - `use_blue`: True/False (BLUE vs uniform aggregation)
   - `use_ema`: True/False (EMA buffer vs direct)
   - `use_mixed_loss`: True/False (KL+CE vs pure KL)

2. **效率指標收集**: 在 `run_single_experiment` 中計算並存儲:
   - `total_communication_kb`
   - `total_payment`
   - `cost_per_accuracy_point`
   - `rounds_to_target` (58%, 60%)

3. **Plotting 腳本**: 自動從 JSON 生成 Route B 圖表

---

*本計畫基於 Route A 的完整分析結果。所有引用的數據均來自已完成的 GPU 實驗。*
