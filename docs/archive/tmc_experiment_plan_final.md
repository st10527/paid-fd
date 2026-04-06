# TMC Performance Evaluation 完整規劃 - 最終版

## 版本資訊
- **版本**: Final v1.0
- **日期**: 2025-02-01
- **方法名稱**: PAID-FD (Privacy-Aware Incentive-Driven Federated Distillation)

---

## 一、核心設計決策

### 1.1 數據集配置

| 數據類型 | 數據集 | 用途 | 說明 |
|---------|--------|------|------|
| **私有數據** | CIFAR-100 | 本地訓練 | Non-IID 分布，100 類 |
| **公共數據** | **STL-10 (unlabeled)** | 知識蒸餾 | Cross-domain，展示泛化能力 |

#### 為什麼選擇 STL-10？

1. **Cross-domain 能力展示**:
   - CIFAR-100: 32×32 自然圖像，100 細粒度類別
   - STL-10: 96×96 自然圖像，10 粗粒度類別 + **100,000 unlabeled images**
   - 使用不同源的公共數據證明 FD 的泛化能力

2. **避免 Data Leakage 嫌疑**:
   - 若公共數據與私有數據同源，審稿人可能質疑隱私保護的意義
   - STL-10 完全獨立，消除這個疑慮

3. **實際場景模擬**:
   - 真實部署中，公共數據通常來自公開數據集（如 ImageNet subset）
   - 與私有數據不同源是常態

4. **STL-10 Unlabeled 特性**:
   - 100,000 unlabeled images（從 ImageNet 抽取）
   - 非常適合作為 distillation 的 transfer set
   - 已被多篇 FD 論文採用

#### 數據處理
```python
# 公共數據處理
# STL-10 原始 96x96，需 resize 到 32x32 以匹配模型輸入
transform_public = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# 從 100,000 unlabeled 中抽取 5,000-10,000 作為公共數據集
public_dataset = STL10(root='./data', split='unlabeled', 
                       transform=transform_public)
public_subset = Subset(public_dataset, indices=range(5000))
```

---

### 1.2 能耗計算模型（完善版）

#### 總能耗分解

$$E_i^{total} = E_i^{train} + E_i^{inf} + E_i^{comm} + E_i^{opt}$$

| 成分 | 符號 | 公式 | 說明 |
|------|------|------|------|
| **訓練能耗** | $E_i^{train}$ | $\kappa_{cpu} f_i^2 \cdot C_{train} \cdot |\mathcal{D}_i|$ | 本地模型訓練 |
| **推論能耗** | $E_i^{inf}$ | $c_i^{inf} \cdot s_i$ | 在公共數據上計算 logits |
| **通訊能耗** | $E_i^{comm}$ | $c_i^{comm} \cdot s_i$ | 上傳 logits |
| **優化能耗** | $E_i^{opt}$ | $O(1)$ | 執行 bisection (可忽略) |

#### 關鍵參數值（基於文獻）

| 參數 | 值 | 來源 |
|------|-----|------|
| $\kappa_{cpu}$ | $10^{-28}$ | DVFS model |
| $C_{train}$ | $10^6$ cycles/sample | CNN training |
| $C_{inf}$ | $10^4$ cycles/sample | CNN inference |
| Logit size | 100 floats = 400 bytes | CIFAR-100 |
| Gradient size | ~11M params × 4 bytes = 44 MB | ResNet-18 |

#### 與 FL 的能耗對比

| 方法 | 通訊內容 | 大小 | 通訊能耗 |
|------|---------|------|---------|
| FL (FedAvg) | Full gradients | 44 MB | $E_{comm}^{FL}$ |
| FL (Top-k) | Sparse gradients | ~4.4 MB | $0.1 \times E_{comm}^{FL}$ |
| **FD (Ours)** | Logits | 400 bytes × $s_i$ | $\ll 0.01 \times E_{comm}^{FL}$ |

#### 能耗優勢論述

即使加上 **Local Inference Energy**，FD 仍遠優於 FL：

$$\frac{E_{FD}^{total}}{E_{FL}^{total}} = \frac{E^{train} + E^{inf} + E^{comm}_{logits}}{E^{train} + E^{comm}_{gradients}}$$

假設 $s_i = 1000$ (上傳 1000 個 logits)：
- $E^{comm}_{logits} = 1000 \times 400 \text{ bytes} = 400 \text{ KB}$
- $E^{comm}_{gradients} = 44 \text{ MB}$
- **通訊能耗比**: $400 \text{ KB} / 44 \text{ MB} \approx 0.9\%$

推論能耗：
- $E^{inf} = c_{inf} \times 1000 = \kappa_{cpu} f^2 C_{inf} \times 1000$
- 對於 RPi 4: $E^{inf} \approx 0.1 \text{ J}$ (1000 samples)
- 對於 FL 傳輸 44 MB @ 1 Mbps: $E^{comm}_{FL} \approx 35 \text{ J}$

**結論**: $E^{inf} + E^{comm}_{logits} \ll E^{comm}_{gradients}$

---

### 1.3 協議效率對比（PAID-FD vs CSRA）

#### Interaction Pattern 對比

| 機制 | 交互模式 | 回合數 | 延遲特性 |
|------|---------|--------|---------|
| **Reverse Auction (CSRA)** | Multi-round bidding | $O(\log N)$ per round | 高延遲 |
| **Stackelberg (PAID-FD)** | **One-shot broadcast** | **1** | **低延遲** |

#### CSRA 的協議流程（每輪）
```
1. Server 廣播預算 B
2. Clients 提交 bid (ε_i, cost_i)     ← 上行通訊
3. Server 收集所有 bids
4. Server 執行 winner determination   ← 計算
5. Server 廣播 winners + payments     ← 下行通訊
6. Winners 訓練並上傳 gradients       ← 上行通訊
```
**Protocol overhead**: 4 次通訊 + auction 計算

#### PAID-FD 的協議流程（每輪）
```
1. Server 廣播 price p               ← 下行通訊 (ONE-SHOT)
2. Clients 本地計算 (s_i*, ε_i*)     ← 無通訊
3. Clients 決定是否參與              ← 無通訊
4. Participants 上傳 logits          ← 上行通訊
```
**Protocol overhead**: 2 次通訊，無額外計算

#### Protocol Latency 量化分析

| 成分 | CSRA | PAID-FD | 差異 |
|------|------|---------|------|
| 下行廣播 | 2 次 | 1 次 | -50% |
| 上行通訊 | 2 次 (bid + gradient) | 1 次 (logits) | -50% |
| Server 計算 | Auction ($O(N \log N)$) | Ternary search ($O(\log \gamma)$) | Lower |
| **總延遲** | **$4T_{comm} + T_{auction}$** | **$2T_{comm}$** | **~50% reduction** |

#### 無線邊緣環境優勢

在無線環境中，協議延遲尤為關鍵：
1. **Channel variability**: 減少通訊次數 = 減少 channel 不確定性影響
2. **Energy for signaling**: 每次通訊都有 protocol overhead (handshake, ACK)
3. **Synchronization**: One-shot 機制不需要等待所有 bids 收齊

```
Wireless Protocol Overhead (典型值):
- TCP handshake: ~50 ms (3-way)
- LTE signaling: ~20-50 ms
- WiFi association: ~10-30 ms

CSRA: 4 × 50 ms = 200 ms overhead
PAID-FD: 2 × 50 ms = 100 ms overhead
Savings: 100 ms per round × 200 rounds = 20 seconds total
```

---

## 二、Baseline 方法（最終版）

| # | 方法 | Venue | 類型 | 選擇理由 |
|---|------|-------|------|----------|
| 1 | **PAID-FD** | TMC (target) | FD + DP + Stackelberg | 我們的方法 |
| 2 | **CSRA** | TIFS 2024 | FL + DP + Reverse Auction | 最直接競品 |
| 3 | **FedGMKD** | NeurIPS 2024 | FD + Prototype | FD SOTA |
| 4 | **FedFed** | NeurIPS 2023 | FD + Feature | FD baseline |
| 5 | **FedMD** | NeurIPS 2019 | 經典 FD | 經典 baseline |
| 6 | **Fixed-ε** | Ablation | FD + Fixed DP | 展示 adaptive 價值 |

---

## 三、系統異質性模型（三維）

### 3.1 異質性維度

| 維度 | 符號 | 範圍 | 分布 |
|------|------|------|------|
| 通訊成本 | $c_i^{comm}$ | 依 channel 計算 | 連續 |
| 隱私敏感度 | $\lambda_i$ | {0.1, 0.5, 1.0} | 離散 (40%/40%/20%) |
| 運算成本 | $c_i^{inf}$ | 依設備類型 | 離散 (3 types) |

### 3.2 設備類型

| Type | 代表硬體 | $f_i$ | $c_i^{inf}$ 係數 | 比例 | Straggler? |
|------|---------|-------|-----------------|------|------------|
| A | Jetson Nano | 1.5 GHz | 0.5× | 30% | No |
| B | RPi 4 (4GB) | 1.2 GHz | 1.0× | 40% | No |
| C | RPi 3 | 0.8 GHz | 2.0× | 30% | **Yes** |

### 3.3 聚合邊際成本

$$c_i = c_i^{comm} + c_i^{inf}$$

這個 $c_i$ 直接進入 device utility function，自然處理異質性。

---

## 四、實驗設計（最終版）

### Exp 1: Algorithm Efficiency
- **目的**: 驗證 $O(N \log^2(1/\delta))$ 複雜度
- **X軸**: $N = \{10, 50, 100, 200, 500, 1000\}$
- **Y軸**: Execution time (ms), log scale
- **比較**: PAID-FD vs Exhaustive search

### Exp 2: Convergence & Accuracy
- **數據集**: Private = CIFAR-100 (Non-IID, Dir 0.5), Public = STL-10 (5000 samples)
- **比較**: 6 methods
- **目標**: 40-45% global accuracy

### Exp 3: Privacy-Accuracy Trade-off
- **ε 範圍**: {0.1, 0.5, 1, 2, 5, 10}
- **重點**: PAID-FD (adaptive) vs CSRA vs Fixed-ε

### Exp 4: Energy Analysis（強化版）
**4A - 總能耗對比**:
- 比較達到相同準確度所需能耗
- PAID-FD vs CSRA vs FedAvg

**4B - 能耗分解（獨立展示 Inference Energy）**:
- Stacked bar chart
- 成分: Training, **Inference**, Communication, Optimization
- 證明: $E^{inf} + E^{comm}_{logits} \ll E^{comm}_{gradients}$

**4C - 通訊量對比**:
- PAID-FD (logits) vs FL (gradients)
- 量化差距: 400 KB vs 44 MB

### Exp 5: Heterogeneity Analysis
**5A - 三維異質性展示**:
- Scatter plot: $(c_i^{comm}, c_i^{inf})$ colored by $\lambda_i$
- 展示 $(s_i^*, \varepsilon_i^*)$ 分布

**5B - Straggler 影響**:
- X軸: Type C 比例 {0%, 10%, 30%, 50%}
- Y軸: Accuracy, Participation rate, Avg round time

**5C - 隱私分布影響**:
- X軸: High-λ 比例 {0%, 20%, 40%, 60%, 80%}
- Y軸: Accuracy, Average ε

### Exp 6: Incentive Mechanism
**6A - Price-Utility Curve**:
- Inverted-U shape of $U^{ES}(p)$

**6B - Participation Dynamics**:
- Participation rate vs price

**6C - Protocol Latency Comparison（新增）**:
- PAID-FD (One-shot) vs CSRA (Auction)
- X軸: Number of clients
- Y軸: Protocol overhead time (ms)

### Exp 7: Scalability
- $N = \{10, 20, 50, 100, 200\}$
- Dual Y-axis: Time & Accuracy

---

## 五、圖表清單（最終版）

| Fig. | 內容 | 類型 | 重點 |
|------|------|------|------|
| 1 | System Architecture | Diagram | 包含 STL-10 public data |
| 2 | Game Sequence | Flow chart | One-shot broadcast |
| **3** | Algorithm Efficiency | Line (log) | $O(N \log^2)$ |
| **4** | Convergence (6 lines) | Line | CIFAR-100 + STL-10 |
| **5** | Privacy-Accuracy | Line | PAID-FD vs CSRA |
| **6a** | Energy Comparison | Bar | Total energy |
| **6b** | Energy Breakdown | Stacked Bar | **Inference 獨立展示** |
| **6c** | Communication Volume | Bar | Logits vs Gradients |
| **7a** | 3D Heterogeneity | Scatter | Device distribution |
| **7b** | Straggler Impact | Line | Type C ratio |
| **7c** | Privacy Distribution | Line | λ ratio |
| **8a** | Price-Utility | Line | Inverted-U |
| **8b** | Participation | Line | Rate vs price |
| **8c** | Protocol Latency | Bar | **One-shot vs Auction** |
| **9** | Scalability | Dual-axis | N scaling |

| Table | 內容 |
|-------|------|
| I | Primary Notation |
| II | Complexity Comparison |
| III | Dataset Configuration (CIFAR-100 + STL-10) |
| IV | Device Specifications (3 types) |
| V | Energy Model Parameters |
| VI | **Protocol Overhead Comparison** |
| VII | Final Performance Summary |

---

## 六、結果儲存機制

### 目錄結構
```
results/
├── experiments/
│   ├── exp1_efficiency/
│   ├── exp2_convergence/
│   ├── exp3_privacy/
│   ├── exp4_energy/
│   ├── exp5_heterogeneity/
│   ├── exp6_incentive/
│   └── exp7_scalability/
├── checkpoints/
├── logs/
└── figures/
```

### 檔案命名
```
{method}_{dataset}_{config}_{timestamp}.json
例: PAID-FD_cifar100_dir0.5_eps1.0_2025-02-01_14-30-00.json
```

### 重跑機制
```bash
# 只重跑 CSRA 的 exp2
python run_experiment.py --exp exp2 --methods CSRA --force_rerun

# 重跑所有方法的 exp4
python run_experiment.py --exp exp4 --methods all --force_rerun
```

---

## 七、關鍵差異化總結

### PAID-FD vs CSRA (TIFS 2024)

| 維度 | CSRA | **PAID-FD** | 優勢 |
|------|------|-------------|------|
| 通訊內容 | Gradients (44 MB) | **Logits (400 KB)** | **110× smaller** |
| 協議模式 | Multi-round auction | **One-shot broadcast** | **50% less latency** |
| 決策變數 | Binary | **Dual (s, ε)** | 更細粒度控制 |
| 隱私機制 | DP on gradients | **LDP on logits** | 更強保護 |
| 異質性 | 2D | **3D** | 更全面 |
| Public data | 不需要 | **STL-10 (cross-domain)** | 展示泛化能力 |

---

## 八、論文章節對應

| 實驗 | 對應論文章節 | 主要貢獻點 |
|------|-------------|-----------|
| Exp 1 | V-A | 演算法效率 |
| Exp 2 | V-B | 收斂性與準確度 |
| Exp 3 | V-C | 隱私-準確度權衡 |
| Exp 4 | V-D | **能耗分析（含 inference）** |
| Exp 5 | V-E | 三維異質性 |
| Exp 6 | V-F | **激勵機制 + 協議效率** |
| Exp 7 | V-G | 可擴展性 |

---

## 九、待辦清單

### 實作優先順序
1. [ ] 結果儲存框架
2. [ ] CIFAR-100 + STL-10 數據載入
3. [ ] 三維異質性生成器
4. [ ] PAID-FD 完整實作
5. [ ] Baseline 實作 (FedMD, Fixed-ε)
6. [ ] CSRA 實作（如無開源）
7. [ ] FedGMKD, FedFed（待確認開源）

### 開源確認（你來找）
- [ ] CSRA (TIFS 2024)
- [ ] FedGMKD (NeurIPS 2024)
- [ ] FedFed (NeurIPS 2023)

---

## 十、確認清單

- [x] 方法名稱: PAID-FD
- [x] 數據集: CIFAR-100 (private) + STL-10 (public)
- [x] Baseline: 6 methods (頂級會議/期刊)
- [x] 異質性: 3D (通訊 + 隱私 + 運算)
- [x] 能耗模型: 包含 Inference Energy
- [x] 協議對比: One-shot vs Auction
- [x] 結果儲存: 獨立存檔機制

---

**最終版本確認完成，明天開始寫程式碼！**
