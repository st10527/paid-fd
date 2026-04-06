# 04 — 待討論議題

> 建立日期：2026-03-29  
> 目的：列出需要決定的設計問題，每個議題附上選項與影響分析

---

## Issue #1：Cubic Solver 的 Cost 正規化（� MODELING — 保持現狀）

### 問題

`stackelberg.py` 中的 cubic 方程使用簡化公式：
$$\lambda_i \varepsilon^3 + \lambda_i \varepsilon^2 + (1-p)\varepsilon + 1 = 0$$

完整的 FOC-derived cubic 應該是：
$$\frac{\lambda_i}{c_i} \varepsilon^3 + \frac{\lambda_i}{c_i} \varepsilon^2 + \left(1 - \frac{p}{c_i}\right)\varepsilon + 1 = 0$$

### 實驗驗證結果

嘗試修正（傳入 p/c 和 λ/c）後，game 完全崩潰：

| γ | 簡化版 participation | 正確版 participation |
|---|---------------------|---------------------|
| 3 | 40% | 0% |
| 5 | 60% | 0% |
| 7 | 90% | 0% |
| 10 | 100% | 0% |
| 20 | 100% | 40%（但 ε*=0.014，無法用） |

原因：c ≈ 0.1-0.4 → λ/c 是 λ 的 2.5-10 倍 → ε* 縮小到 0.01-0.08 → LDP noise ≈ 50-400 → 完全淹沒信號。

### 決定：保持簡化版

理由：
1. 簡化版產生 ε* ∈ [0.3, 0.9]，在 LDP 實用範圍內
2. s-FOC 完全滿足（∂U/∂s ≈ 0）
3. γ sweep 行為符合理論預期（P2, P3, P4 的定性性質都成立）
4. 論文可以將此框架為「sequential optimization」（先解 ε-quality tradeoff，再解 s-cost tradeoff）

### 論文處理建議

在 Algorithm 1 (Device Best Response) 中，將求解步驟描述為：
1. 給定 price p，求解 privacy-quality tradeoff → ε*
2. 給定 ε*，求解 upload-cost tradeoff → s* = p/c - (1+ε*)/ε*
3. 檢查 participation condition

---

## Issue #2：s* 是否應該控制實際上傳量

### 問題

理論中 $q(s_i, \varepsilon_i) = \ln(1 + s_i \cdot g(\varepsilon_i))$，s_i 代表 device 上傳的
logit 向量數量。但程式碼中所有 device 都對全部 20000 個 public samples 算 logits 並上傳。

### 選項

| 方案 | 做法 | 優點 | 缺點 |
|------|------|------|------|
| **A: 按 s* 限制上傳** | 每個 device 只上傳 min(s*, n_public) 個 logit vectors | 完全對齊理論 | 需要處理 index 對齊；可能大幅降低 accuracy |
| **B: 重新定義 s_i** | 把 s_i 重新定義為 local training effort（如 epoch 數或 batch 數）| 不需改 logit pipeline | 需要修改理論推導 |
| **C: 維持現狀** | 所有 device 上傳全部 logits，s* 只影響 cost | 最簡單；accuracy 最好 | 理論與實作不一致 |
| **D: 在論文中重新框架** | 解釋 s_i 控制的是 communication cost（不是 actual count） | 不改 code | 需要仔細的論文敘述 |

### 分析

如果用方案 A：
- Device 上傳 s* 個 logits（s* 可能只有 200-1000）
- Server 需要在所有 device 的共同 subset 上 aggregate（或處理不同 device 不同 subset）
- Accuracy 可能下降很多（因為只用部分 public data）
- 但 γ 的效果會更明顯（大 γ → 大 s* → 更多 logits → 更好）

**建議**：先用方案 C 跑 Phase 0，如果 γ gap 不夠大，再嘗試方案 A。

### 需要討論

- [ ] 論文的理論推導是否已經寫死 s_i 是 logit count？還是可以重新框架？
- [ ] 如果用方案 A，如何處理不同 device 上傳不同 subset 的對齊問題？

---

## Issue #3：LDP Sensitivity 定義

### 問題

Logit vector 有 K=100 個 class。目前 per-element 加 noise：

$$\text{noise per element} = \text{Lap}(0, 2C/\varepsilon)$$

這給每個 element ε-LDP，但整個 K-dim vector 的 privacy 是：
- Basic composition: $K \cdot \varepsilon = 100\varepsilon$
- Advanced composition (RDP): roughly $O(\sqrt{K} \cdot \varepsilon)$

### 選項

| 方案 | noise scale | per-vector privacy | 實際 noise 量 |
|------|------------|-------------------|--------------|
| **Per-element** (現況) | $2C/\varepsilon$ | ~$K\varepsilon$ (basic) | 小 |
| **Per-vector** | $2CK/\varepsilon$ | $\varepsilon$ | 大（100x） |
| **RDP composition** | $2C/\varepsilon'$ where $\varepsilon' = \varepsilon/\sqrt{2K\ln(1/\delta)}$ | $(\varepsilon, \delta)$-DP | 中 |

### 建議

保持 per-element（現況），但在論文中：
1. 說明使用 per-element Laplace mechanism
2. 用 advanced composition 報告 per-round total privacy
3. 用 RDP 做跨 round 的 accounting

### 需要確認

- [ ] 論文 T6 的 noise floor 項用的是哪個 variance？
- [ ] 論文 T7-8 的 composition 用什麼 accounting method？

---

## Issue #4：P6 驗證的公平性

### 問題

Proposition 6 聲稱「在相同 total privacy budget 下，heterogeneous ε (PAID-FD)
的聚合方差 ≤ uniform ε (Fixed-ε)」。

但目前的 Fixed-ε 實驗用固定值（0.5 或 1.0），沒有 match PAID-FD 的 total budget。

### 建議做法

1. 先跑 PAID-FD，記錄每一 round 的 per-device ε 分佈
2. 計算平均：$\bar{\varepsilon} = \frac{1}{N} \sum_i \varepsilon_i^*$
3. 跑 Fixed-ε with $\varepsilon = \bar{\varepsilon}$
4. 比較兩者的 aggregation 方差和 accuracy

### 實驗設計

```
Experiment P6:
  PAID-FD: γ=5, seed=42, 100 rounds → 記錄 avg_eps per round
  Fixed-ε-matched: ε = mean(avg_eps), 100 rounds
  
  比較：
  - 每一 round 的 aggregation 方差
  - 最終 accuracy
  - 收斂速度
```

### 需要確認

- [ ] 是否需要 match 每一 round 的 total budget？還是整體 avg 就好？
- [ ] 如果 PAID-FD 的 avg ε 是 0.67，Fixed-ε-0.67 作為對照組夠不夠？

---

## Issue #5：FedMD 的 distill_epochs 和 clip_bound

### 問題

| 參數 | PAID-FD | Fixed-ε | FedMD |
|------|---------|---------|-------|
| distill_epochs | 1 | 1 | 5 |
| clip_bound | 2.0 | 2.0 | 5.0 |

FedMD 有兩個優勢：(1) 更多 distill epochs (2) 更大 clip bound。

### 分析

- **distill_epochs**: FedMD 用 5 是因為 clean logits 可以承受更多 epoch 不 overfit。
  PAID-FD 用 1 是因為 noisy logits 多 epoch 會 overfit to noise。這是合理的。
  
- **clip_bound**: FedMD 用 5.0 是因為不加 noise 所以 sensitivity 無關。PAID-FD 用 2.0
  是因為 sensitivity = 2C，較小 C 降低 noise。這也是合理的。

### 建議

維持現狀，但在論文中記錄這些差異。也可以做 ablation：
- PAID-FD with distill_epochs=5（看 overfit 效果）
- PAID-FD with clip_bound=5（看 noise 增大的效果）

---

## Issue #6：v8 是否足以讓 γ 產生 Accuracy Gap

### 核心疑慮

v8 每 round 只做 1 epoch distillation（distill_epochs=1），但 public data 有 20000 samples。
如果 1 epoch 的 KL distillation 就足以讓 server model 學到 aggregated logits 的知識，
那不同品質的 aggregation 可能都「足夠好」，導致 γ 差異不大。

### 可能的調整

1. **減少 public_samples**：20000 → 5000。讓每一 round 的 distillation signal 更稀缺，
   品質差異更明顯。
   
2. **增加 distill_epochs**：1 → 3。讓 server model 更深入學習 aggregated knowledge，
   使 noise 差異更明顯。
   
3. **降低 distill_lr**：0.001 → 0.0005。更保守的 learning，需要更多 rounds 收斂。

4. **降低 pretrain_epochs**：10 → 5。降低 pretrain 的效果，讓 distillation 更重要。

### 建議

Phase 0 Step 0 用預設設定跑。如果 γ gap < 1%，按上述順序嘗試調整。

---

## Issue #7：v8 pretrain 是否太強

### 問題

Pretrain 10 epochs on 20000 public data with augmentation，可能已經讓 server model
達到不錯的 accuracy（可能 30-40%）。之後的 FD rounds 只是在 pretrain 基礎上微調。

如果 pretrain 太強，後續 FD rounds 的 marginal improvement 很小，導致不同 γ 的差異
被 pretrain 的高起點淹沒。

### 建議

在 Phase 0 中記錄 pretrain 後（round 0 開始前）的 accuracy。如果 > 35%，考慮：
1. 減少 pretrain_epochs (10 → 5 → 3)
2. 降低 pretrain_lr (0.1 → 0.05)
3. 使用 random init (no pretrain) 作為 ablation

---

## 決策追蹤

| Issue | 狀態 | 決定 | 日期 |
|-------|------|------|------|
| #1 Cubic solver | ✅ 保持 | 簡化版有效，論文用 sequential optimization 框架 | 2026-03-29 |
| #2 s* 上傳量 | ⏳ 待討論 | Phase 0 先用方案 C | — |
| #3 LDP sensitivity | ⏳ 待確認 | 保持 per-element，論文說明 | — |
| #4 P6 公平性 | ⏳ 待設計 | Phase 2 時做 budget-matching 實驗 | — |
| #5 FedMD 參數 | ✅ 暫定 | 維持現狀，論文說明 | 2026-03-29 |
| #6 γ gap 疑慮 | ⏳ 待驗證 | Phase 0 結果出來後決定 | — |
| #7 pretrain 過強 | ⏳ 待觀察 | 記錄 pretrain accuracy 再判斷 | — |
