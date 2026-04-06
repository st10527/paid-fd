# PAID-FD 設計決策追蹤

> 📌 **Living Document** — 做出設計決策或發現新問題時修改此文件  
> 最後更新：2026-04-01

---

## 決策總覽

| # | Issue | 狀態 | 決定 | 日期 |
|---|-------|------|------|------|
| 1 | Cubic solver 正規化 (c=1) | ✅ 保持簡化 | 論文用 sequential optimization 框架 | 03/29 |
| 2 | Cubic solver 雙根 bug | ✅ 已修復 | np.roots + utility-max 選根 | 03/31 |
| 3 | s* 是否控制上傳量 | ⏳ 待討論 | 暫用方案 C（全部上傳，s* 只影響 cost） | — |
| 4 | LDP sensitivity 定義 | ⏳ 待確認 | 保持 per-element，論文需說明 | — |
| 5 | P6 驗證公平性 | ⏳ 待設計 | Phase 2 時做 budget-matching 實驗 | — |
| 6 | FedMD 參數不對稱 | ✅ 暫定 | FedMD(C=5, epochs=5) vs PAID-FD(C=2, epochs=1)，合理 | 03/29 |
| 7 | Denoising 策略 | ✅ 決定 | 移除 class-conditional denoising（D8 證明有害） | 03/29 |
| 8 | CE anchor 角色 | ✅ 決定 | 不作為主要機制（masks γ effect），可選 (α ≤ 0.3) | 03/29 |
| 9 | Optimizer 選擇 | ✅ 決定 | Fresh SGD per round（不用 Adam） | 03/29 |
| 10 | Clip bound C | 🔄 v9.1 測試中 | 可能從 C=2 改為 C=5 | 04/01 |
| 11 | Temperature T | 🔄 v9.1 測試中 | 可能從 T=3 改為 T=1 or T=2 | 04/01 |

---

## Issue #1：Cubic Solver 正規化（✅ 保持 c=1）

### 問題
完整 FOC 推導的 cubic：$\frac{\lambda}{c}\varepsilon^3 + \frac{\lambda}{c}\varepsilon^2 + (1-\frac{p}{c})\varepsilon + 1 = 0$  
程式碼使用簡化版（c=1）：$\lambda\varepsilon^3 + \lambda\varepsilon^2 + (1-p)\varepsilon + 1 = 0$

### 驗證結果
- 正確版（p/c, λ/c）→ 0% participation（c≈0.1-0.4 讓 λ/c 過大 → ε*≈0.01）
- 簡化版 → 正常 participation，s-FOC 滿足，ε-FOC 近似滿足

### 決定
保持簡化版。論文框架：sequential optimization（先解 ε by privacy-quality tradeoff，再解 s by upload-cost tradeoff）。

---

## Issue #2：Cubic Solver 雙根 Bug（✅ 已修復）

### 問題
$f(\varepsilon) = \lambda\varepsilon^3 + \lambda\varepsilon^2 + (1-p)\varepsilon + 1 = 0$ 有兩個正根：
- Root 1（小 ε ≈ 0.3-0.8）：SNR < 1，不滿足 Proposition 2
- Root 2（大 ε ≈ 2-8）：SNR >> 1，滿足 Proposition 2

舊的 bisection solver 在 [0, 100] 找根，always hit Root 1。

### 修復（v9.0, commit `b7d7d14`）
```python
# stackelberg.py: _solve_cubic_bisection()
coeffs = [lambda_i, lambda_i, (1 - p), 1]
all_roots = np.roots(coeffs)
# Filter: positive, real, then pick utility-maximizing root
```

### 驗證
| γ | Old ε* (Root 1) | New ε* (Root 2) | Old SNR | New SNR |
|---|-----------------|-----------------|---------|---------|
| 5 | 0.52 | 2.78 | 0.87 | 34.5 |
| 10 | 0.32 | 3.15 | 0.57 | 433 |

---

## Issue #3：s* 上傳量控制（⏳ 待討論）

### 問題
理論：$q(s_i, \varepsilon_i) = \ln(1 + s_i \cdot g(\varepsilon_i))$，s_i = 上傳 logit 數量。  
程式碼：所有 device 對全部 20,000 public samples 算 logits，s* 只影響 energy 計算。

### 選項
| 方案 | 做法 | 優缺點 |
|------|------|--------|
| A | 按 s* 限制上傳 | 完全對齊理論，但需 index 對齊，accuracy 可能大幅下降 |
| B | 重新定義 s_i | 如 local training effort，需修改理論 |
| **C (現用)** | 維持全部上傳 | 最簡單，但理論與實作不一致 |
| D | 論文重新框架 | s_i 控制 communication cost（非 actual count） |

### 狀態
Phase 0 先用方案 C。如果 v9.1 結果良好且 γ gap 不夠大，可嘗試方案 A 來放大 γ effect。

---

## Issue #4：LDP Sensitivity 定義（⏳ 待確認）

### 問題
K=100 class 的 logit vector，目前 per-element 加 Lap(0, 2C/ε)。

- **Per-element ε-LDP**（現用）：每個 class 的 logit 有 ε-LDP
- **Per-vector ε-LDP**：noise scale 需 ×100 倍
- 大多數 FD 論文使用 per-element 定義

### 需確認
- [ ] 論文 T6 noise floor 用哪個 variance？
- [ ] 論文 T7-8 composition 用什麼 accounting？

---

## Issue #5：P6 驗證公平性（⏳ 待設計）

### 問題
Proposition 6：同 total privacy budget → PAID-FD (heterogeneous ε) 的 BLUE 方差 ≤ Fixed-ε (uniform ε)。  
需要 budget-matching 實驗：$\varepsilon_{fixed} = \frac{1}{N_{part}} \sum_i \varepsilon_i^{PAID-FD}$

---

## Issue #7：Denoising 策略（✅ 移除）

### 決定：移除 class-conditional denoising

**Evidence (D8 smoking gun)**：  
FedMD oracle (clean logits) + denoising → 40.2% → 28.8%（catastrophic）  
FedMD oracle (clean logits) no denoising → 43.5% → 41.2%（mild）

**原因**：  
FD 的價值在 per-sample soft distribution（dark knowledge）。Class-conditional mean 把 200 個 distinct distributions 壓成 1 個 prototype → 丟失 inter-sample variation + confusion patterns。

**Key Insight**：
> 在 FD 中，signal IS the per-sample distribution。任何 collapse per-sample variation 的操作都 destroys dark knowledge。

---

## Issue #8：CE Anchor 角色（✅ 不作為主機制）

### 分析
- CE anchor (α=0.5) 達 45%，最高 accuracy
- 但 CE loss 不依賴 γ → 如果 CE 主導學習，γ 就無意義
- 與 v7 的問題相同：保護機制越強 → γ 差異越小

### 決定
- CE anchor 可以作為 optional component (α ≤ 0.3)
- 主實驗用 pure KL，CE anchor 放在 ablation study
- v9.1 的 D4 config 測試 CE α=0.3 的效果

---

## Issue #9：Optimizer 選擇（✅ Fresh SGD）

### 問題
v8.0 用 Adam（跨 round 累積 momentum → catastrophic collapse）

### 決定
每 round 建新 SGD optimizer（lr=0.001, momentum=0.9, weight_decay=5e-4）。  
無 persistent state → 無 accumulation → 穩定。

---

## Issues #10-11：Clip Bound C 和 Temperature T（🔄 v9.1 Testing）

### 問題
C=2, T=3 on K=100 classes → teacher signal near-uniform：
- clip(C=2) 截斷太多 → logits 被壓在 [-2, 2]
- softmax(T=3) 進一步 flatten → correct class prob 僅 1.93%

### 理論分析（Monte Carlo）

| C | T | Correct Class Prob | Signal Strength |
|---|---|-------------------|-----------------|
| 2 | 3 | 1.93% | ❌ Near-uniform |
| 2 | 1 | 6.55% | ⚠️ Weak |
| 5 | 3 | 4.51% | ⚠️ Weak |
| **5** | **1** | **32.5%** | **✅ Strong (17×)** |

### v9.1 測試
D1(C=2,T=3) vs D2(C=5,T=1) vs D3(C=5,T=2) vs D5(C=2,T=1) vs D6(C=5,T=3)

結果將決定最終的 C 和 T 值。

---

## 理論 vs 程式碼 對齊確認

以下為最後一次完整檢查（v9.0, 2026-03-31）：

| 理論 | 程式碼 | 狀態 |
|------|--------|------|
| T1: s* = p/c − (1+ε)/ε | `s_star = p / c - (1 + eps_star) / eps_star` | ✅ |
| T1: ε* cubic equation | `np.roots()` + utility-max selection | ✅ (v9.0 fixed) |
| P2: Ternary search for p* | `ServerPricing.solve()` | ✅ |
| P4: Participation U_i ≥ E_train | `participates = utility >= E_train` | ✅ |
| T2: Server utility U_ES = (γ−p)·ΣQ | `(self.gamma - p) * total_quality` | ✅ |
| T6: BLUE weights w_i ∝ ε_i² | `all_weights.append(device_eps ** 2)` | ✅ |
| T7-8: Composition tracking | `privacy_spent[dev_id] += eps` | ✅ |
| g(ε) = ε/(1+ε) | `QualityFunction.g()` | ✅ |
| q(s,ε) = log(1 + s·g(ε)) | `QualityFunction.q()` | ✅ |
| Standard FD: fresh copy/round | `copy_model(self.server_model)` | ✅ |
| LDP: Lap(0, 2C/ε) per element | `np.random.laplace(0, noise_scale, ...)` | ✅ |
| Pre-training on public data | `_pretrain_on_public()` | ✅ |

> 📁 完整對齊報告：`docs/archive/01_theory_code_alignment.md`
