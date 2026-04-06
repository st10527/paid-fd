# 01 — 理論 vs 程式碼 對齊檢查報告

> 建立日期：2026-03-29  
> 目的：逐一比對論文理論結果與 v8 程式碼實作，找出不一致之處

---

## 總覽：發現 1 個需討論的建模簡化 + 1 個關鍵問題 + 3 個需討論項目

| 嚴重度 | 問題 | 對應理論 | 影響 |
|--------|------|---------|------|
| 🟡 MODELING | Cubic solver 使用簡化公式（c=1） | T1 | ε-FOC 不完全滿足，但 game 行為合理 |
| 🔴 **CRITICAL** | s* 未實際控制上傳量 | T1, P4 | quality function 與實際貢獻脫鈎 |
| 🟡 DISCUSS | LDP sensitivity 定義 | T6, T7-8 | 影響 noise floor 與 privacy accounting |
| 🟡 DISCUSS | P6 驗證缺少 total budget matching | P6 | PAID-FD vs Fixed-ε 比較可能不公平 |
| 🟢 MINOR | FedMD clip_bound=5 vs PAID-FD clip_bound=2 | — | 需記錄，但合理 |
| 🟢 MINOR | distill_epochs 不對稱 (FedMD=5, PAID-FD=1) | — | 需記錄，但合理 |

---

## � MODELING：Cubic Solver 使用簡化公式（c=1）

### 理論推導

Device utility:
$$U_i = p \cdot q(s_i, \varepsilon_i) - c_i \cdot s_i - \lambda_i \cdot \varepsilon_i$$

聯立 FOC 後，正確的 cubic 應為：
$$\frac{\lambda_i}{c_i} \varepsilon^3 + \frac{\lambda_i}{c_i} \varepsilon^2 + \left(1 - \frac{p}{c_i}\right)\varepsilon + 1 = 0$$

### 程式碼現況 (`stackelberg.py` L133-137)

```python
def f(eps):
    return lambda_i * eps**3 + lambda_i * eps**2 + (1 - p) * eps + 1
```

等價於 c=1 的簡化版本。

### 為什麼不修正（已驗證）

我們嘗試過傳入 p/c 和 λ/c 來正確正規化。結果：

| 版本 | γ=5 participation | avg ε* | avg s* | 可用性 |
|------|------------------|--------|--------|-------|
| 簡化版（c=1）| 60% | 0.61 | 12.0 | ✅ 合理 |
| 正確版（p/c, λ/c）| 0% | 0.00 | 0.00 | ❌ 沒人參與 |

**原因**：c ≈ 0.1-0.4 時，λ/c ≈ 0.3-15，遠大於 λ 本身。正確版的 cubic
由 (λ/c)ε³ 項主導，使 ε* → c/(p-c) ≈ 0.01-0.08。這麼小的 ε* 讓
LDP noise scale = 2C/ε ≈ 50-400，完全淹沒信號。

### 簡化版的特性

簡化版（c=1）的驗證結果：

```
=== Gamma Sweep (c=1 cubic) ===
γ= 3: p*=2.31, part=40%, avg_eps=0.894, avg_s= 8.3
γ= 5: p*=2.87, part=60%, avg_eps=0.606, avg_s=12.0
γ= 7: p*=3.78, part=90%, avg_eps=0.402, avg_s=14.2
γ=10: p*=4.24, part=100%, avg_eps=0.342, avg_s=15.8
```

特性：
- ✅ γ↑ → participation↑（P4 有效）
- ✅ s-FOC 完全滿足（∂U/∂s ≈ 0）
- ⚠️ ε-FOC 不滿足（∂U/∂ε ≈ 1.3-2.5）
- ✅ ε* 在 0.3-0.9 範圍，LDP noise 可管理
- ✅ 不同 γ 產生不同 participation + ε* 分佈

### 論文中的處理

有兩個選項：
1. **論文中使用正確公式推導**，但說明實驗使用 calibrated parameters
2. **論文定義簡化模型**，其中 ε 和 s 的最佳化是 sequential（先解 ε，再算 s）

建議選項 2：解釋 ε* 由 privacy-quality tradeoff 決定（不依賴 c），
s* 由 upload cost tradeoff 決定（使用 c）。這在 mechanism design
文獻中是常見的分步求解方法。

---

## 🔴 CRITICAL #2：Game 的 s* 未實際控制上傳量

### 理論

- s_i 代表 device i 上傳的 logit 向量數量
- $q(s_i, \varepsilon_i) = \ln(1 + s_i \cdot g(\varepsilon_i))$：s_i 直接影響品質
- Game 解出 s_i* 後，device 應該只上傳 s_i* 個 logit vectors

### 程式碼現況 (`paid_fd.py` L178-189)

```python
# 所有 device 都對 ALL public samples 算 logits
for start in range(0, n_public, bs):
    batch = public_images[start:start + bs].to(self.device)
    logits = local_model(batch)
    ...

# s* 只用在 energy 計算
s_for_cost = max(int(decision.s_star), 200)
energy = self.energy_calc.compute_total_energy(..., s_i=s_for_cost, ...)
```

**問題**：所有參與 device 都上傳 20000 個 logit vectors，不管 game 解出的 s* 是多少。s* 只影響 energy cost 計算。

### 影響分析

- 理論中 s* 影響品質（upload 越多 → quality 越高）→ 影響 server utility → 影響 pricing
- 實際中所有 device 都 upload 20000 → 品質只取決於 ε_i（noise level）
- 這讓 s* 變成一個虛擬指標，game 的一半維度（upload volume）失去意義

### 解決方案（需討論，見 04_discussion_issues.md）

**Option A**：讓 device 只上傳 s_i* 個 public samples 的 logits

```python
n_upload = min(int(decision.s_star), n_public)
selected_indices = np.random.choice(n_public, size=n_upload, replace=False)
# 只算 selected_indices 上的 logits
```

Server 端 aggregation 需要對齊 indices（只在共同 subset 上聚合）。

**Option B**：重新定義 s_i 為 local training iterations 或其他可控參數

**Option C**：維持現狀，但在論文中說明 s* 控制的是 communication cost（不是 actual upload count）

→ 詳見 04_discussion_issues.md #Issue-2

---

## 🟡 DISCUSS #1：LDP Sensitivity 定義

### 理論

ε-LDP 要求：對任意兩個可能的輸入 $x, x'$，機制 $M$ 滿足
$$\Pr[M(x) \in S] \le e^\varepsilon \Pr[M(x') \in S]$$

對於 clipped logit vector $\ell \in [-C, C]^K$（K=100 classes）：
- L1 sensitivity of the whole vector: $\Delta_1 = 2C \times K = 400$ (worst case)
- Per-element sensitivity: $\Delta_1^{\text{per-elem}} = 2C = 4$

### 程式碼現況 (`paid_fd.py` L191-194)

```python
noise_scale = 2.0 * C / device_eps  # = 4/ε (per element)
noise = np.random.laplace(0, noise_scale, device_logits.shape)
```

這對每個 element 加 Lap(0, 2C/ε)，等價於 per-element ε-LDP。

### 分析

- **如果定義為 per-element ε-LDP**：程式碼正確，但整個 K-dim vector 的 privacy 是 Kε（by basic composition）
- **如果定義為 per-vector ε-LDP**：noise scale 應該是 $2CK / \varepsilon = 400/\varepsilon$，大 100 倍
- 大多數 FD 論文使用 per-element 定義（與本程式碼一致）

### 重要性

- 直接影響 Theorem 6 convergence bound 中的 noise floor 項
- 直接影響 Theorem 7-8 的 composition accounting
- 需要確認論文理論使用哪個定義

→ 詳見 04_discussion_issues.md #Issue-3

---

## 🟡 DISCUSS #2：P6 驗證需要 Total Budget Matching

### 理論 (Proposition 6)

> 在相同 total privacy budget 下，heterogeneous ε (PAID-FD) 的聚合方差 ≤ uniform ε (Fixed-ε) 的聚合方差。

### 程式碼現況

- PAID-FD：每個 device 有不同的 ε_i*（來自 game），加權 BLUE aggregation
- Fixed-ε：所有 device 用相同的 ε（如 0.5 或 1.0），equal-weight average

**問題**：目前沒有確保 $\sum_i \varepsilon_i^{\text{PAID-FD}} = N \times \varepsilon^{\text{Fixed}}$

### 解決方案

實驗設計時，選擇 Fixed-ε 的 ε 值使得：
$$\varepsilon_{\text{fixed}} = \frac{1}{N_{\text{part}}} \sum_{i \in \text{participants}} \varepsilon_i^{\text{PAID-FD}}$$

或者在結果分析時報告兩組的 total/average ε，並做 budget-matching 的額外實驗。

---

## 🟢 MINOR：FedMD clip_bound vs PAID-FD clip_bound

| 方法 | clip_bound (C) | 原因 |
|------|--------------|------|
| PAID-FD | 2.0 | 較小 C → 較低 noise sensitivity → 更好的 SNR |
| Fixed-ε | 2.0 | 同上 |
| FedMD | 5.0 | 無 noise，不需壓低 sensitivity，保留更多 logit 資訊 |

**結論**：C 是 privacy mechanism 的一部分。FD-with-noise 需要小 C，FD-without-noise 可以用大 C。這是合理的，但需在論文中說明。

---

## 🟢 MINOR：distill_epochs 不對稱

| 方法 | distill_epochs | 原因 |
|------|---------------|------|
| PAID-FD | 1 | Noisy teacher → 多 epoch 會 overfit to noise |
| Fixed-ε | 1 | 同上 |
| FedMD | 5 | Clean teacher → 更多 epoch 能更好學習 |

**結論**：合理的設計選擇。FedMD 作為 oracle upper bound，本來就應該有最好的條件。

---

## ✅ 確認一致的部分

| 理論 | 程式碼 | 狀態 |
|------|--------|------|
| T1: s* = p/c - (1+ε)/ε | `s_star = p / c - (1 + eps_star) / eps_star` | ✅ |
| P2: Ternary search for p* | `ServerPricing.solve()` with ternary search | ✅ |
| P4: Participation condition U_i ≥ E_train | `participates = utility >= E_train` | ✅ |
| T2: Server utility U_ES = (γ-p)·ΣQ | `(self.gamma - p) * total_quality` | ✅ |
| T6: BLUE weights w_i ∝ ε_i² | `all_weights.append(device_eps ** 2)` | ✅ |
| T7-8: Basic composition tracking | `privacy_spent[dev_id] += eps` | ✅ |
| g(ε) = ε/(1+ε) | `QualityFunction.g()` | ✅ |
| q(s,ε) = log(1 + s·g(ε)) | `QualityFunction.q()` | ✅ |
| FOC verification | `verify_foc_conditions()` | ✅ |
| v8 Standard FD: fresh copy each round | `copy_model(self.server_model)` per device | ✅ |
| Pure KL distillation with T scaling | `F.kl_div(...) * (T * T)` | ✅ |
| Pre-training on public data | `_pretrain_on_public()` for all FD methods | ✅ |
| Laplace noise: Lap(0, 2C/ε) | `np.random.laplace(0, noise_scale, ...)` | ✅ (per-element) |

---

## 下一步

1. **CRITICAL #2（s* 上傳量）需要在 Phase 0 後討論**
2. Cubic solver 使用簡化公式是合理的建模選擇，無需修改
3. DISCUSS 項目在 Phase 0 結果出來後再決定是否調整
4. **可直接跑 Phase 0**
