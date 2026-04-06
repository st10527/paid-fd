# PAID-FD 版本變更記錄

> 📌 **Living Document** — 每次版本更新時修改此文件  
> 最後更新：2026-04-01

---

## 版本總覽

| 版本 | 日期 | 核心變更 | 結果 | Commit |
|------|------|---------|------|--------|
| v7 | 02/09–03/04 | Persistent models + EMA + Mixed loss | 60% 穩定，但 γ gap < 1.2% | — |
| v8.0 | 03/29 | Standard FD: fresh copy, pure KL, Adam | 全部崩潰 (44%→7%) | `eaa944f` |
| v8.1 | 03/29 | Fresh SGD + CE anchor | 穩定 45%，但 CE 主導 → γ 無關 | — |
| v8.2 | 03/29 | Class-conditional denoising | D8 smoking gun：denoising 摧毀 dark knowledge | — |
| v8.3 | 03/30 | Pure BLUE (no denoise, no CE, fresh SGD) | γ 全部降到 33-34%，gap 0.3% | — |
| v9.0 | 03/31 | Cubic solver fix (np.roots, utility-max root) | Game 修好 (ε=2.8-3.1, SNR>>1)，但 C=2,T=3 teacher 近 uniform | `b7d7d14` |
| v9.1 | 04/01 | Distillation hyperparams sweep (C, T, CE) | 🔄 GPU 上跑中… | `1cd181b` |

---

## v7 — Persistent Local Models（已廢棄）

**期間**: 2026-02-09 ~ 2026-03-04（17 次修復迭代）

### 架構
```
Pre-train → Persistent local models（跨 round 累積訓練）
→ Clip(C=2) → LDP → BLUE(ε²) → EMA(0.9)
→ softmax(T=3) → Mixed Loss(0.7×KL + 0.3×CE) → Adam(lr=0.001)
```

### 開發歷程（17 fixes）
- Fix 1-3: 基礎框架 → ~10%
- Fix 4-5: Pre-training 突破 → ~40%
- Fix 6-9: Temperature, 噪聲抑制 → ~50%
- Fix 10-14: LDP 正確實現, BLUE 聚合 → ~55%
- Fix 15-17: EMA + Mixed Loss → **60-61% 穩定**

### 核心發現
- **SNR 反轉**：γ↑ → 更多人但 ε↓ → per-device noise↑ → SNR 反而↓
- **三重保護矛盾**：BLUE + EMA + CE 讓系統穩定，但同時消除 γ 差異
- **根本結論**：Persistent models 掩蓋 game 效果 → 所有 γ 收斂到同一 accuracy

### 所有 v7 實驗數據已失效
- Phase 1.1 γ sweep, Phase 1.2 λ sweep
- 7-method comparison (PAID-FD 60.04%)
- Ablation study (CE anchor -46.1%)
- 19 張 PDF 圖表

> 📁 詳細記錄：`docs/archive/route_a_report.md`, `docs/archive/project_status_v7.md`

---

## v8.0 — Standard FD Flow（失敗）

**日期**: 2026-03-29 | **Commit**: `eaa944f`

### 變更
- ❌ 移除 persistent local models → 每 round fresh copy from server
- ❌ 移除 EMA buffer
- ❌ 移除 CE anchor → pure KL
- ❌ 移除 mixed loss
- ✅ 保留 BLUE, LDP, temperature scaling
- Distillation: Adam lr=0.001

### 結果：全面崩潰
所有 config（包括 FedMD oracle）在 20 rounds 內從 44% 降到 7-16%。

### Root Cause
1. **Adam optimizer state leak**：跨 round 累積 momentum → 破壞性更新越來越大
2. **Cumulative server model drift**：model 逐 round 退化 → death spiral
3. **No anchor loss**：pure KL + T²=9 scaling → 無限漂移

> 📁 詳細記錄：`docs/archive/02_v7_to_v8_changelog.md`, `docs/archive/project_status.md`

---

## v8.1 — Fresh SGD + CE Anchor

**日期**: 2026-03-29

### 變更
- ✅ 修復 Adam → **Fresh SGD per round**（no persistent state）
- ✅ 新增 **CE anchor**：loss = α·CE + (1-α)·KL·T²
- SGD: lr=0.001, momentum=0.9, weight_decay=5e-4

### 結果
- D5 (CE α=0.5): 45.0%（✅ 穩定，有改善）
- 但 CE anchor 主導學習 → γ 透過 KL 的影響太小

### 判定
Claude Web 分析：CE anchor 是 band-aid，如果 CE 主導，γ 就無意義。

> 📁 詳細記錄：`docs/archive/05_phase01_diagnostic_report.md`

---

## v8.2 — Class-Conditional Denoising（失敗）

**日期**: 2026-03-29

### 變更
- ✅ 新增 `_denoise_logits()`：按 class 計算條件期望，取代 per-sample logit
- 假設：class-conditional mean 可以 average out LDP noise

### 結果（D1-D8 掃描）

| Config | Description | Final | Verdict |
|--------|-------------|-------|---------|
| D4 | no denoise, pure KL, fresh SGD | 38.7% | ✅ Stable |
| D5 | CE anchor α=0.5 | 45.0% | ✅ Best |
| D7 | FedMD oracle (no noise, no denoise) | 41.2% | ⚠️ Mild drift |
| **D8** | **FedMD oracle + denoise** | **28.8%** | **💀 Catastrophic** |

### Smoking Gun (D8)
D8 = clean logits + denoising → 40.2% → 28.8%。  
**Denoising 摧毀 dark knowledge，即使在無噪聲的乾淨 logits 上也如此。**

原因：class-conditional mean 把每 class 200 個不同 soft distributions 壓成 1 個 prototype → 學生從 100 個 prototype 學習而非 20,000 個 samples。

> 📁 詳細記錄：`docs/archive/06_v82_results_analysis.md`

---

## v8.3 — Pure BLUE γ Sweep（γ 無差異化）

**日期**: 2026-03-30

### 設定
D4 config（no denoise, no CE, fresh SGD, pure KL）× γ={2,3,5,7,10} × 100 rounds

### 結果
所有 active γ 降到 33-34%，gap 僅 0.3%。

### 分析：Impossible Triangle
計算 BLUE aggregate noise variance 後發現：n·ε² 隨 γ 反而 **降低**（更多人但 ε 更小）。  
Noise variance 公式：$\sigma_{BLUE}^2 \propto \frac{1}{\sum_i \varepsilon_i^2}$  
γ↑ → 更多 participants 但 ε↓ → Σε² 先升後飽和 → 差異太小。

---

## v9.0 — Cubic Solver Bug Fix（Game 修好，Distillation 仍失敗）

**日期**: 2026-03-31 | **Commit**: `b7d7d14`

### 發現
Cubic 方程 $\lambda\varepsilon^3 + \lambda\varepsilon^2 + (1-p)\varepsilon + 1 = 0$ 有 **TWO 正根**：
- Root 1（小 ε ≈ 0.3-0.8）：SNR < 1，Proposition 2 不成立 ← **舊 solver 找到這個**
- Root 2（大 ε ≈ 2-8）：SNR >> 1，Proposition 2 成立 ← **正確答案**

### 修復
重寫 `_solve_cubic_bisection()`：
- 使用 `np.roots()` 找所有根
- 篩選正實根
- 計算每個根的 utility，選 utility 最大的

### 驗證
| γ | participation | avg ε* | avg SNR | Prop 2 |
|---|-------------|--------|---------|--------|
| 3 | 38% | 2.83 | 6.1 | ✅ |
| 5 | 82% | 2.78 | 34.5 | ✅ |
| 10 | 100% | 3.15 | 433 | ✅ |

### GPU 結果（v9.0 γ sweep）
| γ | Pretrain | Final | Δ | Part% | avg ε |
|---|---------|-------|---|-------|-------|
| 2 | 45.4% | 45.4% | 0% | 0% | — |
| 3 | 39.2% | 35.2% | -4.0% | 38% | 2.83 |
| 5 | 39.2% | 35.1% | -4.1% | 82% | 2.78 |
| 10 | 39.4% | 35.4% | -3.6% | 100% | 3.15 |

Game 層面修好了（participation 正確分化，ε* 在合理範圍），但 **distillation 仍然退化**（all → 35%）。

### Teacher Signal 診斷
Monte Carlo 分析發現：C=2, T=3, K=100 → correct class probability = **1.93%**（uniform = 1.0%）。  
信號高於 uniform 只有 +0.93% → KL distillation 基本上是從近 uniform 分佈學習。

---

## v9.1 — Distillation Hyperparams Fix（🔄 Running）

**日期**: 2026-04-01 | **Commit**: `1cd181b`

### 理論基礎
Monte Carlo 驗證：
- C=2, T=3: correct class prob = 1.93%（無用）
- **C=5, T=1: correct class prob = 32.5%**（17× stronger signal）

### 掃描設定（8 configs × 50 rounds）

| Config | C | T | CE α | γ | 目的 |
|--------|---|---|------|---|------|
| D1 | 2 | 3 | 0 | 5 | Baseline (v9.0 params) |
| D2 | 5 | 1 | 0 | 5 | ⭐ Best theory signal |
| D3 | 5 | 2 | 0 | 5 | Moderate T |
| D4 | 5 | 1 | 0.3 | 5 | CE anchor effect |
| D5 | 2 | 1 | 0 | 5 | Isolate T effect (small C) |
| D6 | 5 | 3 | 0 | 5 | Isolate C effect (high T) |
| D7 | 5 | 1 | 0 | 10 | γ differentiation test (high) |
| D8 | 5 | 1 | 0 | 3 | γ differentiation test (low) |

### Decision Framework

| Question | Criterion | Meaning |
|----------|-----------|---------|
| Q1 | D2 final ≥ R1 − 2%? | Distillation works (no degradation) |
| Q2 | D8 vs D2 vs D7 gap > 3%, monotonic? | γ-accuracy hypothesis validated |
| Q3 | D3 (C=5, T=3) also works? | Soft label story preserved for paper |

### Outcome Scenarios
- ✅✅✅: Use C=5, T=3 (preserves soft label narrative)
- ✅✅❌: Use C=5, T=1 (works but hard labels)
- ✅❌—: γ no differentiation → efficiency story (Route B revisited)
- ❌——: Fundamental rethink needed

> 📁 Plotting script: `scripts/plot_v9_1_figures.py`（7 figures ready）

---

## 附錄：Git Commits

| Commit | Message |
|--------|---------|
| `eaa944f` | v8: Standard FD flow — no persistent models, no EMA, pure KL |
| `3cbe7b5` | Revert cubic solver to c=1 simplification |
| `b7d7d14` | v9.0: Fix cubic solver — np.roots finds all roots, picks utility-max |
| `1cd181b` | v9.1: Distillation fix sweep (C, T, CE) |
| `430fcf6` | v9.1: Plot script with 7 figures |
