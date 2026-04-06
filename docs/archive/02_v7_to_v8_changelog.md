# 02 — v7 → v8 變更記錄

> 建立日期：2026-03-29  
> 目的：記錄 v7 到 v8 的所有變更，方便 Claude Web 上回溯

---

## 背景：為什麼要從 v7 轉到 v8

### v7 的核心問題

v7 使用 **persistent local models**：每個 device 在 round 之間保留自己的 model，
持續在上面累積訓練。經過足夠多 rounds 後，所有 device 的 model 都趨近收斂
（~60% on CIFAR-100），不管 γ 值多少。

**結果**：γ=3 和 γ=10 的 accuracy 差距 <1.2%，完全無法展示 game mechanism 的價值。

### v8 的解決方案

v8 = **Standard FD flow**：每個 round，device 拿到 server model 的 fresh copy，
本地訓練幾個 epoch，算 logits，加 noise，上傳。每個 round 的 distillation 品質
完全取決於「這一輪」的參與者品質。

**預期**：γ（控制 price → participation + ε*）應該直接影響每一輪的聚合品質 → accuracy。

---

## 檔案變更清單

### 完全重寫的檔案

| 檔案 | 主要變更 |
|------|---------|
| `src/methods/paid_fd.py` | 移除 persistent models、EMA buffer、CE anchor、mixed loss |
| `src/methods/fedmd.py` | 移除 persistent models，改為 fresh copy each round |
| `src/methods/fixed_eps.py` | 移除 persistent models、EMA、mixed loss |

### 修改的檔案

| 檔案 | 主要變更 |
|------|---------|
| `scripts/run_all_experiments.py` | `_create_method()` 移除 ema_momentum, distill_alpha, use_ema, use_mixed_loss |
| `scripts/run_routeB_gpu.py` | EXP6 ablation variants 從 6 個減為 3 個 |

### 新建的檔案

| 檔案 | 用途 |
|------|------|
| `scripts/run_phase0_v8.py` | Phase 0 驗證 runner |
| `scripts/generate_v8.py` | v8 程式碼產生器（一次性使用）|
| `docs/project_status.md` | v8 實驗計畫（取代 v7 版本）|

### 備份的檔案

| 備份 | 原檔 |
|------|------|
| `src/methods/paid_fd_v7.py` | v7 版 PAID-FD |
| `src/methods/fedmd_v7.py` | v7 版 FedMD |
| `src/methods/fixed_eps_v7.py` | v7 版 Fixed-ε |
| `docs/project_status_v7.md` | v7 實驗狀態 |

---

## Config 欄位變更

### PAIDFDConfig

| 欄位 | v7 | v8 | 說明 |
|------|----|----|------|
| gamma | ✅ | ✅ | 不變 |
| delta | ✅ | ✅ | 不變 |
| budget | ✅ | ✅ | 不變 |
| local_epochs | ✅ (5) | ✅ (5) | 不變 |
| local_lr | ✅ (0.01) | ✅ (0.01) | 不變 |
| local_momentum | ✅ (0.9) | ✅ (0.9) | 不變 |
| distill_epochs | ✅ (1) | ✅ (1) | 不變 |
| distill_lr | ✅ (0.001) | ✅ (0.001) | 不變 |
| temperature | ✅ (3.0) | ✅ (3.0) | 不變 |
| pretrain_epochs | ✅ (10) | ✅ (10) | 不變 |
| pretrain_lr | ✅ (0.1) | ✅ (0.1) | 不變 |
| clip_bound | ✅ (2.0) | ✅ (2.0) | 不變 |
| public_samples | ✅ (20000) | ✅ (20000) | 不變 |
| use_blue | ✅ | ✅ | 不變 |
| use_ldp | ✅ | ✅ | 不變 |
| **ema_momentum** | ✅ (0.9) | ❌ 移除 | v8 無 EMA buffer |
| **distill_alpha** | ✅ (0.7) | ❌ 移除 | v8 純 KL（無 CE 混合）|
| **use_ema** | ✅ (True) | ❌ 移除 | v8 無 EMA |
| **use_mixed_loss** | ✅ (True) | ❌ 移除 | v8 無 mixed loss |

### FedMDConfig

| 欄位 | v7 | v8 |
|------|----|----|
| 所有基本欄位 | ✅ | ✅（不變）|
| **persistent local_models dict** | ✅ | ❌ 移除 |

### FixedEpsilonConfig

| 欄位 | v7 | v8 |
|------|----|----|
| 所有基本欄位 | ✅ | ✅（不變）|
| **ema_momentum** | ✅ | ❌ 移除 |
| **distill_alpha** | ✅ | ❌ 移除 |

---

## 行為變更

### PAID-FD run_round()

```
v7:
  device → 使用 persistent local model（跨 round 保留）
  → local train（累積到之前的 state）
  → compute logits → add noise
  → EMA buffer 更新 logits（動量=0.9）
  → BLUE aggregate EMA logits
  → mixed loss: α·KL + (1-α)·CE_on_public

v8:
  device → copy_model(server_model)  ← fresh copy
  → local train（從 server state 開始）
  → compute logits → add noise
  → BLUE aggregate THIS round's noisy logits
  → pure KL distillation (no CE anchor)
```

### FedMD run_round()

```
v7:
  device → 使用 persistent local model（跨 round 保留）
  → local train → compute clean logits → aggregate

v8:
  device → copy_model(server_model) ← fresh copy
  → local train → compute clean logits → aggregate
```

### Ablation Variants (EXP6)

```
v7 (6 variants):
  1. Full (EMA + BLUE + LDP + Mixed)
  2. No-EMA
  3. No-BLUE
  4. No-LDP
  5. No-Mixed (pure KL)
  6. Bare-FD (minimal)

v8 (3 variants):
  1. Full (BLUE + LDP)
  2. No-BLUE (equal weights + LDP)
  3. No-LDP (BLUE + clean logits)
```

---

## Git History

- **Commit eaa944f**: `v8: Standard FD flow — no persistent models, no EMA, pure KL`
  - 14 files changed, +3145, -739

---

## 所有 v7 實驗數據已失效

以下數據來自 v7，不能用於 v8 論文：

- `routeB_exp1_merged_3seeds.json` — 7-method comparison
- `routeB_exp6_merged_3seeds.json` — Ablation
- 所有 Phase 1 sweep 結果 (γ, λ)
- 所有 19 張 PDF figures
- `phase2_comparison_cifar100_*.json`

保留作為歷史參考，但不會出現在論文中。
