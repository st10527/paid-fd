# PAID-FD 實驗結果記錄

> 📌 **Living Document** — 每次實驗結果出來時修改此文件  
> 最後更新：2026-04-01

---

## 實驗結果時間線

| 日期 | 版本 | 實驗 | 核心結果 |
|------|------|------|---------|
| 03/04 | v7 | 7-method comparison (3 seeds) | PAID-FD 60.04%, γ gap < 1.2% |
| 03/29 | v8.0 | Phase 0 γ sweep | ALL崩潰 44%→7-16% |
| 03/29 | v8.1/v8.2 | Phase 0.1 diagnostic (6 configs) | Adam bug + denoising D8 smoking gun |
| 03/30 | v8.3 | Pure BLUE γ sweep (100 rounds) | γ 全 33-34%, gap 0.3% |
| 03/31 | v9.0 | γ sweep (fixed solver) | Game 正確, distillation 仍 45%→35% |
| 04/01 | v9.1 | Distill fix sweep (8 configs) | 🔄 Running on GPU |

---

## Phase 0.1 — v8 Diagnostic（2026-03-29）

**Version**: v8.0 (Adam) + v8.1 (fresh SGD) + v8.2 (denoising)  
**Setting**: 50 devices, CIFAR-100, Dirichlet α=0.5, seed=42, 20 rounds

### v8.0 Results (Adam optimizer — catastrophic)

| Config | Description | R1 | R20 | Trend |
|--------|-------------|-----|-----|-------|
| C1 | C=1.0, lr=0.001 | 25.0% | 7.5% | ❌ monotonic ↓ |
| C2 | C=2.0, lr=0.0001 | 40.2% | 16.4% | ❌ monotonic ↓ |
| C6 | **FedMD oracle** | 36.0% | 16.0% | ❌ monotonic ↓ |

**Root Cause**: Adam state leak + cumulative drift + no anchor → death spiral

### v8.2 Results (fresh SGD + denoising)

| Config | Description | R1 | Final | Verdict |
|--------|-------------|-----|-------|---------|
| D1 | denoise + pure KL (γ=5) | 34.4% | 25.3% | ❌ Denoise harmful |
| D2 | denoise + pure KL (γ=3) | 34.0% | 24.1% | ❌ |
| D3 | denoise + pure KL (γ=10) | 34.0% | 24.5% | ❌ |
| **D4** | **no denoise, pure KL (γ=5)** | **39.3%** | **38.7%** | **✅ Stable** |
| **D5** | **CE anchor α=0.5 (γ=5)** | **43.4%** | **45.0%** | **✅ Best** |
| D6 | denoise + CE α=0.3 (γ=5) | 40.2% | 43.4% | ✅ OK |
| D7 | FedMD oracle (no noise) | 43.5% | 41.2% | ⚠️ Mild drift |
| **D8** | **FedMD oracle + denoise** | **40.2%** | **28.8%** | **💀 SMOKING GUN** |

**Key Findings**:
1. Fresh SGD fixes catastrophic collapse (D4 stable at 38.7%)
2. D8 proves denoising destroys dark knowledge even on clean logits
3. CE anchor gives best accuracy (45%) but masks γ effect

---

## v8.3 — Pure BLUE γ Sweep（2026-03-30）

**Version**: v8.3 (D4 config: no denoise, no CE, fresh SGD, pure KL)  
**Setting**: γ={2,3,5,7,10}, 100 rounds, seed=42

| γ | Part% | avg ε | R1 acc | Final acc | Best |
|---|-------|-------|--------|-----------|------|
| 2 | 0% | — | 44.4% | 44.4% | 44.4% |
| 3 | 38% | 0.89 | — | 33.8% | — |
| 5 | 70% | 0.52 | — | 33.5% | — |
| 7 | 86% | 0.41 | — | 33.6% | — |
| 10 | 100% | 0.32 | — | 33.9% | — |

**Conclusion**: All active γ degrade to ~34%. No differentiation (gap = 0.3%).  
**Insight**: BLUE aggregate noise $\propto 1/\sum\varepsilon_i^2$. With old solver (small ε), Σε² saturates → no γ effect.

---

## v9.0 — Fixed Solver γ Sweep（2026-03-31）

**Version**: v9.0 (np.roots cubic solver)  
**Setting**: γ={2,3,5,7,10}, 50 rounds, seed=42, C=2, T=3  
**GPU**: aelab-3 (RTX 5070 Ti)

### Game-Level Results ✅

| γ | Price p* | Part% | avg ε* | avg s* | avg SNR |
|---|---------|-------|--------|--------|---------|
| 2 | 1.85 | 0% | — | — | — |
| 3 | 2.89 | 38% | 2.83 | 13.4 | 6.1 |
| 5 | 3.96 | 82% | 2.78 | 19.3 | 34.5 |
| 7 | 4.71 | 86% | 2.88 | 23.8 | 67.2 |
| 10 | 5.63 | 100% | 3.15 | 28.1 | 433 |

Solver fix verified: ε* = 2-3 (was 0.3-0.8), SNR >> 1, Proposition 2 holds.

### Distillation Results ❌

| γ | Pretrain | Final | Δ |
|---|---------|-------|---|
| 2 | 45.4% | 45.4% | 0% (no one participates) |
| 3 | 39.2% | 35.2% | -4.0% |
| 5 | 39.2% | 35.1% | -4.1% |
| 7 | 39.2% | 35.1% | -4.1% |
| 10 | 39.4% | 35.4% | -4.0% |

**ALL active γ degrade to ~35%, gap = 0.3%**

### Teacher Signal Diagnosis

Monte Carlo simulation on clipped + noised + BLUE-aggregated + softmax logits:

| C | T | Correct Class Prob | vs Uniform (1.0%) |
|---|---|-------------------|-------------------|
| 2 | 3 | 1.93% | +0.93% (near-uniform → useless) |
| 2 | 1 | 6.55% | +5.55% |
| 5 | 3 | 4.51% | +3.51% |
| **5** | **1** | **32.5%** | **+31.5% (17× stronger)** |

**Root Cause**: C=2 truncates too much, T=3 flattens the rest → teacher signal near-uniform on K=100 classes.

---

## v9.1 — Distillation Fix Sweep（🔄 Running）

**Version**: v9.1 (C/T/CE sweep with fixed solver)  
**Setting**: 8 configs × 50 rounds, seed=42  
**GPU**: aelab-3, started ~2026-04-01  
**Script**: `scripts/run_phase0_v9_1_distill_fix.py`

### Configs

| Config | C | T | CE α | γ | Purpose |
|--------|---|---|------|---|---------|
| D1 | 2 | 3 | 0 | 5 | v9.0 baseline |
| D2 | 5 | 1 | 0 | 5 | ⭐ Best signal (theory) |
| D3 | 5 | 2 | 0 | 5 | Moderate T |
| D4 | 5 | 1 | 0.3 | 5 | CE anchor effect |
| D5 | 2 | 1 | 0 | 5 | Isolate T effect |
| D6 | 5 | 3 | 0 | 5 | Isolate C effect |
| D7 | 5 | 1 | 0 | 10 | γ high |
| D8 | 5 | 1 | 0 | 3 | γ low |

### Decision Framework (Q1/Q2/Q3)

| Q | Check | Target |
|---|-------|--------|
| Q1 | D2 final ≥ R1 − 2%? | Distillation doesn't degrade |
| Q2 | D7 > D2 > D8, gap > 3%? | γ differentiation works |
| Q3 | D3 (C=5, T=3) also works? | Soft labels preserved |

### Results: (待填)

```
TODO: 填入 GPU 結果
```

---

## 附錄：TMC 實驗缺口清單

以下為 TMC 審稿人可能要求的實驗（從 v7 project_status 提取，仍然有效）：

### 🔴 P0: Must-Have
- **P0-A**: 多數據集（至少 +1: CIFAR-10）
- **P0-B**: non-IID 程度 α sweep（α ∈ {0.1, 0.3, 0.5, 1.0}）
- **P0-C**: 設備規模 N sweep（N ∈ {10, 20, 50, 100}）

### 🟡 P1: Strongly Recommended
- **P1-D**: 模型異質性（不同設備用不同 model）
- **P1-E**: 通訊效率量化（logits 400B vs parameters 44MB）

### 🟢 P2: Nice-to-Have
- **P2-F**: Public data size 敏感度
- **P2-G**: Budget constraints
- **P2-H**: Temperature 敏感度（部分由 v9.1 涵蓋）

> 📁 完整計畫：`docs/archive/03_experiment_plan.md`, `docs/archive/tmc_experiment_plan_final.md`
