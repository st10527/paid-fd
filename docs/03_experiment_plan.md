# 03 — 理論驅動實驗計畫

> 建立日期：2026-03-29  
> 來源：Claude Web 分析 + 理論 → 實驗對應表  
> 目的：從理論出發反推實驗需求，TMC 級別規劃

---

## 理論結果 → 實驗驗證 完整對應表

| # | 理論結果 | 論文聲明 | 驗證方式 | 預期圖表 | 實驗類別 |
|---|---------|---------|---------|---------|---------|
| T1 | Coupled Best Response | ε* 是 cubic 唯一正根，s* = p/c − (1+ε*)/ε* | 固定 p, 掃 (c, λ), 驗證 code 輸出 | Game mechanism fig | A |
| P2 | Monotonicity in p | p↑ → s*, ε*, q* 都↑ | 掃 p, 畫 s*, ε*, q* vs p 曲線 | Game mechanism fig | A |
| P3 | Monotonicity in costs | c↑→s*↓, λ↑→ε*↓ | 掃 c 和 λ, 驗證單調性 | Game mechanism fig | A |
| P4 | Participation threshold | ∃ p̲ 使得 p≥p̲ 才參與 | γ sweep → p 變動 → participation 變動 | γ sweep fig | B |
| T2 | SE 存在 | (p*, {s*, ε*}) 構成 SE | 驗證 solver 收斂, utility > 0 | Game mechanism fig | A |
| T5 | IC (Incentive Compatibility) | 報真值是 dominant strategy | **理論結果，不需實驗** | — | — |
| T6 | Convergence bound | 三項 bound: optimization + sampling + LDP noise floor | 跑足夠 rounds 看收斂行為 | 方法比較收斂曲線 | C |
| P6 | Heterogeneous ε 優於 uniform | Var[PAID-FD] ≤ Var[Fixed-ε] under same total ε | PAID-FD vs Fixed-ε 比較 | 方法比較 fig | C |
| T7-8 | Composition theorems | T 輪後的累積 privacy | 跑完後畫 cumulative ε per device vs rounds | Privacy analysis fig | D |
| — | Quality function q(s,ε) | log(1+s·g(ε)) 描述品質 | 不同 γ/λ 產生不同 q*, mapped to accuracy | γ/λ sweep fig | B |

---

## 實驗類別 A：Game Mechanism 驗證

**驗證目標**：T1, P2, P3, P4, T2

**不需要跑 training**，純粹是 game solver 的輸出。幾分鐘內完成。

### A1: Price sweep（驗證 P2 + T2）

```
掃 p ∈ [0.1, γ-0.1]，固定 50 devices
→ 畫 s*(p), ε*(p), q*(p), participation(p)
→ 驗證：全部單調遞增
→ 畫 U_ES(p)：倒 U 形
```

### A2: Cost sweep（驗證 P3）

```
固定 p=p*, 掃 c ∈ [0.05, 1.0], 固定 λ=0.5
→ 畫 s*(c): 應單調遞減
掃 λ ∈ [0.01, 2.0], 固定 c=0.15
→ 畫 ε*(λ): 應單調遞減
```

### A3: Cubic root verification（驗證 T1）

```
選 10 組 (p, c, λ), 用 solver 求 ε*
→ 代回 cubic 方程驗證 f(ε*) ≈ 0
→ 代回 FOC 驗證 ∂U/∂s ≈ 0, ∂U/∂ε ≈ 0
```

### A4: Participation threshold（驗證 P4）

```
掃 γ ∈ [1, 20], 看每個 γ 對應的 p* 和 participation rate
→ 應存在 threshold γ_min 使得 γ < γ_min 時沒人參與
```

**注意**：A 系列的 fig7a-d（v7 結果）可能還能用，因為 game solver 本身沒改。
但如果修了 CRITICAL #1（cubic solver 正規化），就需要重跑。

---

## 實驗類別 B：參數敏感度

**驗證目標**：P4 (accuracy mapping), quality function

**必須用 v8 重跑**。

### B1: γ sweep（最重要，Phase 0 Step 0）

```
γ ∈ {2, 3, 5, 7, 10}, seed=42, 100 rounds
→ 不同 p* → 不同 participation + ε* → 是否映射到不同 accuracy
→ 預期：γ↑ → accuracy↑（因為更多高品質參與者）
→ 關鍵指標：max-min accuracy gap > 3%
```

### B2: λ_mult sweep

```
λ_mult ∈ {0.3, 0.5, 1.0, 2.0, 3.0}, 固定 γ=5
→ 不同 λ → 不同 ε* → 不同 noise → accuracy 差距
→ 預期：λ↑ → ε*↓ → noise↑ → accuracy↓
```

### B3: N sweep（device 數量）

```
N ∈ {10, 20, 50, 100}, 固定 γ=5
→ 更多 device → aggregation noise ∝ 1/N → accuracy↑
→ 也影響 participation count
```

### B4: C sweep（clip bound / sensitivity-accuracy tradeoff）

```
C ∈ {0.5, 1.0, 2.0, 5.0, 10.0}, 固定 γ=5
→ 較小 C → 更多 clipping loss, 但更低 noise
→ 較大 C → 更少 clipping loss, 但更高 noise
→ 存在最優 C
```

---

## 實驗類別 C：方法比較（核心實驗）

**驗證目標**：T6, P6

### C1: 7-method comparison（Phase 0 Step 1 → Phase 2）

```
PAID-FD (γ=5), Fixed-ε-0.5, Fixed-ε-1.0, FedAvg, FedMD, FedGMKD, CSRA
× 3 seeds × 100 rounds

→ 收斂曲線（accuracy vs round）
→ 最終 accuracy bar chart with error bars
→ 驗證 T6 的三項 bound 行為
```

**預期排名**（v8）：
1. FedMD (oracle, no noise) — upper bound
2. PAID-FD (γ=5) — game-optimal noise allocation
3. Fixed-ε-1.0 — uniform noise, more noise than PAID-FD's best devices
4. Fixed-ε-0.5 — stronger privacy, much more noise
5. FedAvg — no public data, parameter averaging
6. FedGMKD — prototype-based, different mechanism
7. CSRA — DP on parameters (high-dim), auction-based

### C2: PAID-FD vs Fixed-ε under same total budget（驗證 P6）

```
跑完 C1 後，計算 PAID-FD 的 avg ε across rounds
設定 Fixed-ε 使其 total budget = PAID-FD 的 total budget
→ 直接驗證 P6
```

### C3: Multi-dataset（generalizability）

```
CIFAR-100, CIFAR-10, (optional: STL-10 as public)
→ 驗證方法在不同 dataset 上的表現一致性
```

---

## 實驗類別 D：深度分析

**驗證目標**：T7-8, ablation, communication

### D1: Privacy composition（驗證 T7-8）

```
跑 100 rounds 後，畫 per-device cumulative ε vs round
→ high-ε devices 累積更多 privacy cost
→ low-ε devices 累積較少
→ 對比 Fixed-ε (每人每 round 相同)
```

### D2: Ablation（v8 版）

```
v8 ablation 候選：
1. Full PAID-FD (BLUE + LDP + Pretrain + T=3)
2. No-BLUE (equal weights)
3. No-LDP (oracle, no noise)
4. No-Pretrain (random init → FD only)
5. T=1 (no temperature scaling)
6. distill_epochs=5 (more aggressive distillation)
7. v7-style (persistent models, as ablation comparison)
```

### D3: Communication efficiency

```
FD 方法: 每 device 上傳 logits = 100 classes × 4 bytes × S samples
FedAvg/CSRA: 每 device 上傳 model = ~44 MB

→ 畫 accuracy vs total communication cost
→ FD 方法在 communication 效率上有數量級優勢
```

### D4: Energy analysis

```
用 EnergyCalculator 算各方法的 total energy breakdown
→ training + inference + communication
→ PAID-FD 的 game 使低成本 device 貢獻更多
```

---

## 執行順序

```
0. 修 CRITICAL #1 (cubic solver)     ← 必須先做
1. Phase 0 Step 0: γ sweep           ← 驗證 v8 是否 work
   → gap > 3%?  → 繼續
   → gap < 1%?  → 需調整 (distill_epochs, distill_lr, C, local_epochs)

2. Phase 0 Step 1: 7-method quick    ← 確認排名合理
3. 實驗類別 A (game mechanism)        ← 幾分鐘搞定
4. 實驗類別 B (parameter sweeps)      ← 各掃一組
5. 實驗類別 C (3 seeds × 100 rounds) ← 核心數據
6. 實驗類別 D (analysis, ablation)    ← 補充分析
```

---

## Fallback Strategies

如果 Phase 0 γ sweep 沒有 gap：

1. **增加 distill_epochs**: 1 → 3 或 5
2. **增加 distill_lr**: 0.001 → 0.005
3. **降低 C**: 2.0 → 1.0（更低 noise but more clipping）
4. **增加 local_epochs**: 5 → 10（更好的 local model → 更好的 logits）
5. **降低 λ base values**：讓 ε* 到更大的範圍（降低 privacy 壓力）
6. **減少 devices**: 50 → 20（每個 device 更重要，noise 不容易被平均掉）
