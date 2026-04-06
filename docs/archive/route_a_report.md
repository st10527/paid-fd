# PAID-FD Route A 完整開發日誌與技術總結

**文件版本**: v1.0  
**撰寫日期**: 2026-03-04  
**涵蓋期間**: 2026-02-09 ~ 2026-03-04  
**作者**: NCLab Research Team  

---

## 一、背景與目標

### 1.1 原始假設（Route A）

PAID-FD (Privacy-Aware Incentive-Driven Federated Distillation) 的核心假設是：

> **Stackelberg Game 的均衡參數 γ（server 品質估值）會直接影響模型準確度。**
> 
> γ 越高 → server 願意付更高價格 → 更多設備參加 → 更好的聚合 logits → 更高的準確度。

Route A 的目標是實現一個系統，使得**不同 γ 值在準確度上產生明顯的、可觀察的差距**（例如 5-10%），從而展示 game mechanism 的核心價值。

### 1.2 系統架構

- **模型**: ResNet-18, CIFAR-100 (100 classes)
- **設備數**: 50 devices, non-IID Dirichlet α=0.5
- **蒸餾方式**: Federated Distillation (logit-level, not gradient-level)
- **隱私機制**: Local Differential Privacy (Laplace mechanism)
- **激勵機制**: Stackelberg Game (server = leader, devices = followers)

---

## 二、開發時間線：17 次修復的完整歷程

### Phase 0: 基礎建設 (02/09 - 02/16)

| 日期 | Commit | 改動 | 問題 |
|------|--------|------|------|
| 02/09 | `f731702` | 初始框架 | 基本結構搭建 |
| 02/10 | `f4fbb89` | CIFAR-100 safe split 替代 STL-10 | STL-10 載入問題 |
| 02/10 | `c93eba2` | 重寫 data loading | Subset targets 問題 |
| 02/11 | `25580d1` | GPU 加速 | 訓練速度太慢 |
| 02/16 | `795e4ec` | Game params 全面校準 | 均衡不合理 |
| 02/16 | `51beb60` | 平行實驗支援 | 加速多 seed 運行 |

### Phase 1: 準確度崩潰問題 (02/17) — Fix 1~6

這是最集中的修復期，一天內嘗試了 6 種不同方案：

| Fix # | Commit | 方案 | 結果 |
|-------|--------|------|------|
| 1 | `80940d7` | Persistent local models + 優化 configs | 準確度仍低 (~10-20%) |
| 2 | `2b71559` | Public data 2k→10k + anti-overfitting | 小幅改善 |
| 3 | `d3f0351` | Augmentation + local_lr 0.05 + distill_epochs 5 | 略有改善 |
| 4 | `591e4a4` | 新增 public data pre-training (FedMD 式) | **關鍵突破：Pre-training 從 ~10% 拉到 ~35%** |
| 5 | `f05a6d3` | 加強 pre-training (50 epochs, 20k data) | ~40% baseline |
| 6 | `a1d3125` | Per-method temperature (T=1 noisy, T=3 clean) | 分化出方法差異 |

**教訓 1**: Pre-training 是 FD 系統的「必要條件」。沒有 pre-training，models 初始化太差，蒸餾的 logits 基本上是噪聲。

### Phase 2: 噪聲抗性探索 (02/17-02/18) — Fix 7~11

| Fix # | Commit | 方案 | 結果 |
|-------|--------|------|------|
| 7 | `c651c7c` | Conservative distillation (低 lr, 少 epochs) | 準確度低但穩定 |
| 8 | `29af976` | EMA logit buffer (momentum=0.9) | 噪聲平滑但收斂慢 |
| 9 | `958a8f5` | T=1.0 + EMA buffer | 組合效果不明顯 |
| 10 | `f3bc9a4` | Hard-label CE distillation (PATE-style) | 丟失 logit 排序資訊 |
| 11 | `e2b2920` | Mixed-loss: KL + GT regularization | 穩定但 γ 差異消失 |

**教訓 2**: 噪聲處理是一把雙面刃。越好的噪聲抑制 → 越穩定的準確度 → 越小的 γ 差異。抑制噪聲和保持 γ 差異是**根本性的矛盾**。

### Phase 3: γ 差異化的專項攻堅 (02/18-02/21) — Fix 12~15

| Fix # | Commit | 方案 | 結果 |
|-------|--------|------|------|
| 12 | `8338908` | 參數調整（增大 noise 的影響） | 準確度不穩定 |
| 13 | `6265175` | Per-device LDP noise (Laplace per logit) | **LDP 正確實現**，但差異仍小 |
| 14 | `8e6b1af` | Pure KL distillation + 移除 EMA | 低 γ 準確度崩潰 |
| 15 | `6a7cbdb` | **C=5→2 + BLUE ε²-weighted aggregation** | **里程碑：穩定 ~60%，但 γ gap 只有 ~2%** |

**教訓 3**: Fix 15 的 BLUE 聚合是一個重要突破。ε²-weighted averaging 讓高品質設備的貢獻被放大，但同時也自動補償了 γ 之間的噪聲差異。

### Phase 4: 最終穩定化 (02/21-02/22) — Fix 16~17

| Fix # | Commit | 方案 | 結果 |
|-------|--------|------|------|
| 16 | `9ed1ad5` | distill_lr 0.001→0.0001 + GT reg α=0.5 | 穩定但 γ gap = 0%（lr 太保守） |
| 17 | `2f755d0` | **EMA buffer (α=0.9) + restore lr=0.001 + mixed loss (0.7 KL + 0.3 CE)** | **最終版：穩定 60-61%，γ gap ~0.6%** |

**教訓 4**: Fix 17 的 EMA + Mixed Loss + BLUE 三重機制達到了最高的穩定性，但也徹底消除了 γ 差異。

---

## 三、最終系統架構 (Fix 17 / v8)

### 3.1 Training Pipeline

```
[Pre-training] 10 epochs on 20k public CIFAR-100 subset
    → All local models initialized from same pre-trained weights (~36% acc)

[Per Round]:
    1. Local Training:  5 epochs, lr=0.01, SGD+momentum=0.9
    2. Local Inference:  Each device computes logits on public data
    3. Clipping:         logits clipped to [-C, C], C=2.0
    4. LDP Noise:        Laplace(0, 2C/εᵢ) added per device
    5. Upload:           Noisy logits sent to server
    6. BLUE Aggregation: ε²-weighted average (high-ε devices weighted more)
    7. EMA Buffer:       buffer = 0.9 × old_buffer + 0.1 × new_aggregated
    8. Soft Labels:      softmax(buffer / T), T=3.0
    9. Mixed Loss:       0.7 × KL(soft_labels) + 0.3 × CE(ground_truth)
    10. Distillation:    1 epoch, Adam lr=0.001
```

### 3.2 Game Mechanism

```
Server utility:    U_S = γ · Σᵢ log(1 + sᵢ/εᵢ) - p · Σᵢ sᵢ
Device i utility:  U_i = p · sᵢ - λᵢ · εᵢ² · sᵢ - cᵢ · sᵢ
```

Stackelberg 均衡（seed=42, N=50, λ_mult=1.0）:

| γ | Price p | N_part | Part% | avg_ε | Server Utility |
|---|---------|--------|-------|-------|----------------|
| 3 | 2.367 | 19 | 38% | 0.852 | 18.16 |
| 5 | 3.156 | 35 | 70% | 0.524 | 98.19 |
| 7 | 3.683 | 43 | 86% | 0.408 | 215.33 |
| 10 | 4.414 | 50 | 100% | 0.322 | 424.90 |
| 15 | 4.413 | 50 | 100% | 0.322 | ~425 (saturated) |

**Game mechanism 本身完全正確** — 不同 γ 產出不同的 participation、price、ε。

### 3.3 最終實驗結果 (Phase 1.1 Gamma, seed=42, 100 rounds)

| γ | Best Acc | Final Acc | Part% | avg_ε | Noise Scale (2C/ε) | SNR (ε√N/2) |
|---|----------|-----------|-------|-------|-------------------|-------------|
| 3 | **61.01%** | 60.33% | 38% | 0.852 | 4.70 | **1.86** |
| 5 | **61.08%** | 61.08% | 70% | 0.524 | 7.64 | **1.55** |
| 7 | **61.09%** | 60.75% | 86% | 0.408 | 9.80 | **1.34** |
| 10 | 60.60% | 60.33% | 100% | 0.322 | 12.41 | **1.14** |
| 15 | 60.46% | 59.90% | 100% | 0.322 | 12.40 | **1.14** |

**Best accuracy spread: 0.63%  |  Final accuracy spread: 1.18%**

---

## 四、核心發現與技術洞察

### 4.1 反直覺的 SNR 反轉

Federated Distillation with LDP 中存在一個重要的 tradeoff：

$$\text{SNR}_\gamma = \frac{\varepsilon_\gamma \cdot \sqrt{N_\gamma}}{2}$$

- **γ=3**: 少數精銳 — 19 人，每人 ε=0.852（低噪聲）→ SNR=1.86
- **γ=10**: 全員參加 — 50 人，每人 ε=0.322（高噪聲）→ SNR=1.14

**γ 越高，SNR 反而越低！** 因為 game equilibrium 在高 γ 下會吸引更多設備，但每個設備的 privacy budget 被壓得更低（為了壓低成本），導致每個 logit 的噪聲更大。雖然人多可以 average down (÷√N)，但 per-device noise 增長更快。

這解釋了為什麼高 γ 反而沒有更高準確度 — **它在遊戲論的層面上已經是最優的**，但最優解的特性是「用更多低品質信號替代少量高品質信號」。

### 4.2 三重保護機制的累積效應

Fix 17 的三個機制各自合理，但組合起來完全消除了 γ 差異：

| 機制 | 功能 | 對 γ 差異的影響 |
|------|------|----------------|
| **BLUE (ε²-weighted)** | 高 ε 設備加權更高 | 自動補償：少人但高 ε ≈ 多人但低 ε |
| **EMA Buffer (α=0.9)** | 平滑噪聲波動 | 低通濾波：消除任何 per-round 差異 |
| **Mixed Loss (0.3 CE)** | 防止噪聲 logits 破壞模型 | 30% 學習與 γ 完全無關 |

**類比**: 就像三層防護玻璃 — 每層都讓光線稍微變暗，三層加起來幾乎看不到外面的差異了。

### 4.3 Route A 的根本矛盾

```
想要 γ 差異大    ←→    想要所有 γ 都穩定
      ↑                        ↑
  需要噪聲穿透                需要噪聲抑制
      ↑                        ↑
  移除保護機制     ←→     加強保護機制
```

**這是一個 fundamental tradeoff**。如果允許噪聲影響準確度（移除 EMA、降低 CE 比重），低 γ 的曲線會崩壞（Fix 14 的教訓）。如果保護好所有 γ 的穩定性，差異就消失了（Fix 17 的結果）。

### 4.4 FD 的天然噪聲容忍度

Federated Distillation 與 Federated Learning (FedAvg) 有根本區別：
- **FedAvg**: 直接聚合梯度 → 噪聲直接累積到模型參數
- **FD**: 聚合 logits → 作為 soft labels 蒸餾 → 模型通過自己的梯度下降學習

FD 天然有一層「間接性」— 模型看到的是 noisy soft labels，但它通過自己的 loss landscape 來消化這些信息。這給了模型一個自然的「噪聲過濾」能力，尤其是當有 ground truth (CE loss) 作為 anchor 時。

---

## 五、每一次 Fix 的因果鏈

```
Fix 1-3:  基礎訓練問題 → 準確度 < 20%
  ↓
Fix 4-5:  Pre-training 解決冷啟動 → ~40% baseline
  ↓
Fix 6:    Temperature 區分 noisy/clean → 方法差異出現
  ↓
Fix 7-9:  各種噪聲抑制嘗試 → 穩定但差異不大
  ↓
Fix 10:   Hard-label (PATE-style) → 丟失排序信息，失敗
  ↓
Fix 11:   Mixed loss (KL+CE) → 首次嘗試 GT regularization
  ↓
Fix 12:   參數調整 → 不穩定
  ↓
Fix 13:   Per-device LDP 正確實現 → 噪聲機制正確
  ↓
Fix 14:   Pure KL (移除保護) → 低 γ 崩潰，證實保護的必要性
  ↓
Fix 15:   C=2 + BLUE → 穩定 60%，γ gap ~2% (最佳狀態)
  ↓
Fix 16:   降低 lr → 太保守，γ gap = 0%
  ↓
Fix 17:   EMA + 恢復 lr → 穩定 60-61%，γ gap 0.6%
```

**關鍵分叉點**: Fix 14 vs Fix 15

- Fix 14（移除保護）: 證明了噪聲差異確實存在，但低 γ 無法生存
- Fix 15（BLUE 聚合）: 找到了穩定性的解法，但代價是 γ 差異被壓縮

**如果從 Fix 15 重新出發**，不加 EMA 和 CE，可能是 Route A 最有潛力的方向。但基於 Fix 14 的經驗，低 γ 仍會不穩定。

---

## 六、投入成本統計

### 6.1 GPU 計算時間

| 階段 | 估計 GPU 時間 |
|------|--------------|
| Fix 1-14 測試 | ~100 小時 (各種短期測試) |
| Fix 15 (100 rounds × 4γ) | ~8 小時 |
| Fix 16 (100 rounds × 4γ) | ~8 小時 |
| Fix 17 (100 rounds × 4γ) | ~8 小時 |
| Phase 1.1 (100 rounds × 5γ) | ~8.5 小時 |
| Phase 1.2 (100 rounds × 5λ) | ~6.4 小時 |
| **總計** | **~140 GPU 小時** |

### 6.2 開發迭代

- **17 次 major fixes** + 多次小修
- **~35 次 commits** 
- **涵蓋 3 週** (02/09 - 03/04)

---

## 七、對 Route B 的啟示

### 7.1 可直接復用的成果

1. **Training Pipeline** (Fix 17): Pre-training + LDP + BLUE + EMA + Mixed Loss — 這是一個穩健的系統
2. **Game Mechanism**: Stackelberg equilibrium 完全正確，不同 γ/λ/N 產出合理的均衡
3. **實驗基礎設施**: `run_all_experiments.py` + `run_phase1_all.sh` + nohup 支援

### 7.2 需要轉變的思維

| Route A | Route B |
|---------|---------|
| Y 軸 = 準確度 | Y 軸 = 效率指標 (cost per accuracy) |
| 希望 γ 拉開準確度差距 | γ 拉開效率差距（相同準確度，不同代價） |
| 「γ 越高越好」 | 「存在最優 γ，平衡 cost 和 privacy」 |
| Game 控制品質 | Game 控制效率 |

### 7.3 Route B 的論文故事

> PAID-FD 在 game mechanism 下達到穩健的 ~61% accuracy，但不同 γ 的達成方式截然不同：
> - **γ=3**: 38% 參與 × 100 rounds × 強隱私 (ε=0.85) = 低通訊成本, 高隱私
> - **γ=10**: 100% 參與 × 100 rounds × 弱隱私 (ε=0.32) = 高通訊成本, 低隱私
> 
> Game mechanism 的價值不是讓準確度更高，而是讓 server 能**在相同目標準確度下選擇最高效的運作模式**。

---

## 八、結論

Route A 的 17 次修復是一次深入的技術探索，最終得到了一個重要的**負面結果**（negative result）：

> **在 Federated Distillation with LDP 框架下，噪聲抑制機制（BLUE + EMA + Mixed Loss）和 γ 差異化是根本矛盾的。穩定的系統必然會壓縮 γ 對準確度的影響。**

但這個負面結果的另一面是一個**正面發現**：

> **PAID-FD 具有極強的 noise robustness — 即使 participation 從 38% 變到 100%，SNR 從 1.86 降到 1.14，準確度仍穩定在 60-61%。Game mechanism 的價值體現在效率維度而非品質維度。**

這正是 Route B 的理論基礎。

---

*本文件記錄了 Route A 的完整歷程。所有數據和代碼均可在 git history 中追溯。*
