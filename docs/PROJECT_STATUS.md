# PAID-FD 專案狀態

> 📌 **Living Document** — 記錄當前狀態與下一步  
> 最後更新：2026-04-17

---

## 快速概覽

| 項目 | 狀態 |
|------|------|
| 目標期刊 | IEEE Transactions on Mobile Computing (TMC) |
| 當前版本 | **v10.1**（Persistent Models + Persistent Adam + Solver Fix） |
| GPU Server | `aelab-2@163.13.127.232`（NVIDIA RTX 5070 Ti） |
| TWCC | u4622524 / ACD114197（V100-SXM2-32GB） |
| GitHub | `git@github.com:st10527/paid-fd.git` |
| 🎯 目前階段 | **寫作** — Phase 5 跑完後開始畫圖 |

---

## 實驗進度

| Phase | 內容 | Runs | 狀態 | 平台 |
|-------|------|------|------|------|
| Phase 1 | CIFAR-100 core (Exp A/A'/B/C) | 33 | ✅ 完成 | TWCC |
| Phase 2 | CIFAR-10 cross-dataset (Exp D) | 9 | ✅ 完成 | aelab-2 |
| Phase 3 | Non-IID α sweep (Exp E) | 12 | ✅ 完成 | aelab-2 |
| Phase 4 | Reviewer defense (Exp F/G/H) | 12 | ✅ 完成 | aelab-2 |
| Phase 5 | Pipeline internal ablation (Exp I) | 3 | 🔄 Running | aelab-2 |
| **Total** | | **69** | **66 done + 3 running** | |

---

## 核心結果摘要

### Accuracy (CIFAR-100, N=50, 100 rounds)
| Method | Best Acc | Final Acc |
|--------|----------|-----------|
| **PAID-FD (game, γ=5)** | **61.4% ±0.3** | **60.8% ±0.2** |
| Fair Fixed-ε=1 (same pipeline, no game) | 61.3% | 60.8% |
| Fair Fixed-ε=3 | 61.2% | 60.2% |
| FedGMKD (no privacy) | 46.5% | 46.5% |
| FedAvg (no privacy) | 45.5% | 44.7% |
| Old Fixed-eps-3 (weak pipeline) | 41.9% | 36.5% |
| CSRA | 1.0% | 1.0% |

### Key Findings
1. **Pipeline ≈ +20pp** over non-persistent baselines
2. **Game ≈ 0pp accuracy** but controls participation (79% vs 100%) and cost
3. **Privacy-utility curve**: flat for ε≥0.5, breaking point at ε=0.1 (56.2%)
4. **Non-IID robust**: Δ<0.5pp across α=0.1→1.0
5. **CIFAR-10**: 84.9% (vs Fixed-eps-3: 68.5%)

### Two Orthogonal Contributions
- **C1: Noise-robust FD pipeline** → accuracy ceiling (~61%)
- **C2: Stackelberg game mechanism** → efficiency frontier (cost, participation, privacy allocation)

---

## 論文結構 (11 sections, ~11 pages)

| Section | 頁數 | 狀態 |
|---------|------|------|
| I: Introduction | 1 | 📝 待寫 |
| II: System Model | 1 | 📝 待寫 |
| III: Incentive Mechanism | 2 | 📝 待寫 |
| IV: Privacy Analysis | 1 | 📝 待寫 |
| V: PAID-FD Algorithm | 1 | 📝 待寫 |
| VI: Experimental Evaluation | 3.5 | ✅ Skeleton complete (`paper/section_vi_results.tex`) |
| VII: Related Work | 0.75 | 📝 待寫 |
| VIII: Discussion & Limitations | 0.5 | 📝 待寫 |
| IX: Conclusion | 0.25 | 📝 待寫 |

---

## Figure/Table Mapping (12 total)

| # | Type | Content | Data Source | Section |
|---|------|---------|-------------|---------|
| Fig 1 | 概念圖 | System architecture | — | Sec II |
| Fig 2 | 概念圖 | Stackelberg game flow | — | Sec III |
| Fig 3 | Figure | Prop 2 monotonicity | v10.1 γ sweep | VI-B |
| Fig 4 | Figure | Efficiency frontier | v10.1 γ sweep | VI-C |
| Fig 5 | Figure | λ sensitivity | v10.1 λ sweep | VI-D |
| Table II | Table | Method comparison | Phase 1+4 | VI-E |
| Fig 6 | Figure | Convergence curves | Phase 1 | VI-E |
| Fig 7 | Figure | Privacy-utility curve | Phase 4 Exp G | VI-F |
| Table III | Table | Pipeline vs Game ablation | Phase 1+4+5 | VI-G |
| Fig 8 | Figure | N scalability | Phase 1 Exp B | VI-H |
| Table IV | Table | CIFAR-10 cross-dataset | Phase 2 | VI-I |
| Fig 9 | Figure | α Non-IID robustness | Phase 3 | VI-J |
| Fig 10 | Figure | Privacy composition | v10.1 replot | VI-K |

---

## 下一步

1. ⏳ 等 Phase 5 結果（~3hr on aelab-2）
2. 🎨 一口氣畫出全部 10 張 figures
3. ✍️ 填入 Section VI 所有數字
4. ✍️ 寫 Section I-V, VII-IX
