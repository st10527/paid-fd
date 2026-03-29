# PAID-FD TMC 論文：v8 專案狀態報告

> 更新日期: 2026-03-29
> 目標期刊: IEEE Transactions on Mobile Computing (TMC)
> GPU: NVIDIA RTX 5070 Ti (aelab-2)

---

## ⚠️ 重大版本變更: v7 → v8

### v7 (已廢棄) vs v8 (現行) 差異

| 元素 | v7 (舊) | v8 (新) |
|------|---------|---------|
| **Local models** | Persistent (跨 round 累積) | Fresh copy each round (標準 FD) |
| **EMA buffer** | ✅ momentum=0.9 | ❌ 移除 |
| **Loss function** | Mixed: 0.7×KL + 0.3×CE | Pure KL (no CE anchor) |
| **Distill source** | EMA-smoothed logits | Single-round aggregated logits |
| **FedMD baseline** | Also persistent models | Also standard FD (fixed) |
| **Fixed-eps baseline** | Persistent + EMA + mixed loss | Standard FD (fixed) |

### 為什麼切到 v8?
- v7 persistent models 跨 round 累積訓練 → 不管 γ 多少，最終都收斂到 ~60%
- 這**掩蓋了 game 的效果** — γ 應該影響準確度但 v7 看不出來
- v8 standard FD: 每 round 的 distillation 品質取決於**這一輪**的參與者 → γ 直接影響 accuracy
- 這也更接近原始 FedMD 論文的協議

### v7 所有實驗結果已失效
以下結果為 v7 版本，**僅供歷史參考**，不會用於論文：
- Phase 1.1 γ sweep (全部 ~60%, 無差異)
- Phase 1.2 λ sweep (全部 ~60%)
- Exp 1: 7-method comparison (PAID-FD 60.04%)
- Exp 6: Ablation (CE anchor -46.1%)
- 19 張 PDF 圖表

📁 歷史記錄: `docs/project_status_v7.md`

---

## 一、v8 系統架構

### Pipeline (v8)
```
Pre-train(10ep, lr=0.1, SGD)
  → [每 round 開始]
  → 每個參與設備: 拿 server model 的 fresh copy
  → Local Train(5ep, lr=0.01, SGD+momentum=0.9)
  → Clip(C=2.0)
  → LDP Laplace(0, 2C/ε_i) — per-device noise
  → Upload noisy logits
  → Server: BLUE Aggregation(weights ∝ ε_i²)
  → Server: softmax(T=3.0)
  → Server: Pure KL distillation (Adam lr=0.001)
```

### v8 vs v7 Ablation 元件對比

| 元件 | v7 | v8 | v8 Ablation 意義 |
|------|----|----|-----------------|
| Persistent local models | ✅ | ❌ | v8 可做 "Persistent vs Standard" ablation |
| EMA buffer | ✅ | ❌ | 已移除，不需 ablation |
| CE anchor (mixed loss) | ✅ | ❌ | 已移除，不需 ablation |
| BLUE aggregation | ✅ | ✅ | vs uniform average |
| LDP noise | ✅ | ✅ | vs no privacy (oracle) |
| Temperature T=3 | ✅ | ✅ | vs T=1 (hard label) |
| Pre-training | ✅ | ✅ | vs random init |

---

## 二、實驗分階段規劃

### Phase 0: v8 驗證 (現在要做 🔴)

| Step | 內容 | 預估 GPU 時間 | 目的 |
|------|------|-------------|------|
| **Step 0** | γ sweep {2,3,5,7,10}, seed=42, 100 rounds | ~5h | 確認 γ 能拉開 accuracy gap |
| **Step 1** | Quick 7-method comparison, seed=42, 50 rounds | ~4h | 確認相對排名 |

📄 **腳本**: `scripts/run_phase0_v8.py`

**如果 Step 0 失敗** (γ 仍無 accuracy gap):
- 嘗試調整 `distill_epochs` (1→3→5)
- 嘗試調整 `distill_lr` (0.001→0.005→0.01)
- 嘗試調整 `clip_bound` C (2.0→1.0)
- 嘗試更多 `local_epochs` (5→10)

### Phase 1: 參數敏感度 (Phase 0 通過後)

| Sweep | 變量 | Values | Seeds | Rounds |
|-------|------|--------|-------|--------|
| 1.1 | γ | {2,3,5,7,10} | 3 | 100 |
| 1.2 | λ_mult | {0.3,0.5,1.0,2.0,3.0} | 1 | 100 |
| 1.3 | N (devices) | {10,20,50,100} | 1 | 100 |
| 1.4 | C (clip bound) | {0.5,1.0,2.0,5.0} | 1 | 100 |

### Phase 2: 方法比較 (Phase 1 定好 default params 後)

| 實驗 | 內容 | Seeds | Rounds |
|------|------|-------|--------|
| 2.1 | 7-method comparison (CIFAR-100) | 3 | 100 |
| 2.2 | 7-method comparison (CIFAR-10) | 3 | 100 |
| 2.3 | non-IID α sweep {0.1,0.3,0.5,1.0} | 1 | 100 |

### Phase 3: 深度分析 (Phase 2 完成後)

| 實驗 | 內容 |
|------|------|
| 3.1 | Ablation study (v8 版: BLUE/LDP/Pretrain/T/Persistent) |
| 3.2 | Privacy composition analysis |
| 3.3 | Communication efficiency comparison |

---

## 三、代碼狀態

### v8 已更新的文件 ✅
| 文件 | 變更 |
|------|------|
| `src/methods/paid_fd.py` | v8: standard FD, no persistent, no EMA, pure KL |
| `src/methods/fedmd.py` | v8: standard FD, no persistent models |
| `src/methods/fixed_eps.py` | v8: standard FD, no persistent, no EMA, no mixed loss |
| `scripts/run_all_experiments.py` | `_create_method()` updated for v8 configs |
| `scripts/run_routeB_gpu.py` | Ablation variants updated for v8 |
| `scripts/run_phase0_v8.py` | **新增** Phase 0 runner |

### v7 備份
| 文件 | 用途 |
|------|------|
| `src/methods/paid_fd_v7.py` | v7 PAID-FD 備份 |
| `src/methods/fedmd_v7.py` | v7 FedMD 備份 |
| `src/methods/fixed_eps_v7.py` | v7 Fixed-eps 備份 |
| `docs/project_status_v7.md` | v7 完整狀態報告 |

### 未修改的方法 (不需要改)
| 方法 | 原因 |
|------|------|
| FedAvg | 每 round 已是 fresh copy (parameter averaging) |
| FedGMKD | 每 round 已是 fresh copy |
| CSRA | 每 round 已是 fresh copy (parameter-level DP) |

---

## 四、接下來的行動

```
現在:
  1. 把 v8 code 推到 GPU server
  2. 跑 python scripts/run_phase0_v8.py --step 0
  3. 等結果 (~5h)

結果出來後:
  如果 γ gap > 3%  → Phase 0 Step 1 → Phase 1 → Phase 2 → Phase 3
  如果 γ gap 1-3%  → 嘗試 distill_epochs=3 or distill_lr=0.005
  如果 γ gap < 1%  → 需要重新診斷，可能要調整 pipeline
```

---

*v8 transition: 2026-03-29*
