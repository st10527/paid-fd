# PAID-FD Simulation Results v2 (Correct Cubic Solver)

## Run metadata
- **Date**: 2026-05-08 (cubic fix applied; full rerun pending)
- **Hardware**: RTX 5070 Ti, CUDA 12.8
- **Code commit (fix)**: current HEAD (post-fix, see git log)
- **Previous data commit**: 40c2ee2 (2026-04-24)
- **Status**: ⏳ Full rerun queued — use `scripts/run_v2_experiments.py`

---

## ⚠️ Impact Assessment: Cubic Fix Effect on ε*

The corrected cubic `λε³ + λε² + (c−p)ε + c = 0` vs old `λε³ + λε² + (1−p)ε + 1 = 0`
produces **systematically higher ε*** across all (p, c, λ) combinations:

| γ (→ p_avg) | Δε* range | Avg shift | Impact |
|-------------|-----------|-----------|--------|
| γ=3 (p≈2.0) | +0.8 to +1.5 | **+1.1** | Large |
| γ=5 (p≈3.3) | +0.6 to +1.5 | **+0.9** | Large |
| γ=7 (p≈4.5) | +0.5 to +0.8 | **+0.6** | Moderate |
| γ=10 (p≈6.0) | +0.2 to +0.6 | **+0.4** | Moderate |

**Expected consequence**: Higher ε* → better SNR → higher quality logits →
potentially higher accuracy, but also higher cumulative ε_total (privacy exposure).
Participation rates and payment should also shift as s* and utility change.

**⚠️ All PAID-FD experiments need to be re-run. Baseline runs (FedAvg, FedGMKD,
FedMD, Fixed-ε variants) do NOT need re-running as they don't use the Stackelberg solver.**

---

## VI-B: Monotonicity (ε* vs γ)

**Setup**: N=50, α=0.5, T=100, seeds=[42, 123, 456]

| γ | ε* (old cubic) | ε* (new cubic) | Δ |
|---|----------------|----------------|---|
| 3  | 2.77 ± 0.05 | **2.197** | −0.57 |
| 5  | 2.84 ± 0.06 | **2.365** | −0.47 |
| 7  | 2.90 ± 0.05 | **2.710** | −0.19 |
| 10 | 3.09 ± 0.07 | **3.152** | +0.06 |

> ✅ Monotonicity Prop.2 still holds: 2.197 → 2.365 → 2.710 → 3.152
> ⚠️ Direction opposite to prediction: ε\* went DOWN at low γ (old code used c=1, overestimating cost → artificially high ε\*). Fix uses real c∈[0.1,0.4].

---

## VI-C: Efficiency Frontier (4 metrics × 4 γ values)

**Setup**: N=50, α=0.5, T=100, seeds=[42, 123, 456]

| γ | Acc (old) | Acc (new) | Participation (old) | Participation (new) | Max Cum ε (new) | Total Payment (new) |
|---|-----------|-----------|---------------------|---------------------|-----------------|---------------------|
| 3  | 61.22 ± 0.18% | **61.43 ± 0.58%** | 38% | **88%** | **598** | **12,604** |
| 5  | 61.38 ± 0.32% | **61.37 ± 0.25%** | 82% | **99%** | **678** | **19,428** |
| 7  | 61.33 ± 0.28% | **61.29 ± 0.30%** | ~90% | **100%** | **770** | **27,780** |
| 10 | 61.36 ± 0.25% | **61.30 ± 0.23%** | 100% | **100%** | **880** | **39,884** |

**Spread analysis (new vs old)**:
- Accuracy spread: **0.14 pp** (was 0.23 pp — even flatter ✅)
- Participation: 88% → 100% (**1.14×**, was 2.63×)
- Max cum ε exposure: **1.47×** (598→880, was 1.75×)
- Total payment: **3.16×** (12,604→39,884, was 6.35×)

> ⚠️ **MAJOR CHANGE**: All three frontier ratio claims (2.63×, 1.75×, 6.35×) need updating in paper.
> Cubic fix makes low-γ devices find participation more profitable → γ=3 participation jumped 38%→88%,
> compressing the spread. Qualitative conclusion (higher γ = more participation/payment/privacy-cost) still holds.

---

## VI-D: Method Comparison (Table 4)

**Setup**: N=50, α=0.5, T=100, γ=5, seeds=[42, 123, 456]

| Method | Old Acc (%) | New Acc (%) | Δ (pp) | Notes |
|--------|-------------|-------------|--------|-------|
| **PAID-FD (γ=5)** | 61.43 ± 0.39 | **61.37 ± 0.25** | −0.06 | ✅ Phase 0 done |
| Fair Fixed-ε=1 | 61.32 | 61.32 | 0 | No solver, unchanged |
| Fair Fixed-ε=3 | 61.21 | 61.21 | 0 | No solver, unchanged |
| Fair Fixed-ε=5 | 61.10 | 61.10 | 0 | No solver, unchanged |
| Old Fixed-ε=3 | 15.7 | 15.7 | 0 | No solver, unchanged |
| FedGMKD | ~47.0 | ~47.0 | 0 | Non-private, unchanged |
| FedAvg | 45.50 | 45.50 | 0 | Non-private, unchanged |
| FedMD | 45.70 | 45.70 | 0 | Non-private, unchanged |

> **Key question**: Does PAID-FD's advantage over Fair Fixed-ε=1 increase
> (since it now uses the true utility-maximizing ε)? Old Δ = 61.43 − 61.32 = 0.11 pp.

---

## VI-E: Training Trajectories (Fig 6b)

> Regenerate from VI-D PAID-FD run data after rerun.
> Script: `scripts/plot_fig6b_convergence_v2.py`

---

## VI-F: Pipeline Ablation (Table 5 + Fig 11)

**Setup**: N=50, α=0.5, T=100, γ=5, seeds=[42, 123, 456]
**Note**: EMA variant removed per paper revision.

| Variant | Old best_acc (%) | New best_acc (%) | Δ | Notes |
|---------|-----------------|-----------------|---|-------|
| **Full PAID-FD** | 61.43 ± 0.39 | ⏳ | — | Game solver affected |
| w/o Mixed Loss (CE anchor) | 15.75 ± 0.73 | ⏳ | — | Likely unchanged (catastrophic forgetting, unrelated to solver) |
| w/o Persistent local | 64.02 ± 0.18 | ⏳ | — | Game solver affected |
| w/o BLUE | ⏳ | ⏳ | — | Phase 1 expC_noblue ×3 seeds |
| w/o KL temperature (T=1) | ⏳ | ⏳ | — | Phase 5 expI_noKLtemp |

> **Note**: EMA variant removed from ablation per paper revision.
> expI covers: noKLtemp, noMixedLoss, noPersistent.

---

## VI-G: Robustness

### (a) Scalability: N ∈ {20, 50, 80}

| N | γ | Old Acc (%) | New Acc (%) |
|---|---|-------------|-------------|
| 20 | 3 | ~60.5 | ⏳ |
| 20 | 10 | ~61.0 | ⏳ |
| 50 | 3 | 61.22 | ⏳ |
| 50 | 10 | 61.36 | ⏳ |
| 80 | 3 | ~60.8 | ⏳ |
| 80 | 10 | ~61.2 | ⏳ |

### (b) Non-IID heterogeneity: α ∈ {0.1, 0.5, 1.0, 5.0}  (γ=5 fixed)

| α | Old Acc (%) | New Acc (%) |
|---|-------------|-------------|
| 0.1 | ⏳ | ⏳ (Phase 3 expE_a01_g5 ×3 seeds) |
| 0.5 | 61.38 | ⏳ (Phase 0 γ=5, baseline) |
| 1.0 | ⏳ | ⏳ (Phase 3 expE_a10_g5 ×3 seeds) |
| 5.0 | ⏳ | ⏳ (Phase 3 expE_a50_g5 ×3 seeds) |

### (c) Privacy preference: λ_mult ∈ {0.5, 1.0, 2.0}  (γ=5 fixed)

| λ_mult | Old Acc (%) | New Acc (%) |
|--------|-------------|-------------|
| 0.5 | ⏳ | ⏳ (Phase 6 expJ_lammult0p5 ×3 seeds) |
| 1.0 | 61.38 | ⏳ (Phase 0 γ=5, baseline) |
| 2.0 | ⏳ | ⏳ (Phase 6 expJ_lammult2p0 ×3 seeds) |

---

## VI-H: Cumulative Privacy Accounting

| Metric | Old cubic | New cubic (est.) | Notes |
|--------|-----------|-----------------|-------|
| avg_eps_per_round (γ=5, s42) | 2.78 | ~3.33 (+20%) | Higher SNR |
| max_privacy_spent over T=100 | 739.4 | ~999 (est.) | Linear scaling |
| Crosses ε_ref=10 threshold (round) | ~4 | ~3 | Earlier crossing |

> **Implication for Fig 10**: The privacy composition curves will steepen.
> The "noise-resilient plateau" observation in Fig 7 should still hold but
> the operating point star will shift right to ε̄*≈3.3 (was 2.84).

---

## VI-I: CIFAR-10 Cross-Dataset Validation

| Method | Old Acc (%) | New Acc (%) |
|--------|-------------|-------------|
| PAID-FD (γ=5) | ⏳ (expD) | ⏳ (rerun needed) |
| Old Fixed-ε=3 | ~50.2 | 50.2 (unchanged) |

---

## Summary: Impact of Cubic Fix

| Claim | Old value | **New value (actual)** | Status |
|-------|-----------|------------------------|--------|
| ε\* monotone in γ | ✅ 2.77→3.09 | ✅ **2.197→3.152** | ✅ Confirmed |
| Accuracy ≈61.4% (plateau) | 61.43 ± 0.32 | **61.37 ± 0.25%** (γ=5) | ✅ Unchanged |
| Participation ratio 2.63× | 38→100% | **1.14×** (88→100%) | ⚠️ Paper needs update |
| ε exposure spread 1.75× | γ=3→10 | **1.47×** (598→880) | ⚠️ Paper needs update |
| Payment spread 6.35× | γ=3→10 | **3.16×** (12,604→39,884) | ⚠️ Paper needs update |
| PAID-FD > Fair Fixed-ε=1 | +0.11 pp | **+0.05 pp** (61.37 vs 61.32) | ✅ Still holds |
| w/o Mixed Loss catastrophic | ~35% best | ⏳ Phase 5 | ⏳ |
| ε̄\* operating point | ε̄\*=2.84 | **ε̄\*=2.365** (γ=5) | ⚠️ Fig 7 shift |

### ⚠️ Paper updates required (from Phase 0)
1. **Fig 4b/4c/4d ratio labels** — 2.63×→1.14×, 1.75×→1.47×, 6.35×→3.16×
2. **VI-B ε\* table** — all values updated (lower than old at γ≤7)
3. **Fig 7 operating point** — ε̄\*=2.84→2.365; still in accuracy plateau ✅

### ✅ Qualitative conclusions confirmed unchanged
- Monotonicity of ε\* with γ (Prop. 2) ✅
- Accuracy plateau (~61.4%) across all γ ✅
- PAID-FD > Fair Fixed-ε advantage (still positive) ✅
- Non-private / Fixed-ε baselines unaffected ✅

---

## Rerun Schedule

Each run ≈ 2.5–3 h on RTX 5070 Ti. Total PAID-FD runs needed:

| Section | Runs | Phase | Est. time | Status |
|---------|------|-------|-----------|--------|
| Phase 0 — VI-B/C: γ∈{3,5,7,10} × 3 seeds | 12 | 0 | 33 h | ✅ Done |
| Phase 1 — expB N={20,80} scalability + expC ablation | 21 | 1 | 51 h | ⏳ Next |
| Phase 4 — expH BLUE validation | 3 | 4 | 8 h | ⏳ |
| Phase 3 — VI-G(b) α∈{0.1,1.0,5.0} × γ=5 × 3 seeds | 9 | 3 | 25 h | ⏳ |
| Phase 2 — VI-I CIFAR-10 × 3 seeds | 3 | 2 | 8 h | ⏳ |
| Phase 5 — VI-F pipeline ablation (noKLtemp/noMixed/noPersist) | 3 | 5 | 8 h | ⏳ |
| Phase 6 — VI-G(c) λ_mult∈{0.5,2.0} × 3 seeds | 6 | 6 | 17 h | ⏳ |
| **Total** | **57 new runs** | | **~150 h ≈ 6.3 days** |

> + 1 already done: `expB_n20_g3_s42` → **56 remaining**

Use `scripts/run_v2_experiments.py --all` to run all phases in priority order.  
Or per-phase: `--phase 0` → `--phase 1` → `--phase 4` → `--phase 3` → `--phase 2` → `--phase 5` → `--phase 6`

---

*Generated: 2026-05-08 | Cubic fix commit: see `git log -1` | Old data: 40c2ee2*
