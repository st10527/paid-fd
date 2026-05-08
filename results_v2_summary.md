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
| 3  | 2.77 ± 0.05 | ⏳ TBD | — |
| 5  | 2.84 ± 0.06 | ⏳ TBD | — |
| 7  | 2.90 ± 0.05 | ⏳ TBD | — |
| 10 | 3.09 ± 0.07 | ⏳ TBD | — |

> Qualitative prediction: monotonicity Prop.2 (ε* increases with γ) should still hold.
> Magnitude will shift up uniformly by ~+0.9 (γ=5 reference).
> **Figure 3 will need regeneration after rerun.**

---

## VI-C: Efficiency Frontier (4 metrics × 4 γ values)

**Setup**: N=50, α=0.5, T=100, seeds=[42, 123, 456]

| γ | Acc (old) | Acc (new) | Participation (old) | Participation (new) | Max Cum ε (old) | Total Payment (old) |
|---|-----------|-----------|---------------------|---------------------|-----------------|---------------------|
| 3  | 61.22 ± 0.18% | ⏳ | 38% | ⏳ | ~500 | ⏳ |
| 5  | 61.38 ± 0.32% | ⏳ | 82% | ⏳ | ~740 | ⏳ |
| 7  | 61.33 ± 0.28% | ⏳ | ~90% | ⏳ | ~800 | ⏳ |
| 10 | 61.36 ± 0.25% | ⏳ | 100% | ⏳ | ~870 | ⏳ |

**Spread analysis (old)**:
- Accuracy spread: 0.23 pp (61.22–61.38%)  
- Participation: 38% → 100% (**2.63×**)
- Max cum ε exposure: **1.75×** (γ=3 to γ=10)
- Total payment: **6.35×** (γ=3 to γ=10)

> **Prediction for new cubic**: accuracy may increase slightly (higher SNR).
> Participation rates could change since equilibrium s* shifts. Payment likely higher.
> Frontier ratios (2.63×, 1.75×, 6.35×) are the key paper claims — watch these.

---

## VI-D: Method Comparison (Table 4)

**Setup**: N=50, α=0.5, T=100, γ=5, seeds=[42, 123, 456]

| Method | Old Acc (%) | New Acc (%) | Δ (pp) | Notes |
|--------|-------------|-------------|--------|-------|
| **PAID-FD (γ=5)** | 61.43 ± 0.39 | ⏳ TBD | — | Rerun needed |
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
| w/o BLUE | ⏳ | ⏳ | — | Game solver affected |
| w/o KL temperature | ⏳ | ⏳ | — | Game solver affected |

> **Note**: expI files cover noEMA, noMixedLoss, noPersistent.
> Need to add expI_noBLUE and expI_noKL experiments.

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

### (b) Non-IID heterogeneity: α ∈ {0.1, 0.5, 1.0}

| α | γ | Old Acc (%) | New Acc (%) |
|---|---|-------------|-------------|
| 0.1 | 3 | ⏳ | ⏳ |
| 0.1 | 10 | ⏳ | ⏳ |
| 0.5 | 3 | 61.22 | ⏳ |
| 0.5 | 10 | 61.36 | ⏳ |
| 1.0 | 3 | ⏳ | ⏳ |
| 1.0 | 10 | ⏳ | ⏳ |

### (c) Privacy preference: λ_mult ∈ {0.5, 1.0, 2.0}

| λ_mult | γ | Old Acc (%) | New Acc (%) |
|--------|---|-------------|-------------|
| 0.5 | 3 | 60.92 | ⏳ |
| 0.5 | 5 | 60.86 | ⏳ |
| 0.5 | 7 | 61.42 | ⏳ |
| 0.5 | 10 | 61.00 | ⏳ |
| 1.0 | 3 | 61.22 | ⏳ |
| 1.0 | 5 | 61.38 | ⏳ |
| 1.0 | 7 | 61.33 | ⏳ |
| 1.0 | 10 | 61.36 | ⏳ |
| 2.0 | 3 | 60.96 | ⏳ |
| 2.0 | 5 | 61.15 | ⏳ |
| 2.0 | 7 | 61.11 | ⏳ |
| 2.0 | 10 | 61.32 | ⏳ |

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

| Claim | Old value | Predicted new | Action needed |
|-------|-----------|--------------|---------------|
| ε* monotone in γ | ✅ 2.77→3.09 | ✅ still monotone (higher) | Re-run + regen Fig 3 |
| Accuracy ≈61.4% (plateau) | 61.43 ± 0.32 | ~61.5–62.0% (est.) | Re-run + update Table 4 |
| Participation 38→100% | 2.63× | Possible change | Re-run + update Fig 4b |
| ε exposure 1.75× spread | γ=3→10 | Likely similar shape | Re-run + update Fig 4c |
| Payment 6.35× spread | γ=3→10 | Likely larger | Re-run + update Fig 4d |
| PAID-FD > Fair Fixed-ε=1 | +0.11 pp | May increase | Re-run |
| w/o Mixed Loss catastrophic | −45.7 pp | Unchanged | No re-run needed |
| Fig 7 operating point | ε̄*=2.84 | ε̄*≈3.3 | Rerun expG-equivalent |

### ⚠️ Qualitative conclusions at risk
1. **Fig 4b participation ratio 2.63×** — may change if s* shifts dramatically
2. **Fig 7 operating point** — will shift from ε̄*=2.84 to ~3.3; still in plateau ✅
3. **Table 4 PAID-FD advantage** — may shrink or grow vs Fixed-ε baselines

### ✅ Qualitative conclusions that will NOT change
- Catastrophic forgetting without Mixed Loss (w/o CE anchor)
- Non-private baselines (FedAvg, FedGMKD, FedMD) accuracy unchanged
- Fixed-ε baselines accuracy unchanged
- Monotonicity of ε* with γ (Prop. 2)

---

## Rerun Schedule

Each run ≈ 2.5–3 h on RTX 5070 Ti. Total PAID-FD runs needed:

| Section | Runs | Est. time |
|---------|------|-----------|
| VI-B/C: γ ∈ {3,5,7,10} × 3 seeds | 12 | 30–36 h |
| VI-D: method comparison | 4 (PAID-FD only) | 10–12 h |
| VI-F: ablation (4 variants × 3 seeds) | 12 | 30–36 h |
| VI-G (a): scalability N={20,80} × 2γ × 3s | 12 | 30–36 h |
| VI-G (b): α sweep × 2γ × 3s | 12 | 30–36 h |
| VI-G (c): λ_mult × 4γ | 12 | 30–36 h |
| VI-I: CIFAR-10 × 3 seeds | 3 | 8–10 h |
| **Total** | **67 runs** | **~7–8 days** |

Use `scripts/run_v2_experiments.py` with `--phase` flags to run in batches.  
Priority order: VI-B/C → VI-D → VI-F → VI-G → VI-I

---

*Generated: 2026-05-08 | Cubic fix commit: see `git log -1` | Old data: 40c2ee2*
