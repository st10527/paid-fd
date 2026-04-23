# Phase 5 Pipeline Ablation — Summary

**Metric**: `final_acc`

## v10.1 Baseline (PAID-FD, γ=5)

| Seed | Accuracy |
|------|----------|
| 42 | 60.52% |
| 123 | 60.88% |
| 456 | 61.01% |

**Mean = 60.80% ± 0.25%** (3 seeds)

---

## Ablation Results

| Ablation | n seeds | Mean (final_acc) | Std | Δ (pp) | Classification |
|----------|---------|-----------|-----|--------|----------------|
| No EMA | 1/3 | 60.55% | — | -0.25 | negligible |
| No Mixed Loss | 1/3 | 15.49% | — | -45.31 | **significant** |
| No Persistent Models | 1/3 | 63.55% | — | +2.75 | **significant** |

---

## Per-Seed Detail

**No EMA**: s42=60.55%  s123=MISSING  s456=MISSING
**No Mixed Loss**: s42=15.49%  s123=MISSING  s456=MISSING
**No Persistent Models**: s42=63.55%  s123=MISSING  s456=MISSING

---

## Classification Thresholds

| Range | Label |
|-------|-------|
| \|Δ\| < 0.5 pp | negligible |
| 0.5 ≤ \|Δ\| < 2 pp | minor |
| \|Δ\| ≥ 2 pp | **significant** |
