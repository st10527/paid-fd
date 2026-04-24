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
| No EMA | 3/3 | 60.89% | 0.30% | +0.09 | negligible |
| No Mixed Loss | 3/3 | 15.75% | 0.73% | -45.05 | **significant** |
| No Persistent Models | 3/3 | 63.43% | 0.13% | +2.62 | **significant** |

---

## Per-Seed Detail

**No EMA**: s42=60.55%  s123=61.04%  s456=61.09%
**No Mixed Loss**: s42=15.49%  s123=15.19%  s456=16.57%
**No Persistent Models**: s42=63.55%  s123=63.43%  s456=63.30%

---

## Classification Thresholds

| Range | Label |
|-------|-------|
| \|Δ\| < 0.5 pp | negligible |
| 0.5 ≤ \|Δ\| < 2 pp | minor |
| \|Δ\| ≥ 2 pp | **significant** |
