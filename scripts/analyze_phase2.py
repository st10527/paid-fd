#!/usr/bin/env python3
"""Analyze Phase 2 (CIFAR-10) results."""
import json, glob, os, sys
import numpy as np

base = os.path.join(os.path.dirname(__file__), '..', 'results', 'experiments', 'tmc')

# Collect all Phase 2 results
phase2 = {}
for f in sorted(glob.glob(os.path.join(base, 'expD_*.json'))):
    with open(f) as fh:
        d = json.load(fh)
    method = d['method']
    seed = d['seed']
    s = d['summary']
    accs = d['accuracies']

    if method not in phase2:
        phase2[method] = []

    phase2[method].append({
        'seed': seed,
        'final_acc': s['final_acc'],
        'best_acc': s['best_acc'],
        'avg_part': s['avg_participation'],
        'avg_price': s['avg_price'],
        'avg_eps': s['avg_eps_per_round'],
        'elapsed': d.get('elapsed_sec', 0),
        'accs': accs,
    })

print("=" * 80)
print("PHASE 2 RESULTS: CIFAR-10 Cross-Dataset Validation (N=50, gamma=5, 100 rounds)")
print("=" * 80)
print()

for method in ['PAID-FD', 'Fixed-eps-3', 'CSRA']:
    if method not in phase2:
        print(f"{method}: NO DATA")
        continue
    runs = phase2[method]
    finals = [r['final_acc'] for r in runs]
    bests = [r['best_acc'] for r in runs]

    print(f"--- {method} ({len(runs)} seeds) ---")
    for r in runs:
        print(f"  Seed {r['seed']}: final={r['final_acc']:.4f}  best={r['best_acc']:.4f}  "
              f"part={r['avg_part']:.1%}  eps/rd={r['avg_eps']:.2f}  "
              f"price={r['avg_price']:.1f}  time={r['elapsed']/3600:.1f}h")

    print(f"  MEAN: final={np.mean(finals):.4f} +/- {np.std(finals):.4f}  "
          f"best={np.mean(bests):.4f} +/- {np.std(bests):.4f}")
    print()

# Accuracy gap
if 'PAID-FD' in phase2 and 'Fixed-eps-3' in phase2:
    paid_best = np.mean([r['best_acc'] for r in phase2['PAID-FD']])
    fixed_best = np.mean([r['best_acc'] for r in phase2['Fixed-eps-3']])
    paid_final = np.mean([r['final_acc'] for r in phase2['PAID-FD']])
    fixed_final = np.mean([r['final_acc'] for r in phase2['Fixed-eps-3']])
    print(f"PAID-FD vs Fixed-eps-3 gap: +{(paid_best - fixed_best)*100:.1f}pp (best), "
          f"+{(paid_final - fixed_final)*100:.1f}pp (final)")

if 'CSRA' in phase2:
    csra_best = np.mean([r['best_acc'] for r in phase2['CSRA']])
    print(f"CSRA best accuracy: {csra_best:.4f} (10 classes => random=10%)")
    print()

# ── Convergence curves (every 10 rounds) ──
print("=" * 80)
print("CONVERGENCE SNAPSHOT (every 10 rounds)")
print("=" * 80)
print(f"{'Round':<8}", end="")
for method in ['PAID-FD', 'Fixed-eps-3', 'CSRA']:
    if method in phase2:
        print(f"{method:<18}", end="")
print()
print("-" * 62)

for r_idx in list(range(0, 100, 10)) + [99]:
    print(f"R{r_idx:<7}", end="")
    for method in ['PAID-FD', 'Fixed-eps-3', 'CSRA']:
        if method in phase2:
            vals = [run['accs'][r_idx] for run in phase2[method] if r_idx < len(run['accs'])]
            if vals:
                print(f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}  ", end="")
            else:
                print(f"{'N/A':<18}", end="")
    print()

# ── Cross-dataset comparison table (CIFAR-10 vs CIFAR-100) ──
print()
print("=" * 80)
print("CROSS-DATASET COMPARISON: CIFAR-100 (Phase 1) vs CIFAR-10 (Phase 2)")
print("=" * 80)

# Load Phase 1 PAID-FD baseline (gamma=5)
phase1_paid = {}
for f in sorted(glob.glob(os.path.join(base, 'expA_*.json')) +
                glob.glob(os.path.join(base, 'expC_*.json'))):
    with open(f) as fh:
        d = json.load(fh)
    method = d['method']
    if method == 'PAID-FD' and d.get('dataset', 'cifar100') == 'cifar100':
        s = d['summary']
        if method not in phase1_paid:
            phase1_paid[method] = []
        phase1_paid[method].append(s)

# Also grab Phase 1 Fixed-eps-3
phase1_fixed = {}
for f in sorted(glob.glob(os.path.join(base, 'expA_fixedeps3_*.json'))):
    with open(f) as fh:
        d = json.load(fh)
    s = d['summary']
    method = d['method']
    if method not in phase1_fixed:
        phase1_fixed[method] = []
    phase1_fixed[method].append(s)

print(f"\n{'Method':<18} {'CIFAR-100 (best)':<20} {'CIFAR-10 (best)':<20} {'Generalizes?':<15}")
print("-" * 73)

if 'PAID-FD' in phase2:
    c10 = np.mean([r['best_acc'] for r in phase2['PAID-FD']])
    # Phase 1 PAID-FD gamma=5 baseline
    c100_str = "~61.4%"
    print(f"{'PAID-FD':<18} {c100_str:<20} {c10:.1%}{'':<13} {'YES' if c10 > 0.5 else 'CHECK'}")

if 'Fixed-eps-3' in phase2:
    c10 = np.mean([r['best_acc'] for r in phase2['Fixed-eps-3']])
    c100_str = "~41.9%"
    print(f"{'Fixed-eps-3':<18} {c100_str:<20} {c10:.1%}{'':<13} {'YES' if c10 > 0.3 else 'CHECK'}")

if 'CSRA' in phase2:
    c10 = np.mean([r['best_acc'] for r in phase2['CSRA']])
    c100_str = "~1.0%"
    print(f"{'CSRA':<18} {c100_str:<20} {c10:.1%}{'':<13} {'Consistent' if c10 < 0.15 else 'CHECK'}")

print("\n(CIFAR-100 values from Phase 1 analysis)")
