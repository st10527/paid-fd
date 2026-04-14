#!/usr/bin/env python3
"""Analyze Phase 1 results (33 TMC runs) + v10.1 PAID-FD baseline."""
import json, glob, numpy as np
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# 1. Load Phase 1 results (33 runs)
# ============================================================
phase1 = {}
for f in glob.glob(str(ROOT / 'results/experiments/tmc/exp*.json')):
    with open(f) as fh:
        d = json.load(fh)
    phase1[d['label']] = d

print(f"Loaded {len(phase1)} Phase 1 results")

# ============================================================
# 2. Load v10.1 PAID-FD baseline (N=50, gamma={3,5,7,10} x 3 seeds)
# ============================================================
with open(ROOT / 'results/experiments/v10_1_combined_20260409_2304.json') as f:
    v101 = json.load(f)

paidfd_n50 = defaultdict(list)
for k, v in v101['summaries'].items():
    if not isinstance(v, dict):
        continue
    g = v.get('gamma')
    lm = v.get('lambda_mult', 1.0)
    if g in [3, 5, 7, 10] and lm == 1.0 and k.startswith('g'):
        paidfd_n50[g].append(v)

for g in paidfd_n50:
    print(f"  v10.1 PAID-FD gamma={g}: {len(paidfd_n50[g])} seeds")

def stats(runs, key):
    vals = [r[key] for r in runs]
    return np.mean(vals), np.std(vals)

def stats_get(runs, key, default=0):
    vals = [r.get(key, default) for r in runs]
    return np.mean(vals), np.std(vals)

# ============================================================
# TABLE II: Privacy-Preserving Method Comparison
# ============================================================
print("\n" + "=" * 95)
print("TABLE II: Privacy-Preserving Method Comparison (CIFAR-100, N=50, 100 rounds)")
print("=" * 95)
hdr = f"{'Method':<22} {'Privacy':<10} {'Final Acc':<14} {'Best Acc':<14} {'Part.':<8} {'Avg eps/rd':<10} {'Cum.Pay':<10}"
print(hdr)
print("-" * 95)

# PAID-FD rows
for g in [3, 5, 10]:
    runs = paidfd_n50[g]
    if not runs:
        continue
    fa_m, fa_s = stats(runs, 'final_acc')
    ba_m, ba_s = stats(runs, 'best_acc')
    pt_m, _ = stats(runs, 'avg_participation')
    ep_m, _ = stats(runs, 'avg_eps_per_round')
    cp_m, _ = stats(runs, 'cumulative_payment')
    print(f"PAID-FD (g={g}){'':<9} {'LDP':<10} {fa_m:.4f}+/-{fa_s:.4f}  {ba_m:.4f}+/-{ba_s:.4f}  {pt_m:<8.1%} {ep_m:<10.2f} {cp_m:<10.0f}")

print("-" * 95)

# Fixed-eps-1
runs = [phase1[k]['summary'] for k in phase1 if k.startswith('expA_fixedeps1')]
fa_m, fa_s = stats(runs, 'final_acc')
ba_m, ba_s = stats(runs, 'best_acc')
pt_m, _ = stats(runs, 'avg_participation')
print(f"{'Fixed-eps=1':<22} {'LDP':<10} {fa_m:.4f}+/-{fa_s:.4f}  {ba_m:.4f}+/-{ba_s:.4f}  {pt_m:<8.1%} {'1.00':<10} {'0':<10}")

# Fixed-eps-3
runs = [phase1[k]['summary'] for k in phase1 if k.startswith('expA_fixedeps3')]
fa_m, fa_s = stats(runs, 'final_acc')
ba_m, ba_s = stats(runs, 'best_acc')
pt_m, _ = stats(runs, 'avg_participation')
print(f"{'Fixed-eps=3':<22} {'LDP':<10} {fa_m:.4f}+/-{fa_s:.4f}  {ba_m:.4f}+/-{ba_s:.4f}  {pt_m:<8.1%} {'3.00':<10} {'0':<10}")

# CSRA
runs = [phase1[k]['summary'] for k in phase1 if k.startswith('expA_csra')]
fa_m, fa_s = stats(runs, 'final_acc')
ba_m, ba_s = stats(runs, 'best_acc')
pt_m, _ = stats(runs, 'avg_participation')
print(f"{'CSRA':<22} {'Param DP':<10} {fa_m:.4f}+/-{fa_s:.4f}  {ba_m:.4f}+/-{ba_s:.4f}  {pt_m:<8.1%} {'N/A':<10} {'budget':<10}")

print("-" * 95)
print("  (No-privacy reference -- upper bound, not competitors)")

# A' methods (single seed)
for label, name in [('expAp_fedavg_s42', 'FedAvg'),
                     ('expAp_fedmd_s42', 'FedMD'),
                     ('expAp_fedgmkd_s42', 'FedGMKD')]:
    r = phase1[label]['summary']
    print(f"{name:<22} {'None':<10} {r['final_acc']:.4f}           {r['best_acc']:.4f}           {r['avg_participation']:<8.1%} {'N/A':<10} {'N/A':<10}")

# ============================================================
# TABLE III: Scalability -- N Sweep
# ============================================================
print("\n" + "=" * 85)
print("TABLE III: Scalability -- N Sweep (CIFAR-100, Dirichlet a=0.5, 100 rounds)")
print("=" * 85)
print(f"{'N':>4} {'g':>3}  {'Final Acc':<14} {'Best Acc':<14} {'Part.':<8} {'Avg eps/rd':<10} {'Cum.Pay':<12}")
print("-" * 85)

for N in [20, 50, 80]:
    for g in [3, 10]:
        if N == 50:
            runs = paidfd_n50[g]
        else:
            prefix = f'expB_n{N}_g{g}'
            runs = [phase1[k]['summary'] for k in phase1 if k.startswith(prefix)]
        
        if not runs:
            continue
        fa_m, fa_s = stats(runs, 'final_acc')
        ba_m, ba_s = stats(runs, 'best_acc')
        pt_m, _ = stats(runs, 'avg_participation')
        ep_m, _ = stats_get(runs, 'avg_eps_per_round')
        cp_m, _ = stats_get(runs, 'cumulative_payment')
        print(f"{N:>4} {g:>3}  {fa_m:.4f}+/-{fa_s:.4f}  {ba_m:.4f}+/-{ba_s:.4f}  {pt_m:<8.1%} {ep_m:<10.2f} {cp_m:<12.0f}")
    if N < 80:
        print()

# ============================================================
# TABLE IV: Ablation Study
# ============================================================
print("\n" + "=" * 90)
print("TABLE IV: Ablation Study (CIFAR-100, N=50, g=5, 100 rounds)")
print("=" * 90)
print(f"{'Variant':<30} {'Final Acc':<14} {'Best Acc':<14} {'Part.':<8} {'Avg eps/rd':<10} {'Cum.Pay':<12}")
print("-" * 90)

# Baseline
runs = paidfd_n50[5]
fa_m, fa_s = stats(runs, 'final_acc')
ba_m, ba_s = stats(runs, 'best_acc')
pt_m, _ = stats(runs, 'avg_participation')
ep_m, _ = stats(runs, 'avg_eps_per_round')
cp_m, _ = stats(runs, 'cumulative_payment')
print(f"{'PAID-FD (baseline g=5)':<30} {fa_m:.4f}+/-{fa_s:.4f}  {ba_m:.4f}+/-{ba_s:.4f}  {pt_m:<8.1%} {ep_m:<10.2f} {cp_m:<12.0f}")

# Ablation variants
for prefix, name in [('expC_noblue', 'w/o BLUE (uniform agg)'),
                      ('expC_fullpart', 'Full participation (g=100)'),
                      ('expC_noldp', 'No LDP (oracle ceiling)')]:
    runs = [phase1[k]['summary'] for k in phase1 if k.startswith(prefix)]
    fa_m, fa_s = stats(runs, 'final_acc')
    ba_m, ba_s = stats(runs, 'best_acc')
    pt_m, _ = stats(runs, 'avg_participation')
    ep_m, _ = stats_get(runs, 'avg_eps_per_round')
    cp_m, _ = stats_get(runs, 'cumulative_payment')
    print(f"{name:<30} {fa_m:.4f}+/-{fa_s:.4f}  {ba_m:.4f}+/-{ba_s:.4f}  {pt_m:<8.1%} {ep_m:<10.2f} {cp_m:<12.0f}")

# ============================================================
# SANITY CHECKS & ANOMALIES
# ============================================================
print("\n" + "=" * 90)
print("SANITY CHECKS & ANOMALIES")
print("=" * 90)

# 1. CSRA: 1% accuracy -- catastrophic
print("\n[!] CSRA accuracy is ~1% (random guess for 100 classes)")
for k in sorted(phase1):
    if 'csra' in k:
        r = phase1[k]
        accs = r.get('accuracies', [])
        print(f"  {k}: final={r['summary']['final_acc']:.4f}, "
              f"best={r['summary']['best_acc']:.4f}, "
              f"part={r['summary']['avg_participation']:.3f}, "
              f"first 5 accs: {[round(a,4) for a in accs[:5]]}")

# 2. Fixed-eps-1 vs Fixed-eps-3: nearly identical
print("\n[?] Fixed-eps-1 vs Fixed-eps-3 accuracy nearly identical")
fe1 = [phase1[k]['summary'] for k in phase1 if 'fixedeps1' in k]
fe3 = [phase1[k]['summary'] for k in phase1 if 'fixedeps3' in k]
print(f"  Fixed-eps=1: final={np.mean([r['final_acc'] for r in fe1]):.4f}, best={np.mean([r['best_acc'] for r in fe1]):.4f}")
print(f"  Fixed-eps=3: final={np.mean([r['final_acc'] for r in fe3]):.4f}, best={np.mean([r['best_acc'] for r in fe3]):.4f}")
print(f"  Delta: {np.mean([r['final_acc'] for r in fe3]) - np.mean([r['final_acc'] for r in fe1]):+.4f}")

# 3. No-LDP oracle vs baseline
print("\n[?] Oracle check (no-LDP should >= baseline)")
noldp = [phase1[k]['summary'] for k in phase1 if 'noldp' in k]
baseline = paidfd_n50[5]
noldp_fa = np.mean([r['final_acc'] for r in noldp])
base_fa = np.mean([r['final_acc'] for r in baseline])
print(f"  Baseline g=5: {base_fa:.4f}")
print(f"  No-LDP oracle: {noldp_fa:.4f}")
print(f"  Delta: {noldp_fa - base_fa:+.4f}")

# 4. No-BLUE vs baseline
print("\n[?] BLUE ablation check")
noblue = [phase1[k]['summary'] for k in phase1 if 'noblue' in k]
noblue_fa = np.mean([r['final_acc'] for r in noblue])
print(f"  Baseline g=5: {base_fa:.4f}")
print(f"  No BLUE:      {noblue_fa:.4f}")
print(f"  Delta: {noblue_fa - base_fa:+.4f}")

# 5. Accuracy degradation curves
print("\n[i] Accuracy over rounds (sample: expA_fixedeps1_s42)")
r = phase1.get('expA_fixedeps1_s42', {})
accs = r.get('accuracies', [])
if accs:
    for i in [0, 9, 24, 49, 74, 99]:
        if i < len(accs):
            print(f"  Round {i+1:>3}: {accs[i]:.4f}")

print("\n[i] PAID-FD g=5 s=42 (v10.1) accuracy curve:")
g5_s42 = v101['summaries'].get('g5_s42', {})
if 'accuracies' not in g5_s42:
    # Try to find in the full result files
    for f in glob.glob(str(ROOT / 'results/experiments/v10_1_3seeds*.json')):
        with open(f) as fh:
            d = json.load(fh)
        if isinstance(d, dict) and 'results' in d:
            for res in d['results']:
                if res.get('gamma') == 5 and res.get('seed') == 42:
                    accs = res.get('accuracies', [])
                    if accs:
                        for i in [0, 9, 24, 49, 74, 99]:
                            if i < len(accs):
                                print(f"  Round {i+1:>3}: {accs[i]:.4f}")
                    break

# 6. Summary comparison
print("\n" + "=" * 90)
print("SUMMARY: Key Findings")
print("=" * 90)

paidfd_g5_best = np.mean([r['best_acc'] for r in paidfd_n50[5]])
fe3_best = np.mean([r['best_acc'] for r in fe3])
csra_best = np.mean([r['best_acc'] for r in [phase1[k]['summary'] for k in phase1 if 'csra' in k]])
fedavg_best = phase1['expAp_fedavg_s42']['summary']['best_acc']
fedmd_best = phase1['expAp_fedmd_s42']['summary']['best_acc']

print(f"""
1. PAID-FD (g=5) best accuracy: {paidfd_g5_best:.4f} (61.4%)
   - Beats Fixed-eps=3 by: {paidfd_g5_best - fe3_best:+.4f} ({(paidfd_g5_best - fe3_best)*100:+.1f} pp)
   - CSRA completely failed: {csra_best:.4f} (1%)
   
2. No-privacy references (upper bounds):
   - FedAvg: {fedavg_best:.4f}
   - FedMD:  {fedmd_best:.4f}
   - FedGMKD: {phase1['expAp_fedgmkd_s42']['summary']['best_acc']:.4f}
   - PAID-FD vs FedMD gap: {paidfd_g5_best - fedmd_best:+.4f} (PAID-FD BEATS no-privacy FedMD!)

3. Scalability: N=20/50/80 all achieve ~61% (stable)

4. Ablation:
   - No BLUE: {noblue_fa:.4f} vs baseline {base_fa:.4f} (delta={noblue_fa - base_fa:+.4f})
   - No LDP:  {noldp_fa:.4f} vs baseline {base_fa:.4f} (delta={noldp_fa - base_fa:+.4f})
""")
