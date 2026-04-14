#!/usr/bin/env python3
"""Diagnostic analysis for Phase 1 warnings."""
import json, glob, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load all Phase 1 results
phase1 = {}
for f in glob.glob(str(ROOT / 'results/experiments/tmc/exp*.json')):
    with open(f) as fh:
        d = json.load(fh)
    phase1[d['label']] = d

# ============================================================
# WARNING 2: Fixed-eps=1 vs Fixed-eps=3 accuracy trajectory
# ============================================================
print("=" * 80)
print("WARNING 2: Fixed-eps=1 vs Fixed-eps=3 Accuracy Trajectories")
print("=" * 80)

print("\nRound-by-round comparison (seed=42):")
print(f"{'Round':>5}  {'Fixed-eps=1':>12}  {'Fixed-eps=3':>12}  {'Delta':>8}")
print("-" * 45)

fe1_42 = phase1['expA_fixedeps1_s42']['accuracies']
fe3_42 = phase1['expA_fixedeps3_s42']['accuracies']

for r in [0, 4, 9, 14, 19, 24, 34, 49, 74, 99]:
    if r < len(fe1_42) and r < len(fe3_42):
        d = fe3_42[r] - fe1_42[r]
        print(f"{r+1:>5}  {fe1_42[r]:>12.4f}  {fe3_42[r]:>12.4f}  {d:>+8.4f}")

# Check all 3 seeds
print("\n\nAll seeds - Full trajectory stats:")
for method_prefix, name in [('expA_fixedeps1', 'Fixed-eps=1'), ('expA_fixedeps3', 'Fixed-eps=3')]:
    runs = {k: phase1[k] for k in phase1 if k.startswith(method_prefix)}
    for label, r in sorted(runs.items()):
        accs = r['accuracies']
        # Find peak and where decline starts
        peak_val = max(accs)
        peak_round = accs.index(peak_val) + 1
        # Check if monotonic decline after peak
        post_peak = accs[accs.index(peak_val):]
        decline = peak_val - accs[-1]
        print(f"  {label}: peak={peak_val:.4f} at R{peak_round}, "
              f"final={accs[-1]:.4f}, decline={decline:+.4f} "
              f"({len(post_peak)} rounds after peak)")

# Correlation between the two
print("\n\nCorrelation test (are curves identical?):")
for seed in [42, 123, 456]:
    a1 = np.array(phase1[f'expA_fixedeps1_s{seed}']['accuracies'])
    a3 = np.array(phase1[f'expA_fixedeps3_s{seed}']['accuracies'])
    corr = np.corrcoef(a1, a3)[0, 1]
    max_diff = np.max(np.abs(a1 - a3))
    mean_diff = np.mean(a3 - a1)
    print(f"  seed={seed}: corr={corr:.6f}, max_diff={max_diff:.4f}, mean_diff(e3-e1)={mean_diff:+.4f}")

# ============================================================
# WARNING 1: CSRA Diagnosis
# ============================================================
print("\n\n" + "=" * 80)
print("WARNING 1: CSRA Diagnosis")
print("=" * 80)

for k in sorted(phase1):
    if 'csra' not in k:
        continue
    r = phase1[k]
    accs = r['accuracies']
    s = r['summary']
    
    print(f"\n{k}:")
    print(f"  Accuracies R1-R10: {[round(a, 4) for a in accs[:10]]}")
    print(f"  All same? {len(set([round(a, 4) for a in accs])) == 1}")
    print(f"  Participation: {s['avg_participation']:.3f}")
    print(f"  Avg price: {s.get('avg_price', 'N/A')}")
    
    # Check extras for any clues
    extras = r.get('extras', [])
    if extras and isinstance(extras[0], dict):
        print(f"  Round 1 extras keys: {list(extras[0].keys())}")
        for key in ['price', 'avg_eps', 'avg_s', 'cumulative_payment']:
            if key in extras[0]:
                print(f"    {key} R1: {extras[0][key]}")
        if len(extras) > 1:
            for key in ['price', 'avg_eps', 'avg_s', 'cumulative_payment']:
                if key in extras[-1]:
                    print(f"    {key} R100: {extras[-1][key]}")

# ============================================================
# WARNING 3: LDP Noise Analysis
# ============================================================
print("\n\n" + "=" * 80)
print("WARNING 3: LDP Noise Effectiveness")
print("=" * 80)

# Compare PAID-FD baseline (with LDP) vs No-LDP oracle
print("\nPAID-FD (g=5) with LDP vs No-LDP (oracle) - per-seed:")
# Load v10.1 data
with open(ROOT / 'results/experiments/v10_1_combined_20260409_2304.json') as f:
    v101 = json.load(f)

for seed in [42, 123, 456]:
    # v10.1 baseline
    v_key = f'g5_s{seed}'
    v_data = v101['summaries'].get(v_key, {})
    if isinstance(v_data, dict):
        bl_final = v_data.get('final_acc', '?')
        bl_best = v_data.get('best_acc', '?')
        bl_eps = v_data.get('avg_eps_per_round', '?')
    else:
        bl_final = bl_best = bl_eps = '?'
    
    # No-LDP
    noldp_key = f'expC_noldp_s{seed}'
    if noldp_key in phase1:
        nl = phase1[noldp_key]['summary']
        nl_final = nl['final_acc']
        nl_best = nl['best_acc']
        nl_eps = nl.get('avg_eps_per_round', 0)
    else:
        nl_final = nl_best = nl_eps = '?'
    
    print(f"  seed={seed}: LDP final={bl_final}, no-LDP final={nl_final}, "
          f"delta={float(nl_final) - float(bl_final):+.4f}" if isinstance(bl_final, float) and isinstance(nl_final, float) else f"  seed={seed}: data incomplete")

# Check the actual eps values in PAID-FD rounds
print("\nPAID-FD avg eps per round across experiments:")
for k in sorted(phase1):
    if 'expB' in k or 'expC' in k:
        s = phase1[k]['summary']
        eps = s.get('avg_eps_per_round', 0)
        if eps > 0:
            print(f"  {k}: avg_eps/rd={eps:.3f}")

# ============================================================
# WARNING 4: BLUE Weights Analysis  
# ============================================================
print("\n\n" + "=" * 80)
print("WARNING 4: BLUE vs Uniform (No-BLUE) Analysis")
print("=" * 80)

print("\nPer-seed comparison (g=5):")
for seed in [42, 123, 456]:
    # Baseline (with BLUE)
    v_key = f'g5_s{seed}'
    v_data = v101['summaries'].get(v_key, {})
    bl_final = v_data.get('final_acc', '?') if isinstance(v_data, dict) else '?'
    bl_best = v_data.get('best_acc', '?') if isinstance(v_data, dict) else '?'
    
    # No-BLUE
    nb_key = f'expC_noblue_s{seed}'
    if nb_key in phase1:
        nb = phase1[nb_key]['summary']
        nb_final = nb['final_acc']
        nb_best = nb['best_acc']
        print(f"  seed={seed}: BLUE final={bl_final:.4f} best={bl_best:.4f} | "
              f"No-BLUE final={nb_final:.4f} best={nb_best:.4f} | "
              f"delta_final={nb_final - bl_final:+.4f}")

# Check eps variance (if eps are all similar, BLUE = uniform)
print("\nEpsilon variance check (if low, BLUE degenerates to uniform):")
for k in sorted(phase1):
    if 'noblue' in k:
        extras = phase1[k].get('extras', [])
        if extras and isinstance(extras[0], dict) and 'avg_eps' in extras[0]:
            eps_vals = [e.get('avg_eps', 0) for e in extras if isinstance(e, dict)]
            if eps_vals:
                print(f"  {k}: eps range [{min(eps_vals):.3f}, {max(eps_vals):.3f}], "
                      f"std={np.std(eps_vals):.3f}")

# Full-participation analysis
print("\nFull participation (g=100) analysis:")
for seed in [42, 123, 456]:
    fp_key = f'expC_fullpart_s{seed}'
    if fp_key in phase1:
        s = phase1[fp_key]['summary']
        print(f"  seed={seed}: final={s['final_acc']:.4f}, best={s['best_acc']:.4f}, "
              f"part={s['avg_participation']:.3f}, eps/rd={s.get('avg_eps_per_round', 0):.2f}, "
              f"cum_pay={s.get('cumulative_payment', 0):.0f}")

# ============================================================
# Additional: Accuracy curves for key methods
# ============================================================
print("\n\n" + "=" * 80)
print("ACCURACY CURVES: Key methods at round milestones")
print("=" * 80)

methods_to_plot = {
    'PAID-FD g=5 s42': None,  # from v10.1, need to find
    'Fixed-eps=1 s42': 'expA_fixedeps1_s42',
    'Fixed-eps=3 s42': 'expA_fixedeps3_s42',
    'CSRA s42': 'expA_csra_s42',
    'FedAvg s42': 'expAp_fedavg_s42',
    'FedMD s42': 'expAp_fedmd_s42',
    'FedGMKD s42': 'expAp_fedgmkd_s42',
    'No-BLUE s42': 'expC_noblue_s42',
    'No-LDP s42': 'expC_noldp_s42',
}

rounds = [1, 5, 10, 20, 30, 50, 70, 100]
print(f"\n{'Method':<22}", end="")
for r in rounds:
    print(f"  R{r:>3}", end="")
print()
print("-" * (22 + 7 * len(rounds)))

for name, label in methods_to_plot.items():
    if label is None:
        continue
    if label not in phase1:
        continue
    accs = phase1[label]['accuracies']
    print(f"{name:<22}", end="")
    for r in rounds:
        idx = r - 1
        if idx < len(accs):
            print(f"  {accs[idx]:.3f}", end="")
        else:
            print(f"  {'N/A':>5}", end="")
    print()
