#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify Route B quick test results."""
import json
import numpy as np

print("=" * 80)
print("  Route B Quick Test Results Verification")
print("=" * 80)

# == Exp 1: 6-Method Comparison ==
with open("results/experiments/routeB_exp1_comparison.json") as f:
    exp1 = json.load(f)

print(f"\nExp 1: Method Comparison (quick: 3 rounds, seed=42)")
print(f"   Seeds: {exp1.get('seeds')}")
print(f"   Methods: {list(exp1['runs'].keys())}")
header = f"{'Method':<20} {'Rounds':>6} {'Final':>10} {'Best':>10} {'Part%':>8} {'AvgEps':>8} {'Time':>8}"
print(f"\n{header}")
print("-" * 80)

for method, runs in exp1["runs"].items():
    for r in runs:
        accs = r["accuracies"]
        parts = r.get("participation_rates", [0])
        eps_list = r.get("avg_eps", [0])
        avg_eps = np.mean([e for e in eps_list if e > 0]) if any(e > 0 for e in eps_list) else 0
        print(f"{method:<20} {len(accs):6d} {r['final_accuracy']:10.4f} {r['best_accuracy']:10.4f} "
              f"{np.mean(parts)*100:8.0f} {avg_eps:8.4f} {r['elapsed_sec']:8.0f}")

print("\nRound-by-round accuracy:")
for method, runs in exp1["runs"].items():
    for r in runs:
        accs = r["accuracies"]
        trend = " -> ".join([f"{a:.4f}" for a in accs])
        mono = all(accs[i] <= accs[i+1] for i in range(len(accs)-1))
        print(f"  {method:<20}: {trend}  {'(monotonic)' if mono else ''}")

# Check energy/extras structure
print("\nData structure check:")
for method, runs in exp1["runs"].items():
    r = runs[0]
    has_energy = len(r.get("energy_history", [])) > 0
    has_extras = len(r.get("extras", [])) > 0
    e0 = r.get("energy_history", [{}])[0] if has_energy else {}
    x0 = r.get("extras", [{}])[0] if has_extras else {}
    print(f"  {method:<20}: energy={has_energy} keys={list(e0.keys())[:4]}, extras={has_extras} keys={list(x0.keys())[:4]}")

# == Exp 6: Ablation ==
print(f"\n{'=' * 80}")
with open("results/experiments/routeB_exp6_ablation.json") as f:
    exp6 = json.load(f)

print(f"\nExp 6: Ablation Study (quick: 3 rounds, seed=42)")
print(f"   Gamma: {exp6.get('gamma')}")
print(f"   Variants: {list(exp6['runs'].keys())}")
print(f"\n{'Variant':<22} {'Rounds':>6} {'Final':>10} {'Best':>10} {'Part%':>8} {'AvgEps':>8} {'Time':>8}")
print("-" * 85)

for variant, runs in exp6["runs"].items():
    for r in runs:
        accs = r["accuracies"]
        parts = r.get("participation_rates", [0])
        eps_list = r.get("avg_eps", [0])
        avg_eps = np.mean([e for e in eps_list if e > 0]) if any(e > 0 for e in eps_list) else 0
        print(f"{variant:<22} {len(accs):6d} {r['final_accuracy']:10.4f} {r['best_accuracy']:10.4f} "
              f"{np.mean(parts)*100:8.0f} {avg_eps:8.4f} {r['elapsed_sec']:8.0f}")

print("\nAblation round-by-round:")
for variant, runs in exp6["runs"].items():
    for r in runs:
        accs = r["accuracies"]
        trend = " -> ".join([f"{a:.4f}" for a in accs])
        print(f"  {variant:<22}: {trend}")

# == Sanity checks ==
print(f"\n{'=' * 80}")
print("SANITY CHECKS:")

# 1. All methods ran 3 rounds?
all_3 = all(len(runs[0]["accuracies"]) == 3 for runs in exp1["runs"].values())
print(f"  [{'OK' if all_3 else 'FAIL'}] All Exp1 methods ran 3 rounds")

all_3_abl = all(len(runs[0]["accuracies"]) == 3 for runs in exp6["runs"].values())
print(f"  [{'OK' if all_3_abl else 'FAIL'}] All Exp6 variants ran 3 rounds")

# 2. PAID-FD accuracy > pre-train baseline (~37%)
paid_acc = exp1["runs"]["PAID-FD"][0]["final_accuracy"]
print(f"  [{'OK' if paid_acc > 0.35 else 'WARN'}] PAID-FD acc ({paid_acc:.4f}) > 0.35 (pre-train baseline)")

# 3. PAID-FD has participation < 1.0 (game mechanism working)
paid_part = np.mean(exp1["runs"]["PAID-FD"][0]["participation_rates"])
print(f"  [{'OK' if paid_part < 1.0 else 'WARN'}] PAID-FD participation ({paid_part:.2f}) < 1.0 (game active)")

# 4. FedAvg no pre-train -> low accuracy
fedavg_acc = exp1["runs"]["FedAvg"][0]["final_accuracy"]
print(f"  [{'OK' if fedavg_acc < 0.15 else 'WARN'}] FedAvg acc ({fedavg_acc:.4f}) low (no pre-train, only 3 rounds)")

# 5. FedMD accuracy near PAID-FD (no noise, same pipeline)
fedmd_acc = exp1["runs"]["FedMD"][0]["final_accuracy"]
print(f"  [{'OK' if abs(fedmd_acc - paid_acc) < 0.15 else 'WARN'}] FedMD acc ({fedmd_acc:.4f}) near PAID-FD ({paid_acc:.4f})")

# 6. Ablation: Full >= Bare-FD (components help)
full_acc = exp6["runs"]["Full (PAID-FD)"][0]["final_accuracy"]
bare_acc = exp6["runs"]["Bare-FD"][0]["final_accuracy"]
print(f"  [{'OK' if full_acc >= bare_acc - 0.02 else 'WARN'}] Full ({full_acc:.4f}) >= Bare-FD ({bare_acc:.4f})")

# 7. Ablation: No-LDP (oracle) >= Full
noldp_acc = exp6["runs"]["No-LDP (oracle)"][0]["final_accuracy"]
print(f"  [{'OK' if noldp_acc >= full_acc - 0.02 else 'WARN'}] No-LDP oracle ({noldp_acc:.4f}) >= Full ({full_acc:.4f})")

# 8. All methods have same gamma=5 in config
all_g5 = True
for v, runs in exp6["runs"].items():
    cfg = runs[0].get("config", {})
    g = cfg.get("gamma", cfg.get("method_config", {}).get("gamma", None))
    if g is not None and g != 5.0:
        all_g5 = False
        print(f"  [FAIL] {v} gamma={g} (expected 5.0)")
print(f"  [{'OK' if all_g5 else 'FAIL'}] All ablation variants use gamma=5.0")

# 9. Expected 7 methods in Exp1, 6 variants in Exp6
n_exp1 = len(exp1["runs"])
n_exp6 = len(exp6["runs"])
print(f"  [{'OK' if n_exp1 == 7 else 'FAIL'}] Exp1 has {n_exp1} methods (expected 7)")
print(f"  [{'OK' if n_exp6 == 6 else 'FAIL'}] Exp6 has {n_exp6} variants (expected 6)")

# 10. Time estimate for full run
total_quick_time = sum(runs[0]["elapsed_sec"] for runs in exp1["runs"].values())
total_quick_abl = sum(runs[0]["elapsed_sec"] for runs in exp6["runs"].values())
# Full = 100 rounds (vs 3) * 3 seeds (vs 1), but pre-train is fixed ~60s
print(f"\nTIME ESTIMATES for full run (100 rounds, 3 seeds):")
for method, runs in exp1["runs"].items():
    t = runs[0]["elapsed_sec"]
    # Rough: pre-train ~60s fixed, rest scales with rounds
    est_per_seed = 60 + (t - 60) * 100 / 3 if t > 60 else t * 100 / 3
    est_total = est_per_seed * 3
    print(f"  {method:<20}: {est_total/3600:.1f}h (3 seeds)")

exp1_est = sum(
    (60 + max(runs[0]["elapsed_sec"] - 60, 0) * 100/3) * 3
    for runs in exp1["runs"].values()
) / 3600
exp6_est = sum(
    (60 + max(runs[0]["elapsed_sec"] - 60, 0) * 100/3) * 3
    for runs in exp6["runs"].values()
) / 3600
print(f"\n  Exp 1 total: ~{exp1_est:.1f}h")
print(f"  Exp 6 total: ~{exp6_est:.1f}h")
print(f"  Combined:    ~{exp1_est + exp6_est:.1f}h")

print(f"\n{'=' * 80}")
print("VERDICT: Check above for any FAIL/WARN items before full run.")
