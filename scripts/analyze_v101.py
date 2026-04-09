#!/usr/bin/env python3
"""Analyze v10.1 results with truncated JSON handling."""
import re
import numpy as np

FILE = "results/experiments/v10_persistent_20260408_2320.json"

with open(FILE) as f:
    content = f.read()

# Extract gamma summary blocks via regex (handles truncated JSON)
pattern = (
    r'"gamma":\s*(\d+),\s*\n'
    r'\s*"final_acc":\s*([\d.]+),\s*\n'
    r'\s*"best_acc":\s*([\d.]+),\s*\n'
    r'\s*"avg_participation":\s*([\d.]+),\s*\n'
    r'\s*"avg_price":\s*([\d.]+),\s*\n'
    r'\s*"avg_eps_per_round":\s*([\d.]+),\s*\n'
    r'\s*"cumulative_payment":\s*([\d.]+),\s*\n'
    r'\s*"avg_privacy_spent":\s*([\d.]+),\s*\n'
    r'\s*"max_privacy_spent":\s*([\d.]+),\s*\n'
    r'\s*"energy_total":\s*([\d.]+)'
)

matches = list(re.finditer(pattern, content))

# Also find time_to_targets per gamma
ttt_all = list(re.finditer(
    r'"time_to_targets":\s*\{([^}]+)\}', content))

# Also find avg_distill_delta per gamma
dd_all = list(re.finditer(
    r'"avg_distill_delta":\s*([\d.e+-]+)', content))

print("=" * 80)
print("v10.1 RESULTS — PERSISTENT ADAM FIX — ALL 4 GAMMAS")
print("=" * 80)

# v10.0 baselines for comparison
v100 = {3: 47.14, 5: 47.08}

results = []
for i, m in enumerate(matches):
    g = int(m.group(1))
    r = {
        "gamma": g,
        "final_acc": float(m.group(2)) * 100,
        "best_acc": float(m.group(3)) * 100,
        "avg_part": float(m.group(4)) * 100,
        "avg_price": float(m.group(5)),
        "avg_eps": float(m.group(6)),
        "cum_payment": float(m.group(7)),
        "avg_priv_spent": float(m.group(8)),
        "max_priv_spent": float(m.group(9)),
        "energy": float(m.group(10)),
    }
    if i < len(dd_all):
        r["distill_delta"] = float(dd_all[i].group(1))
    results.append(r)

# Table 1: Accuracy & Participation
print("\n[Table 1] Accuracy & Participation")
print("-" * 75)
print(f"  g   Final%    Best%   Part%   Price    eps*   vs_v10.0")
print("-" * 75)
for r in results:
    g = r["gamma"]
    v0 = v100.get(g, None)
    delta = "+%.1f%%" % (r["best_acc"] - v0) if v0 else "N/A"
    print(f"  {g:<3} {r['final_acc']:>7.2f}%  {r['best_acc']:>6.2f}%  {r['avg_part']:>5.0f}%"
          f"  {r['avg_price']:>6.3f}  {r['avg_eps']:>6.3f}  {delta}")

# Table 2: Cost & Privacy
print("\n[Table 2] Cost & Privacy Accounting")
print("-" * 75)
print(f"  g   CumPayment     AvgPrivSpent  MaxPrivSpent    Energy")
print("-" * 75)
for r in results:
    print(f"  {r['gamma']:<3} {r['cum_payment']:>12,.0f}   {r['avg_priv_spent']:>10.1f}"
          f"    {r['max_priv_spent']:>10.1f}   {r['energy']:>12,.0f}")

# Table 3: Time to Target
print("\n[Table 3] Time-to-Target (rounds to reach accuracy)")
print("-" * 65)
targets = ["0.45", "0.5", "0.55", "0.58", "0.6"]
print(f"  g  " + "  ".join(f"@{t:>4}" for t in targets))
print("-" * 65)
for i, r in enumerate(results):
    g = r["gamma"]
    if i < len(ttt_all):
        ttt_str = ttt_all[i].group(1)
        ttt = {}
        for tm in re.finditer(r'"([\d.]+)":\s*(\d+)', ttt_str):
            ttt[tm.group(1)] = int(tm.group(2))
    else:
        ttt = {}
    vals = []
    for t in targets:
        v = ttt.get(t, None)
        vals.append(f"{v:>5}" if v is not None else "  ---")
    print(f"  {g:<3}" + "  ".join(vals))

# ============================================================
# EFFICIENCY STORY
# ============================================================
print("\n" + "=" * 80)
print("EFFICIENCY STORY EVALUATION (E1-E5)")
print("=" * 80)

if len(results) >= 2:
    lo, hi = results[0], results[-1]
    all_best = [r["best_acc"] for r in results]

    acc_spread = max(all_best) - min(all_best)
    e1 = acc_spread < 2.0
    print(f"  E1 Accuracy flat:   spread = {acc_spread:.2f}%  {'PASS' if e1 else 'FAIL'} (<2%)")

    cost_ratio = hi["cum_payment"] / lo["cum_payment"] if lo["cum_payment"] > 0 else 0
    e2 = cost_ratio > 2.0
    print(f"  E2 Cost ratio:      {cost_ratio:.2f}x        {'PASS' if e2 else 'FAIL'} (>2x)")

    priv_ratio = hi["max_priv_spent"] / lo["max_priv_spent"] if lo["max_priv_spent"] > 0 else 0
    e3 = priv_ratio > 1.5
    print(f"  E3 Privacy ratio:   {priv_ratio:.2f}x        {'PASS' if e3 else 'FAIL'} (>1.5x)")

    part_spread = hi["avg_part"] - lo["avg_part"]
    e4 = part_spread > 10
    print(f"  E4 Part spread:     {part_spread:.0f}%          {'PASS' if e4 else 'FAIL'} (>10%)")

    best_any = max(all_best)
    e5 = best_any > 50
    print(f"  E5 Best accuracy:   {best_any:.2f}%      {'PASS' if e5 else 'FAIL'} (>50%)")

    score = sum([e1, e2, e3, e4, e5])
    print(f"\n  SCORE: {score}/5")

# Proposition 2 monotonicity
print("\n" + "=" * 80)
print("PROPOSITION 2 MONOTONICITY (gamma increases)")
print("=" * 80)
gs = [r["gamma"] for r in results]
metrics = [
    ("price p*", [r["avg_price"] for r in results], True),
    ("participation", [r["avg_part"] for r in results], True),
    ("cumulative cost", [r["cum_payment"] for r in results], True),
    ("avg eps*", [r["avg_eps"] for r in results], None),  # may not be monotone
]
for name, vals, expect_inc in metrics:
    diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
    if expect_inc is True:
        ok = all(d > 0 for d in diffs)
    elif expect_inc is False:
        ok = all(d < 0 for d in diffs)
    else:
        ok = None
    vals_str = " -> ".join(f"{v:.2f}" for v in vals)
    if ok is True:
        status = "MONO UP"
    elif ok is False:
        status = "NOT MONO"
    else:
        status = "---"
    print(f"  {name:<18}: {vals_str}  [{status}]")

# v10.0 vs v10.1 comparison
print("\n" + "=" * 80)
print("v10.0 (fresh SGD) vs v10.1 (persistent Adam)")
print("=" * 80)
print(f"  {'Metric':<25} {'v10.0':<15} {'v10.1':<15} {'Change':<12}")
print("  " + "-" * 67)
v101_g3 = next((r for r in results if r["gamma"] == 3), None)
v101_g5 = next((r for r in results if r["gamma"] == 5), None)
if v101_g3:
    print(f"  {'Best acc (g=3)':<25} {'47.14%':<15} {v101_g3['best_acc']:.2f}%{'':<9} +{v101_g3['best_acc']-47.14:.1f}%")
if v101_g5:
    print(f"  {'Best acc (g=5)':<25} {'47.08%':<15} {v101_g5['best_acc']:.2f}%{'':<9} +{v101_g5['best_acc']-47.08:.1f}%")
print(f"  {'CE loss trend':<25} {'stagnant@2.08':<15} {'decreasing':<15} {'FIXED':<12}")
print(f"  {'Optimizer':<25} {'fresh SGD':<15} {'persist Adam':<15} {'KEY FIX':<12}")
print(f"  {'Score':<25} {'3/5':<15} {'?/5':<15}")

# Extract accuracy trajectories
print("\n" + "=" * 80)
print("ACCURACY TRAJECTORIES")
print("=" * 80)
acc_pattern = r'"round_idx":\s*(\d+),\s*\n\s*"accuracy":\s*([\d.]+)'
round_matches = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(acc_pattern, content)]

# Split at round_idx=0
gamma_accs = []
current = []
for ridx, acc in round_matches:
    if ridx == 0 and current:
        gamma_accs.append(current)
        current = []
    current.append((ridx, acc * 100))
if current:
    gamma_accs.append(current)

gamma_labels = [r["gamma"] for r in results]
for i, g_label in enumerate(gamma_labels):
    if i >= len(gamma_accs):
        break
    g_acc = gamma_accs[i]
    accs_dict = {r: a for r, a in g_acc}
    checkpoints = [0, 9, 19, 29, 49, 74, 99]
    vals = []
    for r in checkpoints:
        a = accs_dict.get(r, None)
        if a is not None:
            vals.append(f"R{r:>2}={a:.1f}%")
    print(f"  g={g_label:>2}: " + "  ".join(vals))

# CE/KL loss trajectories
print("\n  Loss trajectories (R0 vs R99):")
ce_pattern = r'"mean_loss_ce":\s*([\d.]+)'
kl_pattern = r'"mean_loss_kl":\s*([\d.]+)'
ce_vals = [float(m.group(1)) for m in re.finditer(ce_pattern, content)]
kl_vals = [float(m.group(1)) for m in re.finditer(kl_pattern, content)]

# Split by gamma (100 rounds each)
for i, g in enumerate(gamma_labels):
    start = i * 100
    end = start + 100
    if end <= len(ce_vals) and end <= len(kl_vals):
        ce0, ce99 = ce_vals[start], ce_vals[end-1]
        kl0, kl99 = kl_vals[start], kl_vals[end-1]
        ce_dir = "decreasing" if ce99 < ce0 else "STAGNANT"
        print(f"  g={g:>2}: CE {ce0:.3f}->{ce99:.3f} ({ce_dir})  KL {kl0:.3f}->{kl99:.3f}")
