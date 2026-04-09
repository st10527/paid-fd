#!/usr/bin/env python3
"""Parse and analyze v10.1 3-seed results."""
import json
import numpy as np

FILE = "results/experiments/v10_1_3seeds_20260409_0922.json"

with open(FILE) as f:
    data = json.load(f)

print("=" * 80)
print("v10.1 THREE-SEED ROBUSTNESS RESULTS")
print("=" * 80)

runs = data.get("runs", {})
summaries = data.get("summaries", {})
agg = data.get("aggregated", {})

print(f"Runs completed: {len(runs)}/12")
print(f"Summaries: {len(summaries)}")
print(f"Aggregated: {len(agg)} (empty = need to compute here)")
print()

# Per-run table
print("[Per-Run Results]")
print(f"  {'Label':<12} {'Best%':>7} {'Final%':>7} {'Part%':>6} {'CumCost':>10} {'MaxPriv':>8} {'Price':>7} {'eps*':>6}")
print("-" * 75)
for k in sorted(summaries.keys()):
    s = summaries[k]
    print(f"  {k:<12} {s['best_acc']*100:>6.2f}% {s['final_acc']*100:>6.2f}% "
          f"{s['avg_participation']*100:>5.0f}% {s['cumulative_payment']:>10,.0f} "
          f"{s['max_privacy_spent']:>8.1f} {s['avg_price']:>7.3f} {s['avg_eps_per_round']:>6.3f}")

# Aggregate across seeds
GAMMAS = [3, 5, 7, 10]
SEEDS = [42, 123, 456]

print("\n" + "=" * 80)
print("AGGREGATED (mean +/- std across seeds)")
print("=" * 80)

agg_data = {}
for g in GAMMAS:
    seed_sums = []
    for s in SEEDS:
        key = f"g{g}_s{s}"
        if key in summaries:
            seed_sums.append(summaries[key])
    if not seed_sums:
        continue
    
    metrics = {}
    for mkey in ["final_acc", "best_acc", "avg_participation", "avg_price",
                  "avg_eps_per_round", "cumulative_payment", "avg_privacy_spent",
                  "max_privacy_spent", "energy_total", "avg_distill_delta"]:
        vals = [ss[mkey] for ss in seed_sums if mkey in ss]
        if vals:
            metrics[mkey] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                             "values": vals}
    
    # Time-to-targets
    targets = ["0.45", "0.5", "0.55", "0.58", "0.6"]
    ttt_agg = {}
    for t in targets:
        vals = [ss["time_to_targets"].get(t, None) for ss in seed_sums]
        vals = [v for v in vals if v is not None]
        if vals:
            ttt_agg[t] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                          "n": len(vals)}
    metrics["time_to_targets"] = ttt_agg
    
    agg_data[g] = metrics

# Table 1: Accuracy
print(f"\n{'g':>4} | {'Best Acc':>20} | {'Final Acc':>20} | {'Seeds':>5}")
print("-" * 60)
for g in GAMMAS:
    if g not in agg_data:
        print(f"  {g:<3}  MISSING")
        continue
    m = agg_data[g]
    ba = m["best_acc"]
    fa = m["final_acc"]
    n = len(ba.get("values", []))
    print(f"  {g:<3} | {ba['mean']*100:6.2f} +/- {ba['std']*100:4.2f}%   | "
          f"{fa['mean']*100:6.2f} +/- {fa['std']*100:4.2f}%   | {n}")

# Table 2: Efficiency
print(f"\n{'g':>4} | {'Participation':>18} | {'Cum Payment':>22} | {'Max Privacy':>18}")
print("-" * 75)
for g in GAMMAS:
    if g not in agg_data:
        continue
    m = agg_data[g]
    p = m["avg_participation"]
    c = m["cumulative_payment"]
    pr = m["max_privacy_spent"]
    print(f"  {g:<3} | {p['mean']*100:5.1f} +/- {p['std']*100:4.1f}%    | "
          f"{c['mean']:8,.0f} +/- {c['std']:7,.0f}     | "
          f"{pr['mean']:6.1f} +/- {pr['std']:5.1f}")

# Table 3: Time-to-target
print(f"\n{'g':>4} | {'@50%':>14} | {'@55%':>14} | {'@58%':>14} | {'@60%':>14}")
print("-" * 70)
for g in GAMMAS:
    if g not in agg_data:
        continue
    ttt = agg_data[g]["time_to_targets"]
    vals = []
    for t in ["0.5", "0.55", "0.58", "0.6"]:
        if t in ttt:
            vals.append(f"{ttt[t]['mean']:5.1f}+/-{ttt[t]['std']:4.1f}")
        else:
            vals.append("     N/A    ")
    print(f"  {g:<3} | " + " | ".join(vals))

# Efficiency criteria with error bars
print("\n" + "=" * 80)
print("EFFICIENCY CRITERIA (3-seed validated)")
print("=" * 80)

if len(agg_data) >= 2:
    g_lo = min(agg_data.keys())
    g_hi = max(agg_data.keys())
    lo = agg_data[g_lo]
    hi = agg_data[g_hi]
    
    # E1: Accuracy spread
    all_best = [agg_data[g]["best_acc"]["mean"] for g in agg_data]
    acc_spread = max(all_best) - min(all_best)
    acc_spread_err = max(agg_data[g]["best_acc"]["std"] for g in agg_data)
    e1 = acc_spread < 0.02
    print(f"  E1 Acc spread:    {acc_spread*100:.2f}% (max seed std: {acc_spread_err*100:.2f}%)  {'PASS' if e1 else 'FAIL'}")
    
    # E2: Cost ratio
    c_lo = lo["cumulative_payment"]
    c_hi = hi["cumulative_payment"]
    cr = c_hi["mean"] / c_lo["mean"] if c_lo["mean"] > 0 else 0
    cr_err = cr * np.sqrt((c_lo["std"]/c_lo["mean"])**2 + (c_hi["std"]/c_hi["mean"])**2) if c_lo["mean"] > 0 and c_hi["mean"] > 0 else 0
    e2 = cr > 2.0
    print(f"  E2 Cost ratio:    {cr:.2f} +/- {cr_err:.2f}x  {'PASS' if e2 else 'FAIL'}")
    
    # E3: Privacy ratio
    p_lo = lo["max_privacy_spent"]
    p_hi = hi["max_privacy_spent"]
    pr = p_hi["mean"] / p_lo["mean"] if p_lo["mean"] > 0 else 0
    pr_err = pr * np.sqrt((p_lo["std"]/p_lo["mean"])**2 + (p_hi["std"]/p_hi["mean"])**2) if p_lo["mean"] > 0 and p_hi["mean"] > 0 else 0
    e3 = pr > 1.5
    print(f"  E3 Privacy ratio: {pr:.2f} +/- {pr_err:.2f}x  {'PASS' if e3 else 'FAIL'}")
    
    # E4: Participation spread
    part_lo = lo["avg_participation"]["mean"]
    part_hi = hi["avg_participation"]["mean"]
    ps = part_hi - part_lo
    ps_err = np.sqrt(lo["avg_participation"]["std"]**2 + hi["avg_participation"]["std"]**2)
    e4 = ps > 0.1
    print(f"  E4 Part spread:   {ps*100:.0f} +/- {ps_err*100:.0f}%  {'PASS' if e4 else 'FAIL'}")
    
    # E5: Best accuracy
    best_any = max(all_best)
    best_std = max(agg_data[g]["best_acc"]["std"] for g in agg_data)
    e5 = best_any > 0.5
    print(f"  E5 Best accuracy: {best_any*100:.2f} +/- {best_std*100:.2f}%  {'PASS' if e5 else 'FAIL'}")
    
    score = sum([e1, e2, e3, e4, e5])
    print(f"\n  SCORE: {score}/5")

# Proposition 2 monotonicity
print("\n" + "=" * 80)
print("PROPOSITION 2 MONOTONICITY (mean across seeds)")
print("=" * 80)
for name, key in [("price p*", "avg_price"), ("participation", "avg_participation"),
                   ("cumulative cost", "cumulative_payment")]:
    vals = [agg_data[g][key]["mean"] for g in sorted(agg_data.keys())]
    stds = [agg_data[g][key]["std"] for g in sorted(agg_data.keys())]
    diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
    mono = all(d > 0 for d in diffs)
    vals_str = " -> ".join(f"{v:.2f}" for v in vals)
    print(f"  {name:<18}: {vals_str}  [{'MONO' if mono else 'NOT MONO'}]")

# Seed stability check
print("\n" + "=" * 80)
print("SEED STABILITY (coefficient of variation)")
print("=" * 80)
print(f"  {'g':>3} | {'CV(best_acc)':>12} | {'CV(cost)':>12} | {'CV(max_priv)':>14}")
print("  " + "-" * 50)
for g in sorted(agg_data.keys()):
    m = agg_data[g]
    cv_acc = m["best_acc"]["std"] / m["best_acc"]["mean"] * 100 if m["best_acc"]["mean"] > 0 else 0
    cv_cost = m["cumulative_payment"]["std"] / m["cumulative_payment"]["mean"] * 100 if m["cumulative_payment"]["mean"] > 0 else 0
    cv_priv = m["max_privacy_spent"]["std"] / m["max_privacy_spent"]["mean"] * 100 if m["max_privacy_spent"]["mean"] > 0 else 0
    print(f"  {g:>3} | {cv_acc:>10.1f}%  | {cv_cost:>10.1f}%  | {cv_priv:>12.1f}%")

print("\n  (CV < 5% = excellent stability, < 10% = good, > 20% = concern)")
