#!/usr/bin/env python3
"""Parse v10 persistent models results with EFFICIENCY STORY framing."""

import json
import numpy as np

with open("results/experiments/v10_persistent_20260408_0840.json") as f:
    data = json.load(f)

print("=" * 80)
print(f"v10 Results: {data['experiment']} ({data['framing']})")
print(f"Timestamp: {data['timestamp']}")
print("=" * 80)

gamma_values = [3, 5, 7, 10]
rows = []

for gamma in gamma_values:
    label = f"g{gamma}"
    run = data["results"].get(label)
    if not run:
        print(f"  WARNING: No results for γ={gamma}")
        continue
    
    accs = run["accuracies"]
    extras = run.get("extras", [])
    prices = run.get("prices", [])
    avg_eps_list = run.get("avg_eps", [])
    part_rates = run.get("participation_rates", [])
    energy_hist = run.get("energy_history", [])
    
    final_acc = accs[-1] if accs else 0
    best_acc = max(accs) if accs else 0
    r1_acc = accs[0] if accs else 0
    r10_acc = accs[9] if len(accs) > 9 else 0
    r50_acc = accs[49] if len(accs) > 49 else 0
    
    # Efficiency metrics from extras
    last_extra = extras[-1] if extras else {}
    cum_payment = last_extra.get("cumulative_payment", 0)
    avg_priv_spent = last_extra.get("avg_privacy_spent", 0)
    max_priv_spent = last_extra.get("max_privacy_spent", 0)
    
    # Average per-round
    avg_price = np.mean(prices) if prices else 0
    avg_eps = np.mean(avg_eps_list) if avg_eps_list else 0
    avg_part = np.mean(part_rates) if part_rates else 0
    
    # Round payments
    round_payments = [e.get("round_payment", 0) for e in extras if e]
    avg_round_pay = np.mean(round_payments) if round_payments else 0
    
    # Distill delta
    distill_deltas = [e.get("distill_delta", None) for e in extras if e]
    distill_deltas = [d for d in distill_deltas if d is not None]
    avg_delta = np.mean(distill_deltas) if distill_deltas else float('nan')
    
    # Pre/post distill acc
    pre_accs = [e.get("pre_distill_acc", None) for e in extras if e]
    post_accs = [e.get("post_distill_acc", None) for e in extras if e]
    pre_accs = [a for a in pre_accs if a is not None]
    post_accs = [a for a in post_accs if a is not None]
    
    # Server utility
    server_utils = [e.get("server_utility", 0) for e in extras if e]
    avg_server_util = np.mean(server_utils) if server_utils else 0
    
    # Total quality
    total_quals = [e.get("total_quality", 0) for e in extras if e]
    avg_total_qual = np.mean(total_quals) if total_quals else 0
    
    # eps_std
    eps_stds = [e.get("eps_std", 0) for e in extras if e]
    avg_eps_std = np.mean(eps_stds) if eps_stds else 0
    
    # Energy
    total_energy = sum(
        sum(e.values()) for e in energy_hist if isinstance(e, dict)
    )
    
    # Time to targets
    targets = [0.40, 0.45, 0.50, 0.55, 0.58, 0.60]
    ttt = {}
    for target in targets:
        reached = [i for i, a in enumerate(accs) if a >= target]
        ttt[target] = reached[0] if reached else None
    
    # N local models
    n_models = last_extra.get("n_local_models", 0)
    
    rows.append({
        "gamma": gamma,
        "final_acc": final_acc,
        "best_acc": best_acc,
        "r1_acc": r1_acc,
        "r10_acc": r10_acc,
        "r50_acc": r50_acc,
        "avg_participation": avg_part,
        "avg_price": avg_price,
        "avg_eps": avg_eps,
        "avg_eps_std": avg_eps_std,
        "cum_payment": cum_payment,
        "avg_round_pay": avg_round_pay,
        "avg_priv_spent": avg_priv_spent,
        "max_priv_spent": max_priv_spent,
        "avg_server_util": avg_server_util,
        "avg_total_qual": avg_total_qual,
        "avg_delta": avg_delta,
        "total_energy": total_energy,
        "ttt": ttt,
        "n_models": n_models,
        "n_rounds": len(accs),
    })

# ============================================================
# TABLE 1: ACCURACY (expect ~flat = good for efficiency story)
# ============================================================
print(f"\n{'='*70}")
print("TABLE 1: ACCURACY TRAJECTORY (expect ~flat across γ = efficiency ceiling)")
print(f"{'='*70}")
print(f"{'γ':>4} | {'R1':>7} | {'R10':>7} | {'R50':>7} | {'Final':>7} | {'Best':>7} | {'Δ/rnd':>9}")
print("-" * 60)
for r in rows:
    print(f"{r['gamma']:4d} | {r['r1_acc']*100:6.2f}% | {r['r10_acc']*100:6.2f}% | "
          f"{r['r50_acc']*100:6.2f}% | {r['final_acc']*100:6.2f}% | {r['best_acc']*100:6.2f}% | "
          f"{r['avg_delta']*100:+8.4f}%")

# ============================================================
# TABLE 2: EFFICIENCY METRICS (the story)
# ============================================================
print(f"\n{'='*70}")
print("TABLE 2: EFFICIENCY METRICS (expect clear γ differentiation)")
print(f"{'='*70}")
print(f"{'γ':>4} | {'Part%':>7} | {'Avg p*':>8} | {'Avg ε*':>8} | "
      f"{'Σ Cost':>10} | {'Avg/rnd':>8} | {'Avg Σε':>8} | {'Max Σε':>8}")
print("-" * 80)
for r in rows:
    print(f"{r['gamma']:4d} | {r['avg_participation']*100:6.1f}% | "
          f"{r['avg_price']:8.4f} | {r['avg_eps']:8.4f} | "
          f"{r['cum_payment']:10.2f} | {r['avg_round_pay']:8.2f} | "
          f"{r['avg_priv_spent']:8.2f} | {r['max_priv_spent']:8.2f}")

# ============================================================
# TABLE 3: GAME EQUILIBRIUM
# ============================================================
print(f"\n{'='*70}")
print("TABLE 3: GAME EQUILIBRIUM QUALITY")
print(f"{'='*70}")
print(f"{'γ':>4} | {'Server Util':>12} | {'Total Qual':>11} | {'ε std':>8} | {'N models':>9}")
print("-" * 55)
for r in rows:
    print(f"{r['gamma']:4d} | {r['avg_server_util']:12.4f} | {r['avg_total_qual']:11.4f} | "
          f"{r['avg_eps_std']:8.4f} | {r['n_models']:9d}")

# ============================================================
# TABLE 4: TIME TO TARGET
# ============================================================
print(f"\n{'='*70}")
print("TABLE 4: TIME-TO-TARGET (round # to first reach accuracy)")
print(f"{'='*70}")
targets = [0.40, 0.45, 0.50, 0.55, 0.58, 0.60]
header = f"{'γ':>4} | " + " | ".join(f"{t*100:5.0f}%" for t in targets)
print(header)
print("-" * len(header))
for r in rows:
    vals = []
    for t in targets:
        rnd = r["ttt"].get(t, None)
        vals.append(f"{rnd:6d}" if rnd is not None else "   N/A")
    print(f"{r['gamma']:4d} | " + " | ".join(vals))

# ============================================================
# EFFICIENCY RATIOS
# ============================================================
print(f"\n{'='*70}")
print("EFFICIENCY RATIOS (γ=10 / γ=3)")
print(f"{'='*70}")
g3 = next((r for r in rows if r["gamma"] == 3), None)
g10 = next((r for r in rows if r["gamma"] == 10), None)
g5 = next((r for r in rows if r["gamma"] == 5), None)
g7 = next((r for r in rows if r["gamma"] == 7), None)

if g3 and g10:
    acc_diff = abs(g10["final_acc"] - g3["final_acc"])
    cost_ratio = g10["cum_payment"] / g3["cum_payment"] if g3["cum_payment"] > 0 else float('inf')
    priv_ratio = g10["avg_priv_spent"] / g3["avg_priv_spent"] if g3["avg_priv_spent"] > 0 else float('inf')
    part_ratio = g10["avg_participation"] / g3["avg_participation"] if g3["avg_participation"] > 0 else float('inf')
    price_ratio = g10["avg_price"] / g3["avg_price"] if g3["avg_price"] > 0 else float('inf')
    eps_ratio = g10["avg_eps"] / g3["avg_eps"] if g3["avg_eps"] > 0 else float('inf')
    
    print(f"  Accuracy difference:     {acc_diff*100:.2f}%  (small = good, flat ceiling)")
    print(f"  Cost ratio (Σ pay):      {cost_ratio:.2f}×  (want >2×)")
    print(f"  Privacy ratio (avg Σε):  {priv_ratio:.2f}×  (want >1.5×)")
    print(f"  Participation ratio:     {part_ratio:.2f}×  (want >1×)")
    print(f"  Price ratio (avg p*):    {price_ratio:.2f}×")
    print(f"  Eps ratio (avg ε*):      {eps_ratio:.2f}×")

# ============================================================
# MONOTONICITY CHECK (Proposition 2)
# ============================================================
print(f"\n{'='*70}")
print("PROPOSITION 2 MONOTONICITY CHECK")
print("(Higher γ → higher p* → higher ε* → higher participation)")
print(f"{'='*70}")

price_mono = all(rows[i]["avg_price"] <= rows[i+1]["avg_price"] for i in range(len(rows)-1))
eps_mono = all(rows[i]["avg_eps"] <= rows[i+1]["avg_eps"] for i in range(len(rows)-1))
part_mono = all(rows[i]["avg_participation"] <= rows[i+1]["avg_participation"] for i in range(len(rows)-1))
cost_mono = all(rows[i]["cum_payment"] <= rows[i+1]["cum_payment"] for i in range(len(rows)-1))
priv_mono = all(rows[i]["avg_priv_spent"] <= rows[i+1]["avg_priv_spent"] for i in range(len(rows)-1))

checks = [
    ("p* ↑ with γ", price_mono, [r["avg_price"] for r in rows]),
    ("ε* ↑ with γ", eps_mono, [r["avg_eps"] for r in rows]),
    ("Participation ↑ with γ", part_mono, [r["avg_participation"] for r in rows]),
    ("Σ Cost ↑ with γ", cost_mono, [r["cum_payment"] for r in rows]),
    ("Σ Privacy ↑ with γ", priv_mono, [r["avg_priv_spent"] for r in rows]),
]
for name, passed, vals in checks:
    status = "✓ YES" if passed else "✗ NO "
    val_str = " → ".join(f"{v:.4f}" for v in vals)
    print(f"  {status}  {name:30s}  [{val_str}]")

# ============================================================
# SUCCESS CRITERIA
# ============================================================
print(f"\n{'='*70}")
print("EFFICIENCY STORY SUCCESS CRITERIA")
print(f"{'='*70}")

finals = {r["gamma"]: r["final_acc"] for r in rows}
best_final = max(finals.values()) if finals else 0
worst_final = min(finals.values()) if finals else 0
acc_spread = best_final - worst_final

costs = {r["gamma"]: r["cum_payment"] for r in rows}
cost_ratio_all = max(costs.values()) / min(costs.values()) if min(costs.values()) > 0 else 0

privs = {r["gamma"]: r["avg_priv_spent"] for r in rows}
priv_ratio_all = max(privs.values()) / min(privs.values()) if min(privs.values()) > 0 else 0

parts = {r["gamma"]: r["avg_participation"] for r in rows}
part_spread = max(parts.values()) - min(parts.values())

e1 = acc_spread < 0.05
e2 = cost_ratio_all >= 2.0
e3 = priv_ratio_all >= 1.5
e4 = part_spread >= 0.10
e5 = best_final >= 0.50

print(f"  E1: Acc ceiling ~flat (spread < 5%)?    {'YES ✓' if e1 else 'NO ✗'}  (spread={acc_spread*100:.2f}%)")
print(f"  E2: Cost ratio > 2×?                    {'YES ✓' if e2 else 'NO ✗'}  (ratio={cost_ratio_all:.2f}×)")
print(f"  E3: Privacy ratio > 1.5×?               {'YES ✓' if e3 else 'NO ✗'}  (ratio={priv_ratio_all:.2f}×)")
print(f"  E4: Participation spread > 10%?          {'YES ✓' if e4 else 'NO ✗'}  (spread={part_spread*100:.1f}%)")
print(f"  E5: System works (best acc > 50%)?       {'YES ✓' if e5 else 'NO ✗'}  (best={best_final*100:.1f}%)")

n_pass = sum([e1, e2, e3, e4, e5])
print(f"\n  Score: {n_pass}/5")

# ============================================================
# ACCURACY CURVE SNAPSHOT (every 10 rounds)
# ============================================================
print(f"\n{'='*70}")
print("ACCURACY CURVES (every 10 rounds)")
print(f"{'='*70}")
header = f"{'Round':>6} | " + " | ".join(f"γ={r['gamma']:2d}" for r in rows)
print(header)
print("-" * len(header))
for rnd in range(0, 100, 10):
    vals = []
    for r in rows:
        label = f"g{r['gamma']}"
        run = data["results"][label]
        accs = run["accuracies"]
        if rnd < len(accs):
            vals.append(f"{accs[rnd]*100:5.2f}%")
        else:
            vals.append("   N/A")
    print(f"{rnd:6d} | " + " | ".join(vals))
# Final round
vals = []
for r in rows:
    label = f"g{r['gamma']}"
    run = data["results"][label]
    accs = run["accuracies"]
    vals.append(f"{accs[-1]*100:5.2f}%")
print(f"{'99':>6} | " + " | ".join(vals))

# ============================================================
# PARTICIPATION & COST CURVES (every 10 rounds)
# ============================================================
print(f"\n{'='*70}")
print("CUMULATIVE COST CURVES (every 10 rounds)")
print(f"{'='*70}")
header = f"{'Round':>6} | " + " | ".join(f"γ={r['gamma']:2d}" for r in rows)
print(header)
print("-" * len(header))
for rnd in [0, 9, 19, 29, 49, 69, 99]:
    vals = []
    for r in rows:
        label = f"g{r['gamma']}"
        run = data["results"][label]
        extras = run.get("extras", [])
        if rnd < len(extras) and extras[rnd]:
            cp = extras[rnd].get("cumulative_payment", 0)
            vals.append(f"{cp:8.1f}")
        else:
            vals.append("     N/A")
    print(f"{rnd:6d} | " + " | ".join(vals))

print(f"\n{'='*70}")
print("DONE. Check GPU log for full output.")
print(f"{'='*70}")
