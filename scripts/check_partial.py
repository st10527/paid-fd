#!/usr/bin/env python3
"""Quick check of both partial results."""
import json

# Lambda sweep
with open("results/experiments/v10_1_lambda_20260409_1921.json") as f:
    ld = json.load(f)

sums = ld.get("summaries", {})
print("=== LAMBDA SWEEP (partial) ===")
print("Runs completed: %d/12" % len(sums))
print()
print("%-14s %7s %7s %6s %10s %8s %7s" % ("Label", "Best%", "Final%", "Part%", "CumCost", "MaxPriv", "Price"))
print("-" * 65)
for k in sorted(sums.keys()):
    s = sums[k]
    print("%-14s %6.2f%% %6.2f%% %5.0f%% %10.0f %8.1f %7.3f" % (
        k, s["best_acc"]*100, s["final_acc"]*100,
        s["avg_participation"]*100, s["cumulative_payment"],
        s["max_privacy_spent"], s["avg_price"]))

# 3seeds
print()
with open("results/experiments/v10_1_3seeds_20260409_0922.json") as f:
    sd = json.load(f)
sums2 = sd.get("summaries", {})
print("=== 3-SEED (partial) ===")
print("Runs completed: %d/12" % len(sums2))
for k in sorted(sums2.keys()):
    s = sums2[k]
    print("  %-12s best=%.2f%% cost=%.0f" % (k, s["best_acc"]*100, s["cumulative_payment"]))

# What's missing?
print()
print("=== MISSING RUNS ===")
print("3-seed missing:")
for g in [3, 5, 7, 10]:
    for s in [42, 123, 456]:
        key = "g%d_s%d" % (g, s)
        if key not in sums2:
            print("  " + key)

print("Lambda missing:")
for lm in [0.5, 1.0, 2.0]:
    for g in [3, 5, 7, 10]:
        key = "lm%s_g%d" % (lm, g)
        if key not in sums:
            print("  " + key)
