#!/usr/bin/env python3
"""
Fig 4 Data Sanity Check
========================
Validates efficiency_frontier_data.csv before plotting.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from pathlib import Path
from collections import defaultdict

CSV_PATH = Path("results/analysis/efficiency_frontier_data.csv")

# Expected rough ranges (±5% tolerance used below)
EXPECTED = {
    3:  {"participation_rate": 0.38, "total_payment": 8737,  "best_acc": 61.22},
    5:  {"participation_rate": 0.79, "total_payment": 30178, "best_acc": 61.43},
    7:  {"participation_rate": 0.90, "total_payment": 40549, "best_acc": 61.43},
    10: {"participation_rate": 1.00, "total_payment": 55443, "best_acc": 61.45},
}

METRICS = ["participation_rate", "max_cum_eps", "mean_cum_eps",
           "total_payment", "best_acc"]

def mean(vals):
    return sum(vals) / len(vals)

def std(vals):
    m = mean(vals)
    return (sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5

def main():
    # ── 1. Read CSV ──────────────────────────────────────────
    if not CSV_PATH.exists():
        print("ERROR: %s not found" % CSV_PATH)
        sys.exit(1)

    rows = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        print("Columns:", cols)
        for row in reader:
            rows.append({k: float(v) if k != 'seed' else int(float(v))
                         for k, v in row.items()})

    # ── 2. Check 4 gamma values × 3 seeds = 12 rows ─────────
    gammas = sorted(set(int(r['gamma']) for r in rows))
    print("\nGamma values found:", gammas)
    assert gammas == [3, 5, 7, 10], "Expected [3,5,7,10], got %s" % gammas
    assert len(rows) == 12, "Expected 12 rows, got %d" % len(rows)
    print("Row count: %d  ✓" % len(rows))

    # ── 3. Per-gamma stats ───────────────────────────────────
    by_gamma = defaultdict(list)
    for r in rows:
        by_gamma[int(r['gamma'])].append(r)

    print("\n%s" % ("─" * 70))
    print("  %-4s  %-20s  %-20s  %-20s  %-14s  %-10s" % (
        "γ", "participation", "max_cum_eps", "total_payment(K)", "mean_cum_eps", "best_acc%"))
    print("─" * 70)

    TOLERANCE = 0.10   # 10% tolerance
    failed = []

    for g in [3, 5, 7, 10]:
        vals = by_gamma[g]
        pr   = [v["participation_rate"] for v in vals]
        mce  = [v["max_cum_eps"]        for v in vals]
        mce2 = [v["mean_cum_eps"]       for v in vals]
        pay  = [v["total_payment"]      for v in vals]
        acc  = [v["best_acc"]           for v in vals]

        print("  γ=%-2d  %.3f ± %.3f       %.1f ± %.1f     %.1f ± %.1f (K)     %.1f ± %.1f    %.2f ± %.2f" % (
            g,
            mean(pr),  std(pr),
            mean(mce), std(mce),
            mean(pay)/1000, std(pay)/1000,
            mean(mce2), std(mce2),
            mean(acc), std(acc),
        ))

        if g in EXPECTED:
            exp = EXPECTED[g]
            for key, exp_val in exp.items():
                if key == "best_acc":
                    actual = mean(acc)
                elif key == "participation_rate":
                    actual = mean(pr)
                elif key == "total_payment":
                    actual = mean(pay)
                else:
                    continue
                rel_err = abs(actual - exp_val) / (abs(exp_val) + 1e-9)
                if rel_err > TOLERANCE:
                    failed.append("γ=%d %s: got %.4g, expected %.4g (err=%.1f%%)" % (
                        g, key, actual, exp_val, rel_err * 100))

    print("─" * 70)

    # ── 4. Annotate ratios ───────────────────────────────────
    pr_ratio  = mean([r["participation_rate"] for r in by_gamma[10]]) / \
                mean([r["participation_rate"] for r in by_gamma[3]])
    mce_ratio = mean([r["max_cum_eps"] for r in by_gamma[10]]) / \
                mean([r["max_cum_eps"] for r in by_gamma[3]])
    pay_ratio = mean([r["total_payment"] for r in by_gamma[10]]) / \
                mean([r["total_payment"] for r in by_gamma[3]])

    print("\nKey ratios (γ=10 / γ=3):")
    print("  Participation  : %.2f×" % pr_ratio)
    print("  Max cum ε      : %.2f×" % mce_ratio)
    print("  Total payment  : %.2f×" % pay_ratio)

    # ── 5. PASS / FAIL ───────────────────────────────────────
    print()
    if failed:
        print("SANITY CHECK FAILED:")
        for f in failed:
            print("  ✗ " + f)
        sys.exit(1)
    else:
        print("SANITY CHECK PASSED ✓")
        print("Ready to run: python scripts/plot_fig4_v2.py")

if __name__ == "__main__":
    main()
