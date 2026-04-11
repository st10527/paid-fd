#!/usr/bin/env python3
"""
Aggregate individual v10.1 results from TWCC array job.
========================================================
Reads all v10_1_g*_s*_lm*.json files and produces combined analysis.

Usage:
    python scripts/aggregate_twcc_results.py
    python scripts/aggregate_twcc_results.py --indir results/experiments --outdir results/experiments
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import glob
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime


def load_all_results(indir):
    """Load all individual v10_1_*.json result files."""
    pattern = os.path.join(indir, "v10_1_g*_s*_lm*.json")
    files = sorted(glob.glob(pattern))
    
    results = {}
    for f in files:
        try:
            with open(f) as fp:
                d = json.load(fp)
            label = d.get("label", Path(f).stem.replace("v10_1_", ""))
            results[label] = d
            print("  Loaded: %-25s  best=%.2f%%" % (
                label, d["summary"]["best_acc"] * 100))
        except Exception as e:
            print("  WARN: Failed to load %s: %s" % (f, e))
    
    return results


def analyze_3seeds(results):
    """Aggregate 3-seed robustness results."""
    print("\n" + "=" * 70)
    print("3-SEED ROBUSTNESS (mean ± std)")
    print("=" * 70)
    print("%4s | %20s | %22s | %18s | %10s" % (
        "γ", "Best Acc", "Cum Payment", "Max Privacy", "Part%"))
    print("-" * 80)
    
    for g in [3, 5, 7, 10]:
        bests, costs, privs, parts = [], [], [], []
        for s in [42, 123, 456]:
            key = "g%d_s%d_lm1.0" % (g, s)
            if key in results:
                sm = results[key]["summary"]
                bests.append(sm["best_acc"])
                costs.append(sm["cumulative_payment"])
                privs.append(sm["max_privacy_spent"])
                parts.append(sm["avg_participation"])
        
        if bests:
            print("  %2d | %6.2f ± %4.2f%%    | %8.0f ± %7.0f    | %6.1f ± %5.1f  | %5.1f ± %3.1f%%" % (
                g,
                np.mean(bests)*100, np.std(bests)*100,
                np.mean(costs), np.std(costs),
                np.mean(privs), np.std(privs),
                np.mean(parts)*100, np.std(parts)*100))
            
            cv = np.std(bests) / np.mean(bests) * 100 if np.mean(bests) > 0 else 0
            print("       CV=%.2f%% (%s)" % (cv, "STABLE" if cv < 2 else "UNSTABLE"))
    
    return True


def analyze_lambda_sweep(results):
    """Analyze lambda_mult sensitivity."""
    print("\n" + "=" * 70)
    print("LAMBDA SWEEP (seed=42)")
    print("=" * 70)
    print("%5s %4s | %7s %6s %10s %8s %7s" % (
        "λ", "γ", "Best%", "Part%", "CumCost", "MaxPriv", "Price"))
    print("-" * 60)
    
    for lm in [0.5, 1.0, 2.0]:
        for g in [3, 5, 7, 10]:
            key = "g%d_s42_lm%s" % (g, lm)
            if key in results:
                sm = results[key]["summary"]
                print("  %3s %4d | %6.2f%% %5.0f%% %10.0f %8.1f %7.3f" % (
                    lm, g,
                    sm["best_acc"]*100,
                    sm["avg_participation"]*100,
                    sm["cumulative_payment"],
                    sm["max_privacy_spent"],
                    sm["avg_price"]))
        print("-" * 60)
    
    return True


def analyze_efficiency(results):
    """Check E1-E5 criteria with 3-seed confidence."""
    print("\n" + "=" * 70)
    print("EFFICIENCY CRITERIA (E1-E5, 3-seed validated)")
    print("=" * 70)
    
    gamma_stats = {}
    for g in [3, 5, 7, 10]:
        bests, costs, privs, parts = [], [], [], []
        for s in [42, 123, 456]:
            key = "g%d_s%d_lm1.0" % (g, s)
            if key in results:
                sm = results[key]["summary"]
                bests.append(sm["best_acc"])
                costs.append(sm["cumulative_payment"])
                privs.append(sm["max_privacy_spent"])
                parts.append(sm["avg_participation"])
        if bests:
            gamma_stats[g] = {
                "best": np.mean(bests), "best_std": np.std(bests),
                "cost": np.mean(costs), "cost_std": np.std(costs),
                "priv": np.mean(privs), "priv_std": np.std(privs),
                "part": np.mean(parts), "part_std": np.std(parts),
            }
    
    if 3 in gamma_stats and 10 in gamma_stats:
        all_bests = [gamma_stats[g]["best"] for g in gamma_stats]
        spread = max(all_bests) - min(all_bests)
        cost_r = gamma_stats[10]["cost"] / gamma_stats[3]["cost"]
        priv_r = gamma_stats[10]["priv"] / gamma_stats[3]["priv"]
        part_s = gamma_stats[10]["part"] - gamma_stats[3]["part"]
        best_a = max(all_bests)
        
        e1 = spread < 0.02
        e2 = cost_r > 2
        e3 = priv_r > 1.5
        e4 = part_s > 0.1
        e5 = best_a > 0.5
        
        print("  E1 Acc spread:    %.2f%%  %s (threshold < 2%%)" % (spread*100, "✓ PASS" if e1 else "✗ FAIL"))
        print("  E2 Cost ratio:    %.2fx  %s (threshold > 2x)" % (cost_r, "✓ PASS" if e2 else "✗ FAIL"))
        print("  E3 Privacy ratio: %.2fx  %s (threshold > 1.5x)" % (priv_r, "✓ PASS" if e3 else "✗ FAIL"))
        print("  E4 Part spread:   %.0f%%   %s (threshold > 10%%)" % (part_s*100, "✓ PASS" if e4 else "✗ FAIL"))
        print("  E5 Best accuracy: %.2f%%  %s (threshold > 50%%)" % (best_a*100, "✓ PASS" if e5 else "✗ FAIL"))
        
        score = sum([e1, e2, e3, e4, e5])
        print("\n  Efficiency Score: %d/5" % score)
    else:
        print("  Incomplete: need γ=3 and γ=10 results")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Aggregate TWCC results")
    parser.add_argument("--indir", default="results/experiments")
    parser.add_argument("--outdir", default="results/experiments")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  PAID-FD v10.1 — Result Aggregation")
    print("  Scanning: %s/v10_1_g*_s*_lm*.json" % args.indir)
    print("=" * 70)
    
    results = load_all_results(args.indir)
    print("\n  Total: %d result files loaded" % len(results))
    
    if not results:
        print("\n  No results found! Check --indir path.")
        return
    
    # Run analyses
    analyze_3seeds(results)
    analyze_lambda_sweep(results)
    analyze_efficiency(results)
    
    # Save combined summary
    combined = {
        "experiment": "v10_1_twcc_combined",
        "aggregated_at": datetime.now().isoformat(),
        "n_results": len(results),
        "labels": sorted(results.keys()),
        "summaries": {k: v["summary"] for k, v in results.items()},
    }
    outfile = os.path.join(args.outdir, "v10_1_twcc_aggregated.json")
    with open(outfile, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print("\n  Saved aggregated: %s" % outfile)


if __name__ == "__main__":
    main()
