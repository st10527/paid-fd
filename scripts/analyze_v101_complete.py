#!/usr/bin/env python3
"""
v10.1 Complete Analysis — 3-seed + Lambda Sweep
=================================================
Reads combined results and produces full paper-ready analysis.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np


def load_combined():
    """Load all summaries, merge lm1.0 aliases."""
    with open('results/experiments/v10_1_combined_20260409_2304.json') as f:
        d = json.load(f)
    sm = d['summaries']
    
    # Fix missing aliases: g*_s42 at lm=1.0 = lm1.0_g*
    for g in [3, 5, 7, 10]:
        seed_key = "g%d_s42" % g
        lm_key = "lm1.0_g%d" % g
        if seed_key in sm and lm_key not in sm:
            sm[lm_key] = sm[seed_key]
    
    return sm


def analyze_3seeds(sm):
    """3-seed robustness for each gamma."""
    print("\n" + "=" * 80)
    print("  3-SEED ROBUSTNESS (γ × seed={42,123,456}, λ_mult=1.0)")
    print("=" * 80)
    print("  %-4s | %-22s | %-22s | %-18s | %-14s" % (
        "γ", "Best Acc (%)", "Cum Payment", "Max Privacy (ε)", "Participation"))
    print("  " + "-" * 86)
    
    all_gamma_means = {}
    
    for g in [3, 5, 7, 10]:
        bests, costs, privs, parts, finals = [], [], [], [], []
        for s in [42, 123, 456]:
            key = "g%d_s%d" % (g, s)
            if key in sm:
                bests.append(sm[key]["best_acc"])
                finals.append(sm[key]["final_acc"])
                costs.append(sm[key]["cumulative_payment"])
                privs.append(sm[key]["max_privacy_spent"])
                parts.append(sm[key]["avg_participation"])
        
        if len(bests) == 3:
            all_gamma_means[g] = {
                "best_mean": np.mean(bests), "best_std": np.std(bests),
                "cost_mean": np.mean(costs), "cost_std": np.std(costs),
                "priv_mean": np.mean(privs), "priv_std": np.std(privs),
                "part_mean": np.mean(parts), "part_std": np.std(parts),
            }
            cv = np.std(bests) / np.mean(bests) * 100
            print("  %-4d | %6.2f ± %-5.2f (CV=%.1f%%) | %8.0f ± %-8.0f | %6.1f ± %-6.1f | %5.1f ± %-4.1f%%" % (
                g,
                np.mean(bests)*100, np.std(bests)*100, cv,
                np.mean(costs), np.std(costs),
                np.mean(privs), np.std(privs),
                np.mean(parts)*100, np.std(parts)*100))
        else:
            print("  %-4d | INCOMPLETE (%d/3 seeds)" % (g, len(bests)))
    
    # Per-seed detail
    print("\n  Per-seed detail:")
    print("  %-10s | %7s | %7s | %10s | %8s | %6s" % (
        "Run", "Best%", "Final%", "Cost", "MaxPriv", "Part%"))
    print("  " + "-" * 60)
    for g in [3, 5, 7, 10]:
        for s in [42, 123, 456]:
            key = "g%d_s%d" % (g, s)
            if key in sm:
                print("  %-10s | %6.2f%% | %6.2f%% | %10.0f | %8.1f | %5.1f%%" % (
                    key,
                    sm[key]["best_acc"]*100,
                    sm[key]["final_acc"]*100,
                    sm[key]["cumulative_payment"],
                    sm[key]["max_privacy_spent"],
                    sm[key]["avg_participation"]*100))
        print("  " + "-" * 60)
    
    return all_gamma_means


def analyze_lambda(sm):
    """Lambda sensitivity analysis."""
    print("\n" + "=" * 80)
    print("  LAMBDA SWEEP (λ_mult × γ, seed=42)")
    print("=" * 80)
    print("  %-5s %-4s | %7s %7s %10s %8s %7s %8s" % (
        "λ", "γ", "Best%", "Final%", "CumCost", "MaxPriv", "Price", "AvgEps"))
    print("  " + "-" * 65)
    
    lambda_data = {}
    for lm in [0.5, 1.0, 2.0]:
        lambda_data[lm] = {}
        for g in [3, 5, 7, 10]:
            key = "lm%s_g%d" % (lm, g)
            if key in sm:
                s = sm[key]
                lambda_data[lm][g] = s
                print("  %-5s %-4d | %6.2f%% %6.2f%% %10.0f %8.1f %7.3f %8.4f" % (
                    lm, g,
                    s["best_acc"]*100,
                    s["final_acc"]*100,
                    s["cumulative_payment"],
                    s["max_privacy_spent"],
                    s["avg_price"],
                    s["avg_eps_per_round"]))
        print("  " + "-" * 65)
    
    # Lambda impact summary
    print("\n  Lambda Impact Summary (averaged across γ):")
    print("  %-5s | %7s | %10s | %8s | %6s" % (
        "λ", "Best%", "AvgCost", "AvgPriv", "Part%"))
    print("  " + "-" * 45)
    for lm in [0.5, 1.0, 2.0]:
        if lambda_data[lm]:
            vals = list(lambda_data[lm].values())
            print("  %-5s | %6.2f%% | %10.0f | %8.1f | %5.1f%%" % (
                lm,
                np.mean([v["best_acc"] for v in vals])*100,
                np.mean([v["cumulative_payment"] for v in vals]),
                np.mean([v["max_privacy_spent"] for v in vals]),
                np.mean([v["avg_participation"] for v in vals])*100))
    
    return lambda_data


def analyze_efficiency(gamma_stats):
    """E1-E5 criteria with 3-seed confidence intervals."""
    print("\n" + "=" * 80)
    print("  EFFICIENCY CRITERIA (E1-E5) — 3-seed validated")
    print("=" * 80)
    
    if 3 not in gamma_stats or 10 not in gamma_stats:
        print("  ERROR: Missing γ=3 or γ=10 data!")
        return
    
    all_bests = [gamma_stats[g]["best_mean"] for g in sorted(gamma_stats.keys())]
    spread = max(all_bests) - min(all_bests)
    cost_ratio = gamma_stats[10]["cost_mean"] / gamma_stats[3]["cost_mean"]
    priv_ratio = gamma_stats[10]["priv_mean"] / gamma_stats[3]["priv_mean"]
    part_spread = gamma_stats[10]["part_mean"] - gamma_stats[3]["part_mean"]
    best_acc = max(all_bests)
    
    e1 = spread < 0.02
    e2 = cost_ratio > 2
    e3 = priv_ratio > 1.5
    e4 = part_spread > 0.1
    e5 = best_acc > 0.5
    
    print("  E1 Accuracy spread (γ3→γ10):  %.2f%%     %s (threshold < 2%%)" % (
        spread*100, "✅ PASS" if e1 else "❌ FAIL"))
    print("  E2 Cost ratio (γ10/γ3):       %.2fx     %s (threshold > 2x)" % (
        cost_ratio, "✅ PASS" if e2 else "❌ FAIL"))
    print("  E3 Privacy ratio (γ10/γ3):    %.2fx     %s (threshold > 1.5x)" % (
        priv_ratio, "✅ PASS" if e3 else "❌ FAIL"))
    print("  E4 Participation spread:      %.0f%%      %s (threshold > 10%%)" % (
        part_spread*100, "✅ PASS" if e4 else "❌ FAIL"))
    print("  E5 Best accuracy:             %.2f%%    %s (threshold > 50%%)" % (
        best_acc*100, "✅ PASS" if e5 else "❌ FAIL"))
    
    score = sum([e1, e2, e3, e4, e5])
    print("\n  🏆 Efficiency Score: %d/5" % score)
    
    # Confidence: report 95% CI for key metrics
    print("\n  95%% CI (mean ± 1.96*std/√3):")
    for g in [3, 5, 7, 10]:
        gs = gamma_stats[g]
        ci = 1.96 * gs["best_std"] / np.sqrt(3)
        print("    γ=%-2d: %.2f%% ± %.2f%%  [%.2f%%, %.2f%%]" % (
            g, gs["best_mean"]*100, ci*100,
            (gs["best_mean"]-ci)*100, (gs["best_mean"]+ci)*100))


def analyze_proposition2(sm):
    """Check Proposition 2 monotonicity for all seeds."""
    print("\n" + "=" * 80)
    print("  PROPOSITION 2 MONOTONICITY CHECK")
    print("=" * 80)
    
    for s in [42, 123, 456]:
        costs = []
        privs = []
        parts = []
        for g in [3, 5, 7, 10]:
            key = "g%d_s%d" % (g, s)
            if key in sm:
                costs.append(sm[key]["cumulative_payment"])
                privs.append(sm[key]["max_privacy_spent"])
                parts.append(sm[key]["avg_participation"])
        
        cost_mono = all(costs[i] <= costs[i+1] for i in range(len(costs)-1))
        priv_mono = all(privs[i] <= privs[i+1] for i in range(len(privs)-1))
        part_mono = all(parts[i] <= parts[i+1] for i in range(len(parts)-1))
        
        status = "✅" if (cost_mono and part_mono) else "❌"
        print("  seed=%d: cost↑%s  priv↑%s  part↑%s  %s" % (
            s,
            "✅" if cost_mono else "❌",
            "✅" if priv_mono else "❌", 
            "✅" if part_mono else "❌",
            status))
        print("    costs: %s" % ["%.0f" % c for c in costs])
        print("    privs: %s" % ["%.1f" % p for p in privs])
        print("    parts: %s" % ["%.1f%%" % (p*100) for p in parts])


def main():
    print("=" * 80)
    print("  PAID-FD v10.1 — COMPLETE RESULT ANALYSIS")
    print("  3-seed robustness + Lambda sensitivity sweep")
    print("=" * 80)
    
    sm = load_combined()
    print("  Loaded: %d summaries" % len(sm))
    
    # Count unique configs
    configs = set()
    for key in sm:
        s = sm[key]
        g = s.get("gamma", 0)
        seed = s.get("seed", 42)  # early runs missing seed field
        lm = s.get("lambda_mult", 1.0)  # early 3-seed runs missing lambda_mult
        configs.add((g, seed, lm))
    print("  Unique configs: %d" % len(configs))
    
    gamma_stats = analyze_3seeds(sm)
    lambda_data = analyze_lambda(sm)
    analyze_efficiency(gamma_stats)
    analyze_proposition2(sm)
    
    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
