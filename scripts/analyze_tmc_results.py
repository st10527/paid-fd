#!/usr/bin/env python3
"""
TMC Paper — Results Analyzer
==============================
Reads individual experiment JSONs from results/experiments/tmc/
and produces paper tables and figures.

Can run with partial results — shows what's available.

Usage:
  python scripts/analyze_tmc_results.py                     # Full report
  python scripts/analyze_tmc_results.py --phase 1           # Phase 1 only
  python scripts/analyze_tmc_results.py --exp A             # Exp A only
  python scripts/analyze_tmc_results.py --progress          # Quick progress check
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import glob
import numpy as np
from collections import defaultdict
from pathlib import Path

RESULT_DIR = "results/experiments/tmc"
V101_RESULT = "results/experiments/v10_1_combined_20260409_2304.json"
FIGURE_DIR = "results/figures/tmc"


# ============================================================
# Load results
# ============================================================

def load_all_results():
    """Load all tmc_*.json result files."""
    results = {}
    for f in sorted(glob.glob(os.path.join(RESULT_DIR, "exp*.json"))):
        with open(f) as fp:
            data = json.load(fp)
        label = data.get("label", Path(f).stem)
        results[label] = data
    return results


def load_v101_baselines():
    """Load existing v10.1 PAID-FD results as baselines."""
    if not os.path.exists(V101_RESULT):
        print("[WARN] v10.1 baseline not found: %s" % V101_RESULT)
        return {}
    with open(V101_RESULT) as f:
        data = json.load(f)
    return data


def group_by_method_seed(results, exp_filter=None):
    """Group results by (method, key) → list of {seed, acc, ...}."""
    groups = defaultdict(list)
    for label, r in results.items():
        if exp_filter and r.get("exp") != exp_filter:
            continue
        method = r.get("method", "?")
        seed = r.get("seed", 0)
        accs = r.get("accuracies", [])
        summary = r.get("summary", {})
        groups[method].append({
            "label": label,
            "seed": seed,
            "best_acc": max(accs) if accs else 0,
            "final_acc": accs[-1] if accs else 0,
            "avg_part": summary.get("avg_participation", 0),
            "cum_payment": summary.get("cumulative_payment", 0),
            "avg_eps": summary.get("avg_eps_per_round", 0),
            "max_priv": summary.get("max_privacy_spent", 0),
            "elapsed": r.get("elapsed_sec", 0),
        })
    return groups


# ============================================================
# Progress report
# ============================================================

def show_progress(results):
    """Quick progress check: how many runs completed per experiment."""
    # Expected counts
    expected = {
        "A": 9, "A'": 3, "B": 12, "C": 9,  # Phase 1: 33
        "D": 9,                                 # Phase 2: 9
        "E": 12,                                # Phase 3: 12
    }

    print("\n" + "=" * 60)
    print("  TMC Experiment Progress")
    print("=" * 60)

    exp_counts = defaultdict(int)
    for label, r in results.items():
        exp_counts[r.get("exp", "?")] += 1

    total_done = 0
    total_expected = 54
    for exp in ["A", "A'", "B", "C", "D", "E"]:
        done = exp_counts.get(exp, 0)
        want = expected.get(exp, 0)
        pct = done / want * 100 if want > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        phase = {
            "A": 1, "A'": 1, "B": 1, "C": 1, "D": 2, "E": 3
        }.get(exp, "?")
        print("  Exp %-2s [Phase %s]  %s  %d/%d  (%.0f%%)" % (
            exp, phase, bar, done, want, pct))
        total_done += done

    print("-" * 60)
    pct = total_done / total_expected * 100
    print("  TOTAL: %d/%d (%.0f%%)" % (total_done, total_expected, pct))
    print("=" * 60)


# ============================================================
# Exp A: Method comparison table (Table II in paper)
# ============================================================

def analyze_exp_a(results):
    """
    Table II: Privacy-Preserving Method Comparison
    
    PAID-FD (γ=3,5,7,10) vs Fixed-ε-1 vs Fixed-ε-3 vs CSRA
    All with LDP on CIFAR-100, N=50, α=0.5
    """
    print("\n" + "=" * 70)
    print("  Exp A: Privacy-Preserving Method Comparison (Table II)")
    print("=" * 70)

    groups = group_by_method_seed(results, exp_filter="A")

    # Add v10.1 PAID-FD baselines
    v101 = load_v101_baselines()
    if v101:
        print("  (Including v10.1 PAID-FD baselines)")

    print("\n  %-16s  %7s  %7s  %8s  %10s  %6s" % (
        "Method", "Best%", "±Std", "AvgPart%", "TotalCost", "AvgEps"))
    print("  " + "-" * 62)

    for method in sorted(groups.keys()):
        runs = groups[method]
        if len(runs) == 0:
            continue
        best_accs = [r["best_acc"] * 100 for r in runs]
        parts = [r["avg_part"] * 100 for r in runs]
        costs = [r["cum_payment"] for r in runs]
        eps_list = [r["avg_eps"] for r in runs]

        print("  %-16s  %6.2f  %6.2f  %7.1f  %10.0f  %5.2f  [n=%d]" % (
            method,
            np.mean(best_accs), np.std(best_accs),
            np.mean(parts), np.mean(costs), np.mean(eps_list),
            len(runs)))

    print()


# ============================================================
# Exp A': No-privacy reference (Table II footer)
# ============================================================

def analyze_exp_ap(results):
    """No-privacy reference methods (single seed, upper bound)."""
    print("\n  Exp A': No-Privacy Reference (Table II footer)")
    print("  " + "-" * 50)

    groups = group_by_method_seed(results, exp_filter="A'")

    for method in sorted(groups.keys()):
        runs = groups[method]
        for r in runs:
            print("  %-16s  best=%.2f%%  final=%.2f%%  (seed=%d)" % (
                method, r["best_acc"] * 100, r["final_acc"] * 100, r["seed"]))


# ============================================================
# Exp B: Scalability analysis (Table III / Fig 5)
# ============================================================

def analyze_exp_b(results):
    """N sweep: how does PAID-FD scale with N={20,50,80}?"""
    print("\n" + "=" * 70)
    print("  Exp B: Scalability (N Sweep) — Table III / Fig 5")
    print("=" * 70)

    b_results = {k: v for k, v in results.items() if v.get("exp") == "B"}

    # Parse N and gamma from label: expB_n20_g3_s42
    table = defaultdict(lambda: defaultdict(list))
    for label, r in b_results.items():
        parts = label.split("_")
        n_val = int(parts[1][1:])    # "n20" -> 20
        g_val = int(parts[2][1:])    # "g3"  -> 3
        accs = r.get("accuracies", [])
        table[(n_val, g_val)]["acc"].append(max(accs) if accs else 0)
        summary = r.get("summary", {})
        table[(n_val, g_val)]["cost"].append(summary.get("cumulative_payment", 0))
        table[(n_val, g_val)]["part"].append(summary.get("avg_participation", 0))

    print("\n  %4s  %4s  %7s  %6s  %8s  %8s  n_seeds" % (
        "N", "γ", "Best%", "±Std", "AvgPart%", "Cost"))
    print("  " + "-" * 55)

    for (n, g) in sorted(table.keys()):
        accs = np.array(table[(n, g)]["acc"]) * 100
        costs = table[(n, g)]["cost"]
        parts = np.array(table[(n, g)]["part"]) * 100
        print("  %4d  %4d  %6.2f  %5.2f  %7.1f  %8.0f  %d" % (
            n, g, np.mean(accs), np.std(accs),
            np.mean(parts), np.mean(costs), len(accs)))

    # Include N=50 from v10.1 (note to reader)
    print("\n  Note: N=50 baseline from v10.1 (γ={3,5,7,10} × 3 seeds)")


# ============================================================
# Exp C: Ablation study (Table IV)
# ============================================================

def analyze_exp_c(results):
    """Ablation: remove one component at a time from PAID-FD."""
    print("\n" + "=" * 70)
    print("  Exp C: Ablation Study — Table IV")
    print("=" * 70)

    c_results = {k: v for k, v in results.items() if v.get("exp") == "C"}

    # Group by ablation type
    ablation_types = defaultdict(list)
    for label, r in c_results.items():
        if "noblue" in label:
            atype = "BLUE off (uniform agg)"
        elif "fullpart" in label:
            atype = "Full participation (γ=100)"
        elif "noldp" in label:
            atype = "Oracle (no LDP)"
        else:
            atype = "unknown"
        accs = r.get("accuracies", [])
        summary = r.get("summary", {})
        ablation_types[atype].append({
            "best_acc": max(accs) * 100 if accs else 0,
            "cost": summary.get("cumulative_payment", 0),
            "part": summary.get("avg_participation", 0) * 100,
        })

    print("\n  %-30s  %7s  %6s  %8s  %8s" % (
        "Variant", "Best%", "±Std", "Part%", "Cost"))
    print("  " + "-" * 65)

    # v10.1 baseline (γ=5) for comparison
    print("  %-30s  %6.2f  %5.2f  %7.1f  %8.0f  (baseline)" % (
        "PAID-FD v10.1 (γ=5)", 61.43, 0.32, 79.3, 30178))

    for atype in ["BLUE off (uniform agg)", "Full participation (γ=100)", "Oracle (no LDP)"]:
        runs = ablation_types.get(atype, [])
        if not runs:
            print("  %-30s  %s" % (atype, "(not yet run)"))
            continue
        accs = [r["best_acc"] for r in runs]
        costs = [r["cost"] for r in runs]
        parts = [r["part"] for r in runs]
        print("  %-30s  %6.2f  %5.2f  %7.1f  %8.0f  [n=%d]" % (
            atype,
            np.mean(accs), np.std(accs),
            np.mean(parts), np.mean(costs), len(runs)))


# ============================================================
# Exp D: CIFAR-10 (Table V)
# ============================================================

def analyze_exp_d(results):
    """CIFAR-10 cross-dataset validation."""
    print("\n" + "=" * 70)
    print("  Exp D: CIFAR-10 Validation — Table V")
    print("=" * 70)

    groups = group_by_method_seed(results, exp_filter="D")

    print("\n  %-16s  %7s  %6s  n_seeds" % ("Method", "Best%", "±Std"))
    print("  " + "-" * 40)

    for method in sorted(groups.keys()):
        runs = groups[method]
        best_accs = [r["best_acc"] * 100 for r in runs]
        print("  %-16s  %6.2f  %5.2f  %d" % (
            method, np.mean(best_accs), np.std(best_accs), len(runs)))


# ============================================================
# Exp E: Non-IID α sweep (Fig 6)
# ============================================================

def analyze_exp_e(results):
    """Dirichlet α sensitivity analysis."""
    print("\n" + "=" * 70)
    print("  Exp E: Non-IID α Sweep — Fig 6")
    print("=" * 70)

    e_results = {k: v for k, v in results.items() if v.get("exp") == "E"}

    table = defaultdict(lambda: defaultdict(list))
    for label, r in e_results.items():
        # Parse: expE_a01_g3_s42
        parts = label.split("_")
        a_str = parts[1]  # "a01" or "a10"
        g_val = int(parts[2][1:])
        alpha = float(a_str[1:]) / 10  # "01" -> 0.1, "10" -> 1.0
        accs = r.get("accuracies", [])
        table[(alpha, g_val)]["acc"].append(max(accs) if accs else 0)

    print("\n  %5s  %4s  %7s  %6s  n_seeds" % ("α", "γ", "Best%", "±Std"))
    print("  " + "-" * 40)

    for (alpha, g) in sorted(table.keys()):
        accs = np.array(table[(alpha, g)]["acc"]) * 100
        print("  %5.1f  %4d  %6.2f  %5.2f  %d" % (
            alpha, g, np.mean(accs), np.std(accs), len(accs)))

    print("\n  Note: α=0.5 baseline from v10.1")


# ============================================================
# Efficiency criteria check (Proposition 2)
# ============================================================

def check_efficiency(results):
    """Check if efficiency criteria are met across experiments."""
    print("\n" + "=" * 70)
    print("  Efficiency Criteria Check (Proposition 2)")
    print("=" * 70)

    paid_fd_runs = {k: v for k, v in results.items()
                    if v.get("method") == "PAID-FD" and v.get("exp") in ["B", "E"]}

    if not paid_fd_runs:
        print("  No PAID-FD runs to check yet.")
        return

    for label, r in sorted(paid_fd_runs.items()):
        accs = r.get("accuracies", [])
        extras = r.get("summary", {})
        parts = r.get("participation_rates", [])
        if not accs or not parts:
            continue

        # Check monotonic accuracy (non-decreasing trend)
        increasing_pairs = sum(1 for i in range(1, len(accs)) if accs[i] >= accs[i-1])
        mono_pct = increasing_pairs / (len(accs) - 1) * 100 if len(accs) > 1 else 0

        print("  %-30s  best=%.2f%%  mono=%.0f%%  part=%.0f%%" % (
            label, max(accs) * 100, mono_pct,
            np.mean(parts) * 100))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="TMC Results Analyzer")
    parser.add_argument("--phase", type=int, default=None, choices=[1, 2, 3])
    parser.add_argument("--exp", type=str, default=None)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    results = load_all_results()
    print("Loaded %d result files from %s" % (len(results), RESULT_DIR))

    if args.progress:
        show_progress(results)
        return

    # Filter by phase if requested
    if args.phase:
        phase_map = {
            1: ["A", "A'", "B", "C"],
            2: ["D"],
            3: ["E"],
        }
        allowed = phase_map.get(args.phase, [])
        results = {k: v for k, v in results.items()
                   if v.get("exp") in allowed}
        print("Filtered to Phase %d: %d results" % (args.phase, len(results)))

    # Filter by experiment
    if args.exp:
        results = {k: v for k, v in results.items()
                   if v.get("exp") == args.exp}
        print("Filtered to Exp %s: %d results" % (args.exp, len(results)))

    show_progress(results if not args.phase and not args.exp
                  else load_all_results())

    # Run applicable analyses
    exps_present = set(r.get("exp") for r in results.values())

    if "A" in exps_present:
        analyze_exp_a(results)
    if "A'" in exps_present:
        analyze_exp_ap(results)
    if "B" in exps_present:
        analyze_exp_b(results)
    if "C" in exps_present:
        analyze_exp_c(results)
    if "D" in exps_present:
        analyze_exp_d(results)
    if "E" in exps_present:
        analyze_exp_e(results)

    if "B" in exps_present or "E" in exps_present:
        check_efficiency(results)

    print("\n[Done] Use --progress for quick status check.")


if __name__ == "__main__":
    main()
