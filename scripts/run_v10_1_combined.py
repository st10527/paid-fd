#!/usr/bin/env python3
"""
v10.1 Combined Sweep: Resume all missing runs
===============================================

Merges 3-seed robustness + lambda sweep into one job.
Reads existing partial results and skips completed runs.

Total: 10 missing (3-seed) + 11 missing (lambda) = 21 runs
Estimated: ~7 hours on RTX 5070 Ti

Usage:
  nohup python scripts/run_v10_1_combined.py > results/logs/v10_1_combined.log 2>&1 &
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path

from scripts.run_all_experiments import run_single_experiment, save_json


# ============================================================
# Configuration
# ============================================================
DEVICE = "cuda"
N_ROUNDS = 100

# 3-seed: gamma x seed
SEED_RUNS = []
for g in [3, 5, 7, 10]:
    for s in [42, 123, 456]:
        SEED_RUNS.append({"gamma": g, "seed": s, "lambda_mult": 1.0,
                          "label": "g%d_s%d" % (g, s), "group": "3seed"})

# Lambda sweep: lambda_mult x gamma (seed=42)
LAMBDA_RUNS = []
for lm in [0.5, 1.0, 2.0]:
    for g in [3, 5, 7, 10]:
        LAMBDA_RUNS.append({"gamma": g, "seed": 42, "lambda_mult": lm,
                            "label": "lm%s_g%d" % (lm, g), "group": "lambda"})

ALL_RUNS = SEED_RUNS + LAMBDA_RUNS

BASE_METHOD_CONFIG = {
    'delta': 0.01,
    'clip_bound': 2.0,
    'ema_momentum': 0.9,
    'distill_alpha': 0.7,
    'temperature': 3.0,
    'pretrain_epochs': 10,
    'pretrain_lr': 0.1,
    'use_blue': True,
    'use_ema': True,
    'use_mixed_loss': True,
    'use_ldp': True,
    'use_denoising': False,
}


def load_existing():
    """Load already-completed runs from partial result files."""
    completed = {}

    # 3-seed partial
    p1 = Path("results/experiments/v10_1_3seeds_20260409_0922.json")
    if p1.exists():
        with open(p1) as f:
            d = json.load(f)
        for label, summary in d.get("summaries", {}).items():
            completed[label] = summary
            print("  [resume] %s from 3-seed file" % label)

    # Lambda partial
    p2 = Path("results/experiments/v10_1_lambda_20260409_1921.json")
    if p2.exists():
        with open(p2) as f:
            d = json.load(f)
        for label, summary in d.get("summaries", {}).items():
            completed[label] = summary
            print("  [resume] %s from lambda file" % label)

    # v10.1 original (has g3 s42 through g10 s42 at lambda_mult=1.0)
    # These are the same as 3seed g*_s42 runs
    # Already captured above if present in 3-seed file

    return completed


def extract_summary(run, gamma, seed, lm):
    """Extract efficiency metrics from a single run."""
    accs = run["accuracies"]
    extras = run.get("extras", [])

    final_acc = accs[-1] if accs else 0
    best_acc = max(accs) if accs else 0
    cum_payment = extras[-1].get("cumulative_payment", 0) if extras else 0
    avg_part = np.mean(run["participation_rates"]) if run["participation_rates"] else 0
    avg_priv = extras[-1].get("avg_privacy_spent", 0) if extras else 0
    max_priv = extras[-1].get("max_privacy_spent", 0) if extras else 0
    avg_price = np.mean(run["prices"]) if run["prices"] else 0
    avg_eps = np.mean(run["avg_eps"]) if run["avg_eps"] else 0

    energy_total = sum(
        sum(e.values()) for e in run.get("energy_history", []) if isinstance(e, dict)
    )

    targets = [0.45, 0.50, 0.55, 0.58, 0.60]
    ttt = {}
    for target in targets:
        reached = [i for i, a in enumerate(accs) if a >= target]
        ttt[str(target)] = reached[0] if reached else None

    deltas = [e.get("distill_delta", None) for e in extras if e]
    deltas = [d for d in deltas if d is not None]
    avg_delta = float(np.mean(deltas)) if deltas else 0

    return {
        "gamma": gamma,
        "seed": seed,
        "lambda_mult": lm,
        "final_acc": final_acc,
        "best_acc": best_acc,
        "avg_participation": avg_part,
        "avg_price": avg_price,
        "avg_eps_per_round": avg_eps,
        "cumulative_payment": cum_payment,
        "avg_privacy_spent": avg_priv,
        "max_privacy_spent": max_priv,
        "energy_total": energy_total,
        "time_to_targets": ttt,
        "avg_distill_delta": avg_delta,
    }


def deduplicate_runs(all_runs):
    """Remove duplicate (gamma, seed, lambda_mult) combos.
    3-seed g*_s42 at lm=1.0 overlaps with lambda lm1.0_g* s42.
    Keep both labels but only run once.
    """
    seen = {}
    deduped = []
    for run in all_runs:
        key = (run["gamma"], run["seed"], run["lambda_mult"])
        if key in seen:
            # Record the alias label
            seen[key]["aliases"].append(run["label"])
        else:
            run["aliases"] = [run["label"]]
            seen[key] = run
            deduped.append(run)
    return deduped


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    outfile = "results/experiments/v10_1_combined_%s.json" % timestamp

    print("=" * 70)
    print("v10.1 COMBINED SWEEP (resume mode)")
    print("=" * 70)

    # Load existing
    print("\nLoading existing results...")
    completed = load_existing()
    print("  Completed: %d runs" % len(completed))

    # Deduplicate
    deduped = deduplicate_runs(ALL_RUNS)
    print("  Total unique configs: %d" % len(deduped))

    # Figure out what needs to run
    to_run = []
    for run in deduped:
        # Check if any of its labels are already completed
        already_done = False
        for label in run["aliases"]:
            if label in completed:
                already_done = True
                break
        if not already_done:
            to_run.append(run)

    print("  Need to run: %d" % len(to_run))
    print()

    if not to_run:
        print("All runs already completed!")
        return

    # Estimate time
    est_per_run = 20  # minutes
    print("  Estimated time: ~%.1f hours" % (len(to_run) * est_per_run / 60))
    print("=" * 70)

    all_summaries = dict(completed)  # start with existing
    all_runs_data = {}

    t_total = time.time()

    for idx, run_spec in enumerate(to_run):
        gamma = run_spec["gamma"]
        seed = run_spec["seed"]
        lm = run_spec["lambda_mult"]
        labels = run_spec["aliases"]
        primary_label = labels[0]

        print("\n" + "=" * 60)
        print("  RUN %d/%d: gamma=%d seed=%d lambda_mult=%.1f" % (
            idx + 1, len(to_run), gamma, seed, lm))
        print("  Labels: %s" % ", ".join(labels))
        print("=" * 60)

        config = {
            'n_devices': 50,
            'gamma': gamma,
            'alpha': 0.5,
            'local_epochs': 5,
            'local_lr': 0.01,
            'local_momentum': 0.9,
            'distill_epochs': 1,
            'distill_lr': 0.001,
            'temperature': 3.0,
            'public_samples': 20000,
            'synthetic': False,
            'heterogeneity': {
                'config_file': 'config/devices/heterogeneity.yaml',
                'overrides': {
                    'privacy_sensitivity': {
                        'lambda_mult': lm
                    }
                }
            },
            'method_config': {**BASE_METHOD_CONFIG, 'gamma': gamma},
        }

        t0 = time.time()
        run = run_single_experiment(
            method_name='PAID-FD',
            config=config,
            seed=seed,
            device=DEVICE,
            n_rounds=N_ROUNDS,
            save_decisions=False,
            verbose=True,
        )
        elapsed = time.time() - t0

        summary = extract_summary(run, gamma, seed, lm)
        summary["elapsed_sec"] = elapsed

        # Store under all alias labels
        for label in labels:
            all_summaries[label] = summary

        all_runs_data[primary_label] = run

        print("\n  [%s] best=%.2f%% final=%.2f%% part=%.0f%% cost=%.0f time=%.0fs" % (
            primary_label, summary["best_acc"] * 100, summary["final_acc"] * 100,
            summary["avg_participation"] * 100, summary["cumulative_payment"], elapsed))

        # Crash-safe save after every run
        save_data = {
            "experiment": "v10_1_combined",
            "timestamp": timestamp,
            "summaries": all_summaries,
            # Don't save full run data each time (too large), only summaries
        }
        save_json(save_data, outfile)
        print("  Saved checkpoint (%d/%d summaries)" % (len(all_summaries), len(ALL_RUNS)))

    # ============================================================
    # FINAL ANALYSIS
    # ============================================================
    elapsed_total = time.time() - t_total

    print("\n" + "=" * 70)
    print("ALL RUNS COMPLETE — FINAL ANALYSIS")
    print("=" * 70)

    # ---- 3-SEED AGGREGATION ----
    print("\n--- 3-SEED ROBUSTNESS (mean +/- std) ---")
    print("%4s | %20s | %22s | %18s" % ("g", "Best Acc", "Cum Payment", "Max Privacy"))
    print("-" * 70)
    for g in [3, 5, 7, 10]:
        bests, costs, privs = [], [], []
        for s in [42, 123, 456]:
            key = "g%d_s%d" % (g, s)
            if key in all_summaries:
                bests.append(all_summaries[key]["best_acc"])
                costs.append(all_summaries[key]["cumulative_payment"])
                privs.append(all_summaries[key]["max_privacy_spent"])
        if bests:
            print("  %2d | %6.2f +/- %4.2f%%    | %8.0f +/- %7.0f    | %6.1f +/- %5.1f" % (
                g, np.mean(bests)*100, np.std(bests)*100,
                np.mean(costs), np.std(costs),
                np.mean(privs), np.std(privs)))

    # ---- LAMBDA SWEEP ----
    print("\n--- LAMBDA SWEEP ---")
    print("%4s %4s | %7s %6s %10s %8s %7s" % ("lm", "g", "Best%", "Part%", "CumCost", "MaxPriv", "Price"))
    print("-" * 55)
    for lm in [0.5, 1.0, 2.0]:
        for g in [3, 5, 7, 10]:
            key = "lm%s_g%d" % (lm, g)
            if key in all_summaries:
                s = all_summaries[key]
                print("%4s %4d | %6.2f%% %5.0f%% %10.0f %8.1f %7.3f" % (
                    lm, g, s["best_acc"]*100, s["avg_participation"]*100,
                    s["cumulative_payment"], s["max_privacy_spent"], s["avg_price"]))
        print("-" * 55)

    # ---- EFFICIENCY CRITERIA (3-seed, gamma 3 vs 10) ----
    print("\n--- EFFICIENCY CRITERIA (3-seed validated) ---")
    g3_bests = [all_summaries["g3_s%d" % s]["best_acc"] for s in [42, 123, 456]
                if "g3_s%d" % s in all_summaries]
    g10_bests = [all_summaries["g10_s%d" % s]["best_acc"] for s in [42, 123, 456]
                 if "g10_s%d" % s in all_summaries]
    g3_costs = [all_summaries["g3_s%d" % s]["cumulative_payment"] for s in [42, 123, 456]
                if "g3_s%d" % s in all_summaries]
    g10_costs = [all_summaries["g10_s%d" % s]["cumulative_payment"] for s in [42, 123, 456]
                 if "g10_s%d" % s in all_summaries]
    g3_privs = [all_summaries["g3_s%d" % s]["max_privacy_spent"] for s in [42, 123, 456]
                if "g3_s%d" % s in all_summaries]
    g10_privs = [all_summaries["g10_s%d" % s]["max_privacy_spent"] for s in [42, 123, 456]
                 if "g10_s%d" % s in all_summaries]
    g3_parts = [all_summaries["g3_s%d" % s]["avg_participation"] for s in [42, 123, 456]
                if "g3_s%d" % s in all_summaries]
    g10_parts = [all_summaries["g10_s%d" % s]["avg_participation"] for s in [42, 123, 456]
                 if "g10_s%d" % s in all_summaries]

    if g3_bests and g10_bests:
        all_gamma_bests = []
        for g in [3, 5, 7, 10]:
            vals = [all_summaries["g%d_s%d" % (g, s)]["best_acc"]
                    for s in [42, 123, 456] if "g%d_s%d" % (g, s) in all_summaries]
            if vals:
                all_gamma_bests.append(np.mean(vals))

        spread = max(all_gamma_bests) - min(all_gamma_bests) if all_gamma_bests else 0
        cost_r = np.mean(g10_costs) / np.mean(g3_costs) if g3_costs else 0
        priv_r = np.mean(g10_privs) / np.mean(g3_privs) if g3_privs else 0
        part_s = np.mean(g10_parts) - np.mean(g3_parts) if g3_parts else 0
        best_a = max(all_gamma_bests) if all_gamma_bests else 0

        print("  E1 Acc spread:    %.2f%%  %s" % (spread*100, "PASS" if spread < 0.02 else "FAIL"))
        print("  E2 Cost ratio:    %.2fx  %s" % (cost_r, "PASS" if cost_r > 2 else "FAIL"))
        print("  E3 Privacy ratio: %.2fx  %s" % (priv_r, "PASS" if priv_r > 1.5 else "FAIL"))
        print("  E4 Part spread:   %.0f%%   %s" % (part_s*100, "PASS" if part_s > 0.1 else "FAIL"))
        print("  E5 Best accuracy: %.2f%%  %s" % (best_a*100, "PASS" if best_a > 0.5 else "FAIL"))

    print("\n  Total wall time: %.1f hours" % (elapsed_total / 3600))
    print("  Results: %s" % outfile)

    # Final save with full data
    save_data["completed_at"] = datetime.now().isoformat()
    save_data["elapsed_total_sec"] = elapsed_total
    save_json(save_data, outfile)


if __name__ == "__main__":
    main()
