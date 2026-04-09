#!/usr/bin/env python3
"""
v10.1 Three-Seed Robustness Validation
========================================

Purpose: Validate that v10.1 efficiency story is robust across seeds.
  - γ ∈ {3, 5, 7, 10}  ×  seed ∈ {42, 123, 456}  =  12 runs
  - Output: per-seed results + aggregated mean±std for each γ

This produces the data for TMC paper Fig 3 (with error bars).

Expected wall time: ~4 hours on RTX 5070 Ti (~20 min per run × 12 runs)

Usage:
  nohup python scripts/run_v10_1_3seeds.py > results/logs/v10_1_3seeds.log 2>&1 &
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
from datetime import datetime

from scripts.run_all_experiments import run_single_experiment, save_json


# ============================================================
# Configuration
# ============================================================
GAMMA_VALUES = [3, 5, 7, 10]
SEEDS = [42, 123, 456]
N_ROUNDS = 100
DEVICE = "cuda"

BASE_CONFIG = {
    'n_devices': 50,
    'alpha': 0.5,          # Dirichlet non-IID
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
    },
    'method_config': {
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
}


def extract_summary(run, gamma):
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
    
    # Time to targets
    targets = [0.45, 0.50, 0.55, 0.58, 0.60]
    ttt = {}
    for target in targets:
        reached = [i for i, a in enumerate(accs) if a >= target]
        ttt[str(target)] = reached[0] if reached else None
    
    # Distill delta
    deltas = [e.get("distill_delta", None) for e in extras if e]
    deltas = [d for d in deltas if d is not None]
    avg_delta = float(np.mean(deltas)) if deltas else 0
    
    # CE loss trend
    ce_first = extras[0].get("mean_loss_ce", None) if extras else None
    ce_last = extras[-1].get("mean_loss_ce", None) if extras else None
    
    return {
        "gamma": gamma,
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
        "ce_first": ce_first,
        "ce_last": ce_last,
    }


def run_3seeds():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    total_runs = len(GAMMA_VALUES) * len(SEEDS)
    
    print("=" * 70)
    print("v10.1 THREE-SEED ROBUSTNESS VALIDATION")
    print("=" * 70)
    print(f"  γ values: {GAMMA_VALUES}")
    print(f"  Seeds:    {SEEDS}")
    print(f"  Rounds:   {N_ROUNDS}")
    print(f"  Device:   {DEVICE}")
    print(f"  Total:    {total_runs} runs")
    print(f"  Timestamp: {timestamp}")
    print("=" * 70)
    
    all_results = {
        "experiment": "v10_1_3seeds",
        "timestamp": timestamp,
        "architecture": "v10.1_persistent_adam",
        "gamma_values": GAMMA_VALUES,
        "seeds": SEEDS,
        "n_rounds": N_ROUNDS,
        "runs": {},          # key: "g{gamma}_s{seed}"
        "summaries": {},     # key: "g{gamma}_s{seed}"
        "aggregated": {},    # key: "g{gamma}" -> mean±std across seeds
    }
    
    run_idx = 0
    t_total = time.time()
    
    for gamma in GAMMA_VALUES:
        for seed in SEEDS:
            run_idx += 1
            label = f"g{gamma}_s{seed}"
            
            print(f"\n{'='*60}")
            print(f"  RUN {run_idx}/{total_runs}: γ={gamma}, seed={seed} ({label})")
            print(f"{'='*60}")
            
            config = {**BASE_CONFIG}
            config['gamma'] = gamma
            config['method_config'] = {**BASE_CONFIG['method_config'], 'gamma': gamma}
            
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
            
            run["label"] = label
            run["gamma"] = gamma
            run["seed"] = seed
            run["elapsed_sec"] = elapsed
            
            summary = extract_summary(run, gamma)
            summary["seed"] = seed
            summary["elapsed_sec"] = elapsed
            
            all_results["runs"][label] = run
            all_results["summaries"][label] = summary
            
            print(f"\n  [{label}] final={summary['final_acc']*100:.2f}% "
                  f"best={summary['best_acc']*100:.2f}% "
                  f"part={summary['avg_participation']:.0%} "
                  f"cost={summary['cumulative_payment']:.0f} "
                  f"time={elapsed:.0f}s")
            
            # Save after every run (crash safety)
            save_json(all_results, f"results/experiments/v10_1_3seeds_{timestamp}.json")
    
    # ============================================================
    # AGGREGATE: mean ± std across seeds for each γ
    # ============================================================
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS (mean ± std across 3 seeds)")
    print("=" * 70)
    
    agg = {}
    for gamma in GAMMA_VALUES:
        seed_summaries = [all_results["summaries"][f"g{gamma}_s{s}"] for s in SEEDS]
        
        metrics = {}
        for key in ["final_acc", "best_acc", "avg_participation", "avg_price",
                     "avg_eps_per_round", "cumulative_payment", "avg_privacy_spent",
                     "max_privacy_spent", "energy_total", "avg_distill_delta"]:
            vals = [s[key] for s in seed_summaries]
            metrics[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        
        # Time-to-targets aggregation
        targets = ["0.45", "0.5", "0.55", "0.58", "0.6"]
        ttt_agg = {}
        for t in targets:
            vals = [s["time_to_targets"].get(t, None) for s in seed_summaries]
            vals = [v for v in vals if v is not None]
            if vals:
                ttt_agg[t] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                              "n_reached": len(vals)}
        metrics["time_to_targets"] = ttt_agg
        
        agg[f"g{gamma}"] = metrics
        all_results["aggregated"][f"g{gamma}"] = metrics
    
    # ---- Table 1: Accuracy ----
    print(f"\n{'γ':>4} | {'Final Acc':>16} | {'Best Acc':>16} | {'Distill Δ':>16}")
    print("-" * 65)
    for gamma in GAMMA_VALUES:
        m = agg[f"g{gamma}"]
        fa = m["final_acc"]
        ba = m["best_acc"]
        dd = m["avg_distill_delta"]
        print(f"{gamma:4d} | {fa['mean']*100:6.2f} ± {fa['std']*100:4.2f}% | "
              f"{ba['mean']*100:6.2f} ± {ba['std']*100:4.2f}% | "
              f"{dd['mean']*100:+6.4f} ± {dd['std']*100:5.4f}%")
    
    # ---- Table 2: Efficiency ----
    print(f"\n{'γ':>4} | {'Participation':>16} | {'Cum Payment':>20} | {'Max Priv':>16}")
    print("-" * 75)
    for gamma in GAMMA_VALUES:
        m = agg[f"g{gamma}"]
        p = m["avg_participation"]
        c = m["cumulative_payment"]
        pr = m["max_privacy_spent"]
        print(f"{gamma:4d} | {p['mean']*100:5.1f} ± {p['std']*100:4.1f}%   | "
              f"{c['mean']:8.0f} ± {c['std']:7.0f}    | "
              f"{pr['mean']:6.1f} ± {pr['std']:5.1f}")
    
    # ---- Table 3: Time-to-target ----
    print(f"\n{'γ':>4} | {'@50%':>12} | {'@55%':>12} | {'@58%':>12} | {'@60%':>12}")
    print("-" * 60)
    for gamma in GAMMA_VALUES:
        m = agg[f"g{gamma}"]
        ttt = m["time_to_targets"]
        vals = []
        for t in ["0.5", "0.55", "0.58", "0.6"]:
            if t in ttt:
                vals.append(f"{ttt[t]['mean']:5.1f}±{ttt[t]['std']:4.1f}")
            else:
                vals.append("    N/A  ")
        print(f"{gamma:4d} | " + " | ".join(vals))
    
    # ---- Efficiency criteria with error bars ----
    print(f"\n{'='*70}")
    print("EFFICIENCY CRITERIA (with confidence)")
    print(f"{'='*70}")
    
    g3, g10 = agg["g3"], agg["g10"]
    
    # E1: Accuracy spread
    all_best_means = [agg[f"g{g}"]["best_acc"]["mean"] for g in GAMMA_VALUES]
    acc_spread = max(all_best_means) - min(all_best_means)
    print(f"  E1 Acc spread:    {acc_spread*100:.2f}%  {'PASS' if acc_spread < 0.02 else 'FAIL'}")
    
    # E2: Cost ratio
    cost_lo = g3["cumulative_payment"]["mean"]
    cost_hi = g10["cumulative_payment"]["mean"]
    cost_ratio = cost_hi / cost_lo if cost_lo > 0 else 0
    # Error propagation for ratio
    cost_lo_std = g3["cumulative_payment"]["std"]
    cost_hi_std = g10["cumulative_payment"]["std"]
    cost_ratio_err = cost_ratio * np.sqrt((cost_lo_std/cost_lo)**2 + (cost_hi_std/cost_hi)**2) if cost_lo > 0 and cost_hi > 0 else 0
    print(f"  E2 Cost ratio:    {cost_ratio:.2f} ± {cost_ratio_err:.2f}x  {'PASS' if cost_ratio > 2 else 'FAIL'}")
    
    # E3: Privacy ratio
    priv_lo = g3["max_privacy_spent"]["mean"]
    priv_hi = g10["max_privacy_spent"]["mean"]
    priv_ratio = priv_hi / priv_lo if priv_lo > 0 else 0
    priv_lo_std = g3["max_privacy_spent"]["std"]
    priv_hi_std = g10["max_privacy_spent"]["std"]
    priv_ratio_err = priv_ratio * np.sqrt((priv_lo_std/priv_lo)**2 + (priv_hi_std/priv_hi)**2) if priv_lo > 0 and priv_hi > 0 else 0
    print(f"  E3 Privacy ratio: {priv_ratio:.2f} ± {priv_ratio_err:.2f}x  {'PASS' if priv_ratio > 1.5 else 'FAIL'}")
    
    # E4: Participation spread
    part_lo = g3["avg_participation"]["mean"]
    part_hi = g10["avg_participation"]["mean"]
    part_spread = part_hi - part_lo
    part_spread_err = np.sqrt(g3["avg_participation"]["std"]**2 + g10["avg_participation"]["std"]**2)
    print(f"  E4 Part spread:   {part_spread*100:.0f} ± {part_spread_err*100:.0f}%  {'PASS' if part_spread > 0.1 else 'FAIL'}")
    
    # E5: Best accuracy
    best_any = max(all_best_means)
    best_any_std = max(agg[f"g{g}"]["best_acc"]["std"] for g in GAMMA_VALUES)
    print(f"  E5 Best accuracy: {best_any*100:.2f} ± {best_any_std*100:.2f}%  {'PASS' if best_any > 0.5 else 'FAIL'}")
    
    score = sum([acc_spread < 0.02, cost_ratio > 2, priv_ratio > 1.5,
                 part_spread > 0.1, best_any > 0.5])
    print(f"\n  SCORE: {score}/5")
    
    # ---- Proposition 2 monotonicity (all seeds) ----
    print(f"\n{'='*70}")
    print("PROPOSITION 2 MONOTONICITY (mean across seeds)")
    print(f"{'='*70}")
    for metric_name, metric_key in [("price p*", "avg_price"),
                                      ("participation", "avg_participation"),
                                      ("cumulative cost", "cumulative_payment")]:
        vals = [agg[f"g{g}"][metric_key]["mean"] for g in GAMMA_VALUES]
        diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
        mono = all(d > 0 for d in diffs)
        vals_str = " -> ".join(f"{v:.2f}" for v in vals)
        print(f"  {metric_name:<18}: {vals_str}  [{'MONO' if mono else 'NOT MONO'}]")
    
    # Final save
    save_json(all_results, f"results/experiments/v10_1_3seeds_{timestamp}.json")
    
    elapsed_total = time.time() - t_total
    print(f"\n  Total wall time: {elapsed_total/3600:.1f} hours")
    print(f"  Results saved: results/experiments/v10_1_3seeds_{timestamp}.json")
    
    return all_results


if __name__ == "__main__":
    run_3seeds()
