#!/usr/bin/env python3
"""
v10.1 Lambda-mult Sensitivity Sweep
=====================================

Purpose: Show that different user populations (λ_mult) create
different efficiency frontiers. This is TMC paper Fig 4.

  λ_mult=0.5 → privacy-relaxed population (lower privacy cost)
  λ_mult=1.0 → baseline population
  λ_mult=2.0 → privacy-strict population (higher privacy cost)

For each λ_mult, sweep γ ∈ {3, 5, 7, 10} to trace the frontier.
Total: 3 × 4 = 12 runs, seed=42 (single seed, for shape validation).

Expected: higher λ_mult → same accuracy but higher cost/privacy.
Each λ_mult traces a separate Pareto frontier.

Usage:
  nohup python scripts/run_v10_1_lambda_sweep.py > results/logs/v10_1_lambda.log 2>&1 &
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
LAMBDA_MULTS = [0.5, 1.0, 2.0]
GAMMA_VALUES = [3, 5, 7, 10]
SEED = 42
N_ROUNDS = 100
DEVICE = "cuda"

BASE_CONFIG = {
    'n_devices': 50,
    'alpha': 0.5,
    'local_epochs': 5,
    'local_lr': 0.01,
    'local_momentum': 0.9,
    'distill_epochs': 1,
    'distill_lr': 0.001,
    'temperature': 3.0,
    'public_samples': 20000,
    'synthetic': False,
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


def extract_summary(run, gamma, lm):
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
    
    return {
        "gamma": gamma,
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
    }


def run_lambda_sweep():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    total_runs = len(LAMBDA_MULTS) * len(GAMMA_VALUES)
    
    print("=" * 70)
    print("v10.1 LAMBDA-MULT SENSITIVITY SWEEP")
    print("=" * 70)
    print(f"  lambda_mult: {LAMBDA_MULTS}")
    print(f"  gamma:       {GAMMA_VALUES}")
    print(f"  seed:        {SEED}")
    print(f"  rounds:      {N_ROUNDS}")
    print(f"  total runs:  {total_runs}")
    print("=" * 70)
    
    all_results = {
        "experiment": "v10_1_lambda_sweep",
        "timestamp": timestamp,
        "lambda_mults": LAMBDA_MULTS,
        "gamma_values": GAMMA_VALUES,
        "seed": SEED,
        "runs": {},
        "summaries": {},
    }
    
    run_idx = 0
    t_total = time.time()
    
    for lm in LAMBDA_MULTS:
        for gamma in GAMMA_VALUES:
            run_idx += 1
            label = f"lm{lm}_g{gamma}"
            
            print(f"\n{'='*60}")
            print(f"  RUN {run_idx}/{total_runs}: lambda_mult={lm}, gamma={gamma}")
            print(f"{'='*60}")
            
            config = {**BASE_CONFIG}
            config['gamma'] = gamma
            config['method_config'] = {**BASE_CONFIG['method_config'], 'gamma': gamma}
            config['heterogeneity'] = {
                'config_file': 'config/devices/heterogeneity.yaml',
                'overrides': {
                    'privacy_sensitivity': {
                        'lambda_mult': lm
                    }
                }
            }
            
            t0 = time.time()
            run = run_single_experiment(
                method_name='PAID-FD',
                config=config,
                seed=SEED,
                device=DEVICE,
                n_rounds=N_ROUNDS,
                save_decisions=False,
                verbose=True,
            )
            elapsed = time.time() - t0
            
            run["label"] = label
            run["gamma"] = gamma
            run["lambda_mult"] = lm
            run["elapsed_sec"] = elapsed
            
            summary = extract_summary(run, gamma, lm)
            summary["elapsed_sec"] = elapsed
            
            all_results["runs"][label] = run
            all_results["summaries"][label] = summary
            
            print(f"\n  [{label}] final={summary['final_acc']*100:.2f}% "
                  f"best={summary['best_acc']*100:.2f}% "
                  f"part={summary['avg_participation']:.0%} "
                  f"cost={summary['cumulative_payment']:.0f} "
                  f"time={elapsed:.0f}s")
            
            save_json(all_results, f"results/experiments/v10_1_lambda_{timestamp}.json")
    
    # ============================================================
    # ANALYSIS: Compare frontiers across lambda_mult
    # ============================================================
    print("\n" + "=" * 70)
    print("LAMBDA SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    # Table: each row = (lambda_mult, gamma), columns = metrics
    print(f"\n{'lm':>4} {'g':>3} | {'Best%':>7} {'Part%':>6} {'CumCost':>10} {'MaxPriv':>8} {'Price':>7}")
    print("-" * 55)
    for lm in LAMBDA_MULTS:
        for gamma in GAMMA_VALUES:
            s = all_results["summaries"][f"lm{lm}_g{gamma}"]
            print(f"{lm:>4} {gamma:>3} | "
                  f"{s['best_acc']*100:>6.2f}% {s['avg_participation']*100:>5.0f}% "
                  f"{s['cumulative_payment']:>10,.0f} {s['max_privacy_spent']:>8.1f} "
                  f"{s['avg_price']:>7.3f}")
        print("-" * 55)
    
    # Cross-frontier comparison (γ=5 fixed, vary λ)
    print(f"\nCross-frontier at γ=5:")
    for lm in LAMBDA_MULTS:
        s = all_results["summaries"][f"lm{lm}_g5"]
        print(f"  λ_mult={lm}: acc={s['best_acc']*100:.2f}% "
              f"cost={s['cumulative_payment']:.0f} "
              f"maxPriv={s['max_privacy_spent']:.1f}")
    
    # Within-frontier spread (for each λ, compare γ=3 vs γ=10)
    print(f"\nWithin-frontier efficiency ratios (γ=10 / γ=3):")
    for lm in LAMBDA_MULTS:
        g3 = all_results["summaries"][f"lm{lm}_g3"]
        g10 = all_results["summaries"][f"lm{lm}_g10"]
        cost_r = g10["cumulative_payment"] / g3["cumulative_payment"] if g3["cumulative_payment"] > 0 else 0
        priv_r = g10["max_privacy_spent"] / g3["max_privacy_spent"] if g3["max_privacy_spent"] > 0 else 0
        acc_d = abs(g10["best_acc"] - g3["best_acc"]) * 100
        print(f"  λ_mult={lm}: cost_ratio={cost_r:.2f}x  priv_ratio={priv_r:.2f}x  acc_spread={acc_d:.2f}%")
    
    save_json(all_results, f"results/experiments/v10_1_lambda_{timestamp}.json")
    
    elapsed_total = time.time() - t_total
    print(f"\n  Total wall time: {elapsed_total/3600:.1f} hours")
    print(f"  Results saved: results/experiments/v10_1_lambda_{timestamp}.json")


if __name__ == "__main__":
    run_lambda_sweep()
