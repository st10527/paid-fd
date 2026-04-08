#!/usr/bin/env python3
"""
v10 Experiment: Persistent Local Models + Solver Fix
=====================================================

This is the critical test: does returning to persistent local models (v7)
with the fixed cubic solver (v9.0) unlock γ differentiation?

Architecture:
  - Persistent local models (each device keeps model across rounds)
  - Fixed cubic solver (ε* ≈ 3.0 instead of old 0.5)
  - Fresh SGD per round (no Adam state leak)
  - EMA logit buffer (momentum=0.9)
  - Mixed loss (α=0.7: 70% KL + 30% CE)
  - C=2, T=3

Sweep: γ ∈ {3, 5, 7, 10}, 100 rounds, seed=42
Expected: 55-65% accuracy, visible γ differentiation

Success criteria:
  Q1: Is final accuracy > 55%?              (vs v9.2's ~38%)
  Q2: Is γ gap (γ=10 - γ=3) > 3%?          (vs v9.2's 0.3-0.9%)
  Q3: Is per-round distill_delta positive?  (vs v9.2's -0.15%)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
from datetime import datetime

from scripts.run_all_experiments import (
    run_single_experiment,
    save_json,
)


def run_v10_persistent():
    """Run v10 persistent models sweep: γ ∈ {3, 5, 7, 10}."""
    
    device = "cuda"
    seed = 42
    n_rounds = 100
    gamma_values = [3, 5, 7, 10]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    print("=" * 70)
    print("v10: Persistent Local Models + Solver Fix")
    print("=" * 70)
    print(f"  γ values:   {gamma_values}")
    print(f"  n_rounds:   {n_rounds}")
    print(f"  seed:       {seed}")
    print(f"  device:     {device}")
    print(f"  timestamp:  {timestamp}")
    print()
    print("  Architecture:")
    print("    - Persistent local models (v7)")
    print("    - Fixed cubic solver (v9.0, ε* ≈ 3.0)")
    print("    - Fresh SGD per distill round (no Adam)")
    print("    - EMA logit buffer (momentum=0.9)")
    print("    - Mixed loss (α=0.7: 70% KL + 30% CE)")
    print("    - C=2, T=3, 5 local epochs/round")
    print("=" * 70)
    
    all_results = {
        "experiment": "v10_persistent_models",
        "timestamp": timestamp,
        "architecture": {
            "persistent_local_models": True,
            "solver_fix": True,
            "fresh_sgd": True,
            "ema_momentum": 0.9,
            "distill_alpha": 0.7,
            "clip_bound": 2.0,
            "temperature": 3.0,
            "local_epochs": 5,
        },
        "configs": [],
        "results": {},
    }
    
    for gamma in gamma_values:
        label = f"g{gamma}"
        print(f"\n{'='*60}")
        print(f"  γ = {gamma}  (label: {label})")
        print(f"{'='*60}")
        
        config = {
            'n_devices': 50,
            'gamma': gamma,
            'alpha': 0.5,  # Dirichlet
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
                'gamma': gamma,
                'delta': 0.01,
                'clip_bound': 2.0,
                'ema_momentum': 0.9,
                'distill_alpha': 0.7,
                'temperature': 3.0,
                'pretrain_epochs': 10,
                'pretrain_lr': 0.1,
                # v10 flags
                'use_blue': True,
                'use_ema': True,
                'use_mixed_loss': True,
                'use_ldp': True,
                'use_denoising': False,
            }
        }
        
        all_results["configs"].append({
            "label": label,
            "gamma": gamma,
            "config": config,
        })
        
        t0 = time.time()
        run = run_single_experiment(
            method_name='PAID-FD',
            config=config,
            seed=seed,
            device=device,
            n_rounds=n_rounds,
            save_decisions=False,
            verbose=True,
        )
        elapsed = time.time() - t0
        
        run["label"] = label
        run["gamma"] = gamma
        run["elapsed_sec"] = elapsed
        all_results["results"][label] = run
        
        # Quick summary
        accs = run["accuracies"]
        final_acc = accs[-1] if accs else 0
        best_acc = max(accs) if accs else 0
        r1_acc = accs[0] if accs else 0
        
        # Extract diagnostics from extras
        extras = run.get("extras", [])
        distill_deltas = [e.get("distill_delta", None) for e in extras if e]
        distill_deltas = [d for d in distill_deltas if d is not None]
        avg_delta = np.mean(distill_deltas) if distill_deltas else float('nan')
        
        n_models = extras[-1].get("n_local_models", 0) if extras else 0
        
        print(f"\n  SUMMARY γ={gamma}:")
        print(f"    R1 acc:       {r1_acc*100:.2f}%")
        print(f"    Final acc:    {final_acc*100:.2f}%")
        print(f"    Best acc:     {best_acc*100:.2f}%")
        print(f"    Avg distill_delta: {avg_delta*100:.4f}%")
        print(f"    N local models: {n_models}")
        print(f"    Time: {elapsed:.0f}s")
        
        # Save intermediate results
        save_json(all_results, f"results/experiments/v10_persistent_{timestamp}.json")
    
    # ============================================================
    # Final analysis
    # ============================================================
    print("\n" + "=" * 70)
    print("v10 FINAL ANALYSIS")
    print("=" * 70)
    
    summary_rows = []
    for gamma in gamma_values:
        label = f"g{gamma}"
        run = all_results["results"][label]
        accs = run["accuracies"]
        final_acc = accs[-1] if accs else 0
        best_acc = max(accs) if accs else 0
        r1_acc = accs[0] if accs else 0
        
        extras = run.get("extras", [])
        distill_deltas = [e.get("distill_delta", None) for e in extras if e]
        distill_deltas = [d for d in distill_deltas if d is not None]
        avg_delta = np.mean(distill_deltas) if distill_deltas else float('nan')
        
        summary_rows.append({
            "gamma": gamma,
            "r1_acc": r1_acc,
            "final_acc": final_acc,
            "best_acc": best_acc,
            "avg_distill_delta": avg_delta,
        })
    
    # Table
    print(f"\n{'γ':>4} | {'R1 acc':>8} | {'Final acc':>10} | {'Best acc':>9} | {'Avg Δ/round':>12}")
    print("-" * 55)
    for row in summary_rows:
        print(f"{row['gamma']:4d} | {row['r1_acc']*100:7.2f}% | "
              f"{row['final_acc']*100:9.2f}% | {row['best_acc']*100:8.2f}% | "
              f"{row['avg_distill_delta']*100:+11.4f}%")
    
    # Success criteria
    finals = {row["gamma"]: row["final_acc"] for row in summary_rows}
    gamma_gap = finals.get(10, 0) - finals.get(3, 0)
    best_final = max(finals.values()) if finals else 0
    avg_deltas = {row["gamma"]: row["avg_distill_delta"] for row in summary_rows}
    any_positive_delta = any(d > 0 for d in avg_deltas.values() if not np.isnan(d))
    
    print(f"\n  SUCCESS CRITERIA:")
    print(f"    Q1: Final acc > 55%?              {'YES ✓' if best_final > 0.55 else 'NO ✗'}  (best={best_final*100:.1f}%)")
    print(f"    Q2: γ gap (γ=10 - γ=3) > 3%?     {'YES ✓' if gamma_gap > 0.03 else 'NO ✗'}  (gap={gamma_gap*100:.2f}%)")
    print(f"    Q3: Avg distill_delta positive?    {'YES ✓' if any_positive_delta else 'NO ✗'}")
    
    if best_final > 0.55 and gamma_gap > 0.03:
        print(f"\n  🎉 v10 PASSED! This is the final pipeline for the paper.")
    elif best_final > 0.55:
        print(f"\n  ⚠️  v10 accuracy good but γ gap insufficient. May need parameter tuning.")
    else:
        print(f"\n  ❌ v10 did not meet accuracy target. Consider Option D (efficiency story).")
    
    # Save final
    all_results["summary"] = summary_rows
    all_results["success_criteria"] = {
        "Q1_acc_above_55": best_final > 0.55,
        "Q2_gamma_gap_above_3": gamma_gap > 0.03,
        "Q3_positive_distill_delta": any_positive_delta,
        "best_final_acc": best_final,
        "gamma_gap": gamma_gap,
    }
    save_json(all_results, f"results/experiments/v10_persistent_{timestamp}.json")
    print(f"\n  Results saved to: results/experiments/v10_persistent_{timestamp}.json")
    
    return all_results


if __name__ == "__main__":
    run_v10_persistent()
