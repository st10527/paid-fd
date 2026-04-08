#!/usr/bin/env python3
"""
v10 Experiment: Persistent Local Models + Solver Fix
=====================================================

Reframed for the EFFICIENCY STORY (not accuracy gap).

Key insight: Game theory's value is resource allocation efficiency,
not ML accuracy improvement. The LDP noise floor is physical —
game mechanism can't make distillation signal better. But it CAN:
  - Make device participation voluntary & rational
  - Allocate privacy budget efficiently across heterogeneous devices
  - Let server trade off cost vs quality via γ tuning
  - Avoid free-riders and over-contribution

Architecture:
  - Persistent local models (v7): genuine local knowledge
  - Fixed cubic solver (v9.0): ε* ≈ 3.0
  - Fresh SGD, EMA buffer, mixed loss (α=0.7)
  - C=2, T=3, 5 local epochs/round

Sweep: γ ∈ {3, 5, 7, 10}, 100 rounds, seed=42

What to look for (EFFICIENCY metrics, not accuracy gap):
  E1: Do all γ values converge to ~same accuracy ceiling? (efficiency frontier)
  E2: How much does cumulative cost (Σ payments) vary across γ?  (>3× ratio?)
  E3: How much does cumulative privacy (Σε) vary across γ?       (>2× ratio?)
  E4: How does participation rate vary across γ?
  E5: Time-to-target: which γ reaches 55% fastest? At what cost?

Paper Fig 3 should be cost-accuracy Pareto frontier or privacy-accuracy
Pareto frontier, NOT accuracy-vs-γ (which will be flat).
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
    """Run v10 persistent models sweep: γ ∈ {3, 5, 7, 10}.
    
    Framing: mobile computing efficiency story.
    γ controls the server's willingness-to-pay. Different γ → different
    operating points on the cost-accuracy-privacy Pareto frontier.
    """
    
    device = "cuda"
    seed = 42
    n_rounds = 100
    gamma_values = [3, 5, 7, 10]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    print("=" * 70)
    print("v10: Persistent Local Models + Solver Fix")
    print("      *** EFFICIENCY STORY FRAMING ***")
    print("=" * 70)
    print(f"  γ values:   {gamma_values}")
    print(f"  n_rounds:   {n_rounds}")
    print(f"  seed:       {seed}")
    print(f"  device:     {device}")
    print(f"  timestamp:  {timestamp}")
    print()
    print("  What we're measuring:")
    print("    - Accuracy ceiling (should be ~same for all γ)")
    print("    - Cumulative cost (Σ payments) — should DIFFER across γ")
    print("    - Cumulative privacy (Σε per device) — should DIFFER across γ")
    print("    - Participation pattern — should DIFFER across γ")
    print("    - Time-to-target accuracy — should DIFFER across γ")
    print("=" * 70)
    
    all_results = {
        "experiment": "v10_persistent_efficiency",
        "framing": "efficiency_story",
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
        
        # Quick summary with EFFICIENCY metrics
        accs = run["accuracies"]
        final_acc = accs[-1] if accs else 0
        best_acc = max(accs) if accs else 0
        extras = run.get("extras", [])
        
        # Cumulative cost
        cum_payment = extras[-1].get("cumulative_payment", 0) if extras else 0
        # Average participation
        avg_part = np.mean(run["participation_rates"]) if run["participation_rates"] else 0
        # Cumulative avg privacy
        avg_priv = extras[-1].get("avg_privacy_spent", 0) if extras else 0
        max_priv = extras[-1].get("max_privacy_spent", 0) if extras else 0
        # Average price
        avg_price = np.mean(run["prices"]) if run["prices"] else 0
        avg_eps = np.mean(run["avg_eps"]) if run["avg_eps"] else 0
        
        # Time-to-target
        targets = [0.45, 0.50, 0.55, 0.58, 0.60]
        ttt = {}
        for target in targets:
            reached = [i for i, a in enumerate(accs) if a >= target]
            ttt[target] = reached[0] if reached else None
        
        print(f"\n  SUMMARY γ={gamma} (EFFICIENCY VIEW):")
        print(f"    Final acc:         {final_acc*100:.2f}%")
        print(f"    Best acc:          {best_acc*100:.2f}%")
        print(f"    Avg participation: {avg_part:.2%}")
        print(f"    Avg price p*:      {avg_price:.4f}")
        print(f"    Avg ε* per round:  {avg_eps:.4f}")
        print(f"    Cumulative cost:   {cum_payment:.2f}")
        print(f"    Avg privacy spent: {avg_priv:.2f}")
        print(f"    Max privacy spent: {max_priv:.2f}")
        print(f"    Time-to-targets:")
        for target, rnd in ttt.items():
            status = f"Round {rnd}" if rnd is not None else "NOT REACHED"
            print(f"      {target*100:.0f}%: {status}")
        print(f"    Wall time: {elapsed:.0f}s")
        
        # Save intermediate
        save_json(all_results, f"results/experiments/v10_persistent_{timestamp}.json")
    
    # ============================================================
    # EFFICIENCY ANALYSIS
    # ============================================================
    print("\n" + "=" * 70)
    print("v10 EFFICIENCY ANALYSIS")
    print("(Game theory value = resource allocation, not accuracy)")
    print("=" * 70)
    
    summary_rows = []
    for gamma in gamma_values:
        label = f"g{gamma}"
        run = all_results["results"][label]
        accs = run["accuracies"]
        extras = run.get("extras", [])
        
        final_acc = accs[-1] if accs else 0
        best_acc = max(accs) if accs else 0
        
        cum_payment = extras[-1].get("cumulative_payment", 0) if extras else 0
        avg_part = np.mean(run["participation_rates"]) if run["participation_rates"] else 0
        avg_priv = extras[-1].get("avg_privacy_spent", 0) if extras else 0
        max_priv = extras[-1].get("max_privacy_spent", 0) if extras else 0
        avg_price = np.mean(run["prices"]) if run["prices"] else 0
        avg_eps_round = np.mean(run["avg_eps"]) if run["avg_eps"] else 0
        
        # Cumulative energy
        energy_total = sum(
            sum(e.values()) for e in run.get("energy_history", []) if isinstance(e, dict)
        )
        
        # Time to targets
        targets = [0.45, 0.50, 0.55, 0.58, 0.60]
        ttt = {}
        for target in targets:
            reached = [i for i, a in enumerate(accs) if a >= target]
            ttt[target] = reached[0] if reached else None
        
        # Distill delta (still useful diagnostic)
        distill_deltas = [e.get("distill_delta", None) for e in extras if e]
        distill_deltas = [d for d in distill_deltas if d is not None]
        avg_delta = np.mean(distill_deltas) if distill_deltas else float('nan')
        
        summary_rows.append({
            "gamma": gamma,
            "final_acc": final_acc,
            "best_acc": best_acc,
            "avg_participation": avg_part,
            "avg_price": avg_price,
            "avg_eps_per_round": avg_eps_round,
            "cumulative_payment": cum_payment,
            "avg_privacy_spent": avg_priv,
            "max_privacy_spent": max_priv,
            "energy_total": energy_total,
            "time_to_targets": ttt,
            "avg_distill_delta": avg_delta,
        })
    
    # ---- Table 1: Accuracy (should be ~flat = good) ----
    print(f"\n--- TABLE 1: Accuracy Ceiling (expect ~flat across γ) ---")
    print(f"{'γ':>4} | {'Final':>8} | {'Best':>8} | {'Avg Δ/rnd':>10}")
    print("-" * 40)
    for row in summary_rows:
        print(f"{row['gamma']:4d} | {row['final_acc']*100:7.2f}% | "
              f"{row['best_acc']*100:7.2f}% | "
              f"{row['avg_distill_delta']*100:+9.4f}%")
    
    # ---- Table 2: Efficiency (should DIFFER = the story) ----
    print(f"\n--- TABLE 2: Efficiency Metrics (expect clear γ differentiation) ---")
    print(f"{'γ':>4} | {'Participation':>13} | {'Avg p*':>8} | {'Avg ε*/rnd':>10} | "
          f"{'Σ Cost':>10} | {'Avg Σε':>8} | {'Max Σε':>8}")
    print("-" * 80)
    for row in summary_rows:
        print(f"{row['gamma']:4d} | {row['avg_participation']:12.1%} | "
              f"{row['avg_price']:8.4f} | {row['avg_eps_per_round']:10.4f} | "
              f"{row['cumulative_payment']:10.2f} | "
              f"{row['avg_privacy_spent']:8.2f} | {row['max_privacy_spent']:8.2f}")
    
    # ---- Table 3: Time-to-target ----
    print(f"\n--- TABLE 3: Time-to-Target (round # to reach accuracy) ---")
    targets = [0.45, 0.50, 0.55, 0.58, 0.60]
    header = f"{'γ':>4} | " + " | ".join(f"{t*100:4.0f}%" for t in targets)
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        vals = []
        for t in targets:
            rnd = row["time_to_targets"].get(t, None)
            vals.append(f"{rnd:5d}" if rnd is not None else "  N/A")
        print(f"{row['gamma']:4d} | " + " | ".join(vals))
    
    # ---- Efficiency ratios ----
    print(f"\n--- EFFICIENCY RATIOS (γ=10 / γ=3) ---")
    g3 = next((r for r in summary_rows if r["gamma"] == 3), None)
    g10 = next((r for r in summary_rows if r["gamma"] == 10), None)
    
    if g3 and g10:
        acc_diff = abs(g10["final_acc"] - g3["final_acc"])
        cost_ratio = g10["cumulative_payment"] / g3["cumulative_payment"] if g3["cumulative_payment"] > 0 else float('inf')
        priv_ratio = g10["avg_privacy_spent"] / g3["avg_privacy_spent"] if g3["avg_privacy_spent"] > 0 else float('inf')
        part_ratio = g10["avg_participation"] / g3["avg_participation"] if g3["avg_participation"] > 0 else float('inf')
        
        print(f"  Accuracy difference:     {acc_diff*100:.2f}%  (expect small = good)")
        print(f"  Cost ratio (Σ pay):      {cost_ratio:.2f}×  (expect >2× = differentiation)")
        print(f"  Privacy ratio (avg Σε):  {priv_ratio:.2f}×  (expect >1.5× = differentiation)")
        print(f"  Participation ratio:     {part_ratio:.2f}×  (expect >1× = γ matters)")
    
    # ---- Success criteria (EFFICIENCY framing) ----
    print(f"\n{'='*60}")
    print("  EFFICIENCY STORY SUCCESS CRITERIA")
    print(f"{'='*60}")
    
    finals = {r["gamma"]: r["final_acc"] for r in summary_rows}
    best_final = max(finals.values()) if finals else 0
    worst_final = min(finals.values()) if finals else 0
    acc_spread = best_final - worst_final
    
    costs = {r["gamma"]: r["cumulative_payment"] for r in summary_rows}
    cost_ratio = max(costs.values()) / min(costs.values()) if min(costs.values()) > 0 else 0
    
    privs = {r["gamma"]: r["avg_privacy_spent"] for r in summary_rows}
    priv_ratio = max(privs.values()) / min(privs.values()) if min(privs.values()) > 0 else 0
    
    parts = {r["gamma"]: r["avg_participation"] for r in summary_rows}
    part_spread = max(parts.values()) - min(parts.values())
    
    e1_pass = acc_spread < 0.05  # All γ within 5% of each other
    e2_pass = cost_ratio >= 2.0  # Cost varies by >2×
    e3_pass = priv_ratio >= 1.5  # Privacy varies by >1.5×
    e4_pass = part_spread >= 0.10  # Participation varies by >10%
    e5_pass = best_final >= 0.50  # System is actually working (acc > 50%)
    
    print(f"  E1: Acc ceiling ~flat (spread < 5%)?    {'YES ✓' if e1_pass else 'NO ✗'}  (spread={acc_spread*100:.2f}%)")
    print(f"  E2: Cost ratio > 2×?                    {'YES ✓' if e2_pass else 'NO ✗'}  (ratio={cost_ratio:.2f}×)")
    print(f"  E3: Privacy ratio > 1.5×?               {'YES ✓' if e3_pass else 'NO ✗'}  (ratio={priv_ratio:.2f}×)")
    print(f"  E4: Participation spread > 10%?          {'YES ✓' if e4_pass else 'NO ✗'}  (spread={part_spread*100:.1f}%)")
    print(f"  E5: System works (best acc > 50%)?       {'YES ✓' if e5_pass else 'NO ✗'}  (best={best_final*100:.1f}%)")
    
    n_pass = sum([e1_pass, e2_pass, e3_pass, e4_pass, e5_pass])
    
    if n_pass >= 4 and e5_pass:
        print(f"\n  🎉 EFFICIENCY STORY WORKS ({n_pass}/5)!")
        print(f"     γ controls operating point on cost-accuracy-privacy frontier.")
        print(f"     This is the TMC paper narrative.")
    elif e5_pass and (e2_pass or e3_pass):
        print(f"\n  ⚠️  PARTIAL ({n_pass}/5): System works and some efficiency differentiation.")
        print(f"     Enough for paper, may need framing adjustments.")
    elif e5_pass:
        print(f"\n  ⚠️  System works but game differentiation weak ({n_pass}/5).")
        print(f"     Check if γ range needs expanding.")
    else:
        print(f"\n  ❌ System not working (accuracy < 50%). Debug pipeline first.")
    
    # Also print the OLD accuracy-gap metrics for reference
    print(f"\n--- LEGACY: Accuracy Gap Metrics (for reference only) ---")
    gamma_gap = finals.get(10, 0) - finals.get(3, 0)
    print(f"  γ gap (γ=10 - γ=3): {gamma_gap*100:.2f}%")
    print(f"  (In efficiency framing, this being ~0 is EXPECTED and OK)")
    
    # Save final
    all_results["summary"] = summary_rows
    all_results["efficiency_criteria"] = {
        "E1_acc_ceiling_flat": e1_pass,
        "E2_cost_ratio_gt2": e2_pass,
        "E3_privacy_ratio_gt1_5": e3_pass,
        "E4_participation_spread_gt10pct": e4_pass,
        "E5_system_works": e5_pass,
        "acc_spread": acc_spread,
        "cost_ratio": cost_ratio,
        "privacy_ratio": priv_ratio,
        "participation_spread": part_spread,
        "best_final_acc": best_final,
    }
    save_json(all_results, f"results/experiments/v10_persistent_{timestamp}.json")
    print(f"\n  Results saved to: results/experiments/v10_persistent_{timestamp}.json")
    
    return all_results


if __name__ == "__main__":
    run_v10_persistent()
