#!/usr/bin/env python3
"""
PAID-FD v10.1 Single Run (CLI)
================================
Runs a single (gamma, seed, lambda_mult) experiment and saves result JSON.
Designed for SLURM array jobs on TWCC — each task runs one config.

Usage:
    python scripts/run_v10_1_single.py --gamma 5 --seed 42 --lambda-mult 1.0
    python scripts/run_v10_1_single.py --gamma 3 --seed 123 --lambda-mult 2.0 --device cuda:0
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path

from scripts.run_all_experiments import run_single_experiment, save_json


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


def main():
    parser = argparse.ArgumentParser(description="PAID-FD v10.1 single run")
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-mult", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--outdir", type=str, default="results/experiments")
    args = parser.parse_args()

    gamma = args.gamma
    seed = args.seed
    lm = args.lambda_mult
    label = "g%d_s%d_lm%s" % (int(gamma), seed, lm)
    outfile = Path(args.outdir) / ("v10_1_%s.json" % label)

    # Skip if already completed
    if outfile.exists():
        print("[SKIP] %s already exists: %s" % (label, outfile))
        return

    print("=" * 60)
    print("  PAID-FD v10.1 — %s" % label)
    print("  gamma=%.1f  seed=%d  lambda_mult=%.1f" % (gamma, seed, lm))
    print("  device=%s  rounds=%d" % (args.device, args.rounds))
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
        device=args.device,
        n_rounds=args.rounds,
        save_decisions=False,
        verbose=True,
    )
    elapsed = time.time() - t0

    summary = extract_summary(run, gamma, seed, lm)
    summary["elapsed_sec"] = elapsed

    # Save result
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    result = {
        "label": label,
        "config": {"gamma": gamma, "seed": seed, "lambda_mult": lm},
        "summary": summary,
        "accuracies": run["accuracies"],
        "participation_rates": run["participation_rates"],
        "prices": run["prices"],
        "avg_eps": run["avg_eps"],
        "completed_at": datetime.now().isoformat(),
        "elapsed_sec": elapsed,
    }
    save_json(result, str(outfile))

    print("\n" + "=" * 60)
    print("  DONE: %s" % label)
    print("  best=%.2f%%  final=%.2f%%  part=%.0f%%  cost=%.0f" % (
        summary["best_acc"] * 100, summary["final_acc"] * 100,
        summary["avg_participation"] * 100, summary["cumulative_payment"]))
    print("  time=%.0fs (%.1f min)" % (elapsed, elapsed / 60))
    print("  saved: %s" % outfile)
    print("=" * 60)


if __name__ == "__main__":
    main()
