#!/usr/bin/env python3
"""
Phase 0: v8 Verification Experiments
=====================================
Run on GPU server to verify v8 (standard FD) works as expected.

Step 0: gamma sweep (seed=42, 100 rounds, gamma in {2,3,5,7,10})
  → Check if gamma creates accuracy gap in v8

Step 1: Quick 7-method comparison (seed=42, 50 rounds)
  → Check relative ranking of all methods under v8

Usage:
  python scripts/run_phase0_v8.py --step 0          # gamma sweep (~5h)
  python scripts/run_phase0_v8.py --step 1          # quick 7-method (~4h)
  python scripts/run_phase0_v8.py --step all         # both (~9h)
  python scripts/run_phase0_v8.py --step 0 --quick  # quick test (3 rounds)
"""

import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_all_experiments import run_single_experiment


def run_step0_gamma_sweep(seed=42, n_rounds=100, device="cuda", quick=False):
    """Step 0: gamma sweep to verify v8 creates accuracy gap."""
    if quick:
        n_rounds = 3

    gammas = [2, 3, 5, 7, 10]
    results = {
        "experiment": "v8_phase0_gamma_sweep",
        "version": "v8",
        "seed": seed,
        "n_rounds": n_rounds,
        "runs": {}
    }

    for gamma in gammas:
        label = f"gamma={gamma}"
        print(f"\n{'='*60}")
        print(f"  Phase 0 Step 0: PAID-FD v8, {label}, seed={seed}, {n_rounds} rounds")
        print(f"{'='*60}")

        config = {
            "n_devices": 50,
            "gamma": gamma,
            "alpha": 0.5,
            "model": "resnet18",
            "local_epochs": 5,
            "local_lr": 0.01,
            "local_momentum": 0.9,
            "distill_epochs": 1,
            "distill_lr": 0.001,
            "temperature": 3.0,
            "public_samples": 20000,
            "synthetic": False,
            "heterogeneity": {
                "config_file": "config/devices/heterogeneity.yaml",
            },
            "method_config": {
                "gamma": gamma,
                "delta": 0.01,
                "clip_bound": 2.0,
            },
        }

        run = run_single_experiment(
            method_name="PAID-FD",
            config=config,
            seed=seed,
            device=device,
            n_rounds=n_rounds,
        )
        results["runs"][label] = run

        print(f"  → {label}: final={run['final_accuracy']:.4f}, "
              f"best={run['best_accuracy']:.4f}, time={run['elapsed_sec']:.0f}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Phase 0 Step 0: gamma sweep SUMMARY")
    print(f"{'='*60}")
    for label, run in results["runs"].items():
        print(f"  {label:12s}: final={run['final_accuracy']:.4f}, "
              f"best={run['best_accuracy']:.4f}")

    finals = [r["final_accuracy"] for r in results["runs"].values()]
    gap = max(finals) - min(finals)
    print(f"\n  Accuracy gap: {gap:.4f} ({gap*100:.1f}%)")
    if gap > 0.03:
        print(f"  ✅ Gap > 3% — gamma affects accuracy in v8!")
    elif gap > 0.01:
        print(f"  ⚠️  Gap 1-3% — marginal. May need tuning.")
    else:
        print(f"  ❌ Gap < 1% — gamma still doesn't affect accuracy. Need to investigate.")

    # Save
    out_path = "results/experiments/v8_phase0_gamma_sweep.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ Saved to {out_path}")
    return results


def run_step1_quick_comparison(seed=42, n_rounds=50, device="cuda", quick=False):
    """Step 1: Quick 7-method comparison under v8."""
    if quick:
        n_rounds = 3

    methods = {
        "PAID-FD": {
            "gamma": 5.0, "delta": 0.01, "clip_bound": 2.0,
        },
        "Fixed-eps-0.5": {
            "participation_rate": 1.0,
        },
        "Fixed-eps-1.0": {
            "participation_rate": 1.0,
        },
        "FedAvg": {
            "local_epochs": 5,
            "local_lr": 0.01,
            "participation_rate": 0.5,
        },
        "FedMD": {},
        "FedGMKD": {
            "alpha": 0.5, "beta": 0.5, "dat_temperature": 1.0,
        },
        "CSRA": {
            "budget_per_round": 50.0,
        },
    }

    results = {
        "experiment": "v8_phase0_quick_comparison",
        "version": "v8",
        "seed": seed,
        "n_rounds": n_rounds,
        "runs": {}
    }

    for method_name, mc in methods.items():
        print(f"\n{'='*60}")
        print(f"  Phase 0 Step 1: {method_name}, seed={seed}, {n_rounds} rounds")
        print(f"{'='*60}")

        config = {
            "n_devices": 50,
            "gamma": mc.get("gamma", 5.0),
            "alpha": 0.5,
            "model": "resnet18",
            "local_epochs": mc.get("local_epochs", 5),
            "local_lr": mc.get("local_lr", 0.01),
            "local_momentum": 0.9,
            "distill_epochs": 1,
            "distill_lr": 0.001,
            "temperature": 3.0,
            "public_samples": 20000,
            "synthetic": False,
            "heterogeneity": {
                "config_file": "config/devices/heterogeneity.yaml",
            },
            "method_config": mc,
        }

        run = run_single_experiment(
            method_name=method_name,
            config=config,
            seed=seed,
            device=device,
            n_rounds=n_rounds,
        )
        results["runs"][method_name] = run

        print(f"  → {method_name}: final={run['final_accuracy']:.4f}, "
              f"best={run['best_accuracy']:.4f}, time={run['elapsed_sec']:.0f}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Phase 0 Step 1: 7-method comparison SUMMARY")
    print(f"{'='*60}")
    ranked = sorted(results["runs"].items(),
                    key=lambda x: x[1]["final_accuracy"], reverse=True)
    for i, (name, run) in enumerate(ranked, 1):
        print(f"  {i}. {name:20s}: final={run['final_accuracy']:.4f}, "
              f"best={run['best_accuracy']:.4f}")

    # Save
    out_path = "results/experiments/v8_phase0_quick_comparison.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ Saved to {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0: v8 Verification")
    parser.add_argument("--step", type=str, default="all",
                        choices=["0", "1", "all"])
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (3 rounds)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        args.device = "cpu"
    else:
        print(f"🖥 Using device: {args.device}")
        if args.device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

    if args.step in ["0", "all"]:
        run_step0_gamma_sweep(seed=args.seed, device=args.device, quick=args.quick)

    if args.step in ["1", "all"]:
        run_step1_quick_comparison(seed=args.seed, device=args.device, quick=args.quick)

    print("\n🏁 Phase 0 complete!")
