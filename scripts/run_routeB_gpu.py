#!/usr/bin/env python3
"""
Route B – GPU Runner: Exp 1 (6-method comparison) & Exp 6 (Ablation)
=====================================================================
Run on GPU server. Saves results to results/experiments/routeB_*.json.

Usage:
  python scripts/run_routeB_gpu.py --exp 1          # 6-method comparison (~12h)
  python scripts/run_routeB_gpu.py --exp 6          # Ablation study (~8h)
  python scripts/run_routeB_gpu.py --exp all        # Both (~20h)
  python scripts/run_routeB_gpu.py --exp 1 --quick  # Quick test (3 rounds)
"""

import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_all_experiments import run_single_experiment, _create_method

# ══════════════════════════════════════════════════════════════
#  Exp 1: 6-Method Convergence Comparison
# ══════════════════════════════════════════════════════════════
# Methods: PAID-FD (γ=5), Fixed-ε (0.5, 1.0), FedAvg, FedMD, FedGMKD, CSRA
# Conditions: 3 seeds × 100 rounds each
# GPU time: ~12h (6 methods × 3 seeds × ~40min)

EXP1_METHODS = {
    "PAID-FD": {
        "gamma": 5.0, "delta": 0.01,
        "clip_bound": 2.0,
        "ema_momentum": 0.9,
        "distill_alpha": 0.7,
    },
    "Fixed-eps-0.5": {
        "participation_rate": 1.0,
        "samples_per_device": 100,
    },
    "Fixed-eps-1.0": {
        "participation_rate": 1.0,
        "samples_per_device": 100,
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


def run_exp1(seeds=[42, 123, 456], n_rounds=100, device="cuda", quick=False):
    """Run Exp 1: 6-method comparison."""
    if quick:
        seeds = [42]
        n_rounds = 3
    
    results = {"experiment": "exp1_method_comparison", "seeds": seeds, "runs": {}}
    
    for method_name, mc in EXP1_METHODS.items():
        results["runs"][method_name] = []
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"  Exp 1: {method_name}, seed={seed}, {n_rounds} rounds")
            print(f"{'='*60}")
            
            config = {
                "n_devices": 50, "gamma": mc.get("gamma", 5.0),
                "alpha": 0.5, "model": "resnet18",
                "local_epochs": mc.get("local_epochs", 5),
                "local_lr": mc.get("local_lr", 0.01),
                "local_momentum": 0.9,
                "distill_epochs": 1, "distill_lr": 0.001,
                "temperature": 3.0, "public_samples": 20000,
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
            results["runs"][method_name].append(run)
            
            print(f"  → {method_name} seed={seed}: "
                  f"final={run['final_accuracy']:.4f}, "
                  f"best={run['best_accuracy']:.4f}, "
                  f"time={run['elapsed_sec']:.0f}s")
    
    # Save
    out_path = "results/experiments/routeB_exp1_comparison.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ Exp 1 saved to {out_path}")
    return results


# ══════════════════════════════════════════════════════════════
#  Exp 6: Ablation Study
# ══════════════════════════════════════════════════════════════
# 6 variants of PAID-FD with components disabled.
# Uses γ=5 (baseline), 3 seeds × 100 rounds each.
# GPU time: ~8h (6 variants × 3 seeds × ~25min)

EXP6_VARIANTS = {
    "Full (PAID-FD)": {
        "use_blue": True, "use_ema": True, "use_mixed_loss": True, "use_ldp": True,
    },
    "No-EMA": {
        "use_blue": True, "use_ema": False, "use_mixed_loss": True, "use_ldp": True,
    },
    "No-BLUE": {
        "use_blue": False, "use_ema": True, "use_mixed_loss": True, "use_ldp": True,
    },
    "No-CE (pure KL)": {
        "use_blue": True, "use_ema": True, "use_mixed_loss": False, "use_ldp": True,
    },
    "Bare-FD": {
        "use_blue": False, "use_ema": False, "use_mixed_loss": False, "use_ldp": True,
    },
    "No-LDP (oracle)": {
        "use_blue": True, "use_ema": True, "use_mixed_loss": True, "use_ldp": False,
    },
}


def run_exp6(seeds=[42, 123, 456], n_rounds=100, device="cuda", quick=False):
    """Run Exp 6: Ablation study."""
    if quick:
        seeds = [42]
        n_rounds = 3
    
    results = {"experiment": "exp6_ablation", "gamma": 5.0, "seeds": seeds, "runs": {}}
    
    for variant_name, ablation_flags in EXP6_VARIANTS.items():
        results["runs"][variant_name] = []
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"  Exp 6: {variant_name}, seed={seed}, {n_rounds} rounds")
            print(f"  Flags: {ablation_flags}")
            print(f"{'='*60}")
            
            mc = {
                "gamma": 5.0, "delta": 0.01,
                "clip_bound": 2.0,
                "ema_momentum": 0.9,
                "distill_alpha": 0.7,
                **ablation_flags,  # Ablation flags passed to PAIDFDConfig
            }
            
            config = {
                "n_devices": 50, "gamma": 5.0,
                "alpha": 0.5, "model": "resnet18",
                "local_epochs": 5, "local_lr": 0.01,
                "local_momentum": 0.9,
                "distill_epochs": 1, "distill_lr": 0.001,
                "temperature": 3.0, "public_samples": 20000,
                "synthetic": False,
                "heterogeneity": {
                    "config_file": "config/devices/heterogeneity.yaml",
                },
                "method_config": mc,
            }
            
            run = run_single_experiment(
                method_name="PAID-FD",
                config=config,
                seed=seed,
                device=device,
                n_rounds=n_rounds,
            )
            results["runs"][variant_name].append(run)
            
            print(f"  → {variant_name} seed={seed}: "
                  f"final={run['final_accuracy']:.4f}, "
                  f"best={run['best_accuracy']:.4f}, "
                  f"time={run['elapsed_sec']:.0f}s")
    
    # Save
    out_path = "results/experiments/routeB_exp6_ablation.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ Exp 6 saved to {out_path}")
    return results


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Route B GPU experiments")
    parser.add_argument("--exp", type=str, default="all",
                        choices=["1", "6", "all"], help="Which experiment to run")
    parser.add_argument("--quick", action="store_true", help="Quick test (3 rounds, 1 seed)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--rounds", type=int, default=100)
    args = parser.parse_args()
    
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        args.device = "cpu"
    else:
        print(f"🖥 Using device: {args.device}")
        if args.device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    if args.exp in ["1", "all"]:
        run_exp1(seeds=args.seeds, n_rounds=args.rounds,
                 device=args.device, quick=args.quick)
    
    if args.exp in ["6", "all"]:
        run_exp6(seeds=args.seeds, n_rounds=args.rounds,
                 device=args.device, quick=args.quick)
    
    print("\n🏁 All requested experiments complete!")
