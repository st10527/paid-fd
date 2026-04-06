#!/usr/bin/env python3
"""
Phase 0 v9.1: Fix Distillation Hyperparameters (C, T)
======================================================

v9.0 diagnosis: Solver fix works (SNR >> 1) but teacher signal is near-uniform
because C=2 clips logits heavily and T=3 flattens softmax on 100 classes.

                  correct_class_prob   signal_above_uniform
  C=2, T=3:      1.93%                +0.93%  (CURRENT - useless!)
  C=5, T=1:      32.5%                +31.5%  (17x stronger signal)

This sweep tests C={2,5}, T={1,3}, CE_anchor={0,0.3} at gamma=5 and gamma=10.
If C=5,T=1 stops the accuracy degradation, we found the fix.

Usage:
  python scripts/run_phase0_v9_1_distill_fix.py                  # full (50 rounds)
  python scripts/run_phase0_v9_1_distill_fix.py --quick           # quick (10 rounds)
  python scripts/run_phase0_v9_1_distill_fix.py --rounds 100      # longer
"""

import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_all_experiments import run_single_experiment


CONFIGS = {
    # Key configs to test (ordered by expected quality)
    # Format: (C, T, ce_alpha, gamma, label)
    "D1_C2_T3_g5":     (2.0, 3.0, 0.0, 5,  "C=2 T=3 (v9.0 baseline)"),
    "D2_C5_T1_g5":     (5.0, 1.0, 0.0, 5,  "C=5 T=1 (strong signal)"),
    "D3_C5_T2_g5":     (5.0, 2.0, 0.0, 5,  "C=5 T=2 (moderate)"),
    "D4_C5_T1_CE_g5":  (5.0, 1.0, 0.3, 5,  "C=5 T=1 +CE anchor"),
    "D5_C2_T1_g5":     (2.0, 1.0, 0.0, 5,  "C=2 T=1 (only T fix)"),
    "D6_C5_T3_g5":     (5.0, 3.0, 0.0, 5,  "C=5 T=3 (only C fix)"),
    # Gamma=10 with best config
    "D7_C5_T1_g10":    (5.0, 1.0, 0.0, 10, "C=5 T=1 g=10"),
    "D8_C5_T1_g3":     (5.0, 1.0, 0.0, 3,  "C=5 T=1 g=3"),
}


def run_distill_fix_sweep(configs_to_run, n_rounds=50, seed=42, device="cuda"):
    """Run distillation hyperparameter sweep."""

    results = {
        "experiment": "v9_1_distill_fix",
        "description": "Fix distillation: C, T, CE_anchor sweep with fixed cubic solver",
        "version": "v9.1",
        "seed": seed,
        "n_rounds": n_rounds,
        "configs": {k: CONFIGS[k] for k in configs_to_run},
        "runs": {},
    }

    total_start = time.time()

    for config_key in configs_to_run:
        C, T, ce_alpha, gamma, desc = CONFIGS[config_key]
        print("\n" + "=" * 70)
        print("  %s: %s | gamma=%d | %d rounds" % (config_key, desc, gamma, n_rounds))
        print("=" * 70)

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
            "temperature": T,
            "public_samples": 20000,
            "synthetic": False,
            "heterogeneity": {
                "config_file": "config/devices/heterogeneity.yaml",
            },
            "method_config": {
                "gamma": float(gamma),
                "delta": 0.01,
                "clip_bound": C,
                "distill_lr": 0.001,
                "distill_epochs": 1,
                "temperature": T,
                "ce_anchor_alpha": ce_alpha,
                "use_denoising": False,       # No denoising
            },
        }

        run = run_single_experiment(
            method_name="PAID-FD",
            config=config,
            seed=seed,
            device=device,
            n_rounds=n_rounds,
        )
        results["runs"][config_key] = run

        # Print summary
        accs = run["accuracies"]
        r1 = accs[0]
        final = accs[-1]
        best = max(accs)
        best_r = accs.index(best)
        parts = run.get("participation_rates", [])
        avg_part = sum(parts) / len(parts) if parts else 0
        eps_list = run.get("avg_eps", [])
        avg_eps = sum(eps_list) / len(eps_list) if eps_list else 0

        print("\n  %s" % desc)
        print("  R1=%.4f  Final=%.4f  Best=%.4f (R%d)  Part=%.0f%%  Eps=%.3f  Time=%.0fs"
              % (r1, final, best, best_r, avg_part * 100, avg_eps, run["elapsed_sec"]))
        
        # Trajectory
        print("  Trajectory (every 10 rounds):")
        for i in range(0, len(accs), 10):
            print("    R%3d: %.4f" % (i, accs[i]))
        if (len(accs) - 1) % 10 != 0:
            print("    R%3d: %.4f (final)" % (len(accs) - 1, accs[-1]))

    total_time = time.time() - total_start

    # ====== SUMMARY ======
    print("\n" + "=" * 85)
    print("  v9.1 DISTILLATION FIX SWEEP — SUMMARY")
    print("  (%d rounds, seed=%d)" % (n_rounds, seed))
    print("=" * 85)
    print("  {:<20s} {:>4s} {:>4s} {:>4s} {:>6s} {:>6s} {:>6s} {:>7s} {:>5s} {:>5s}".format(
        "Config", "C", "T", "CE", "R1", "Final", "Best", "Delta", "Part", "Eps"))
    print("  " + "-" * 80)

    for config_key in configs_to_run:
        C, T, ce_alpha, gamma, desc = CONFIGS[config_key]
        run = results["runs"][config_key]
        accs = run["accuracies"]
        r1 = accs[0]
        final = accs[-1]
        best = max(accs)
        delta = final - r1
        parts = run.get("participation_rates", [])
        avg_part = sum(parts) / len(parts) if parts else 0
        eps_list = run.get("avg_eps", [])
        avg_eps = sum(eps_list) / len(eps_list) if eps_list else 0

        tag = "+" if delta > 0.005 else ("~" if delta > -0.02 else "-")
        print("  {:<20s} {:4.0f} {:4.0f} {:4.1f} {:5.1f}% {:5.1f}% {:5.1f}% {:+5.1f}% {:4.0f}% {:5.2f} {}".format(
            config_key, C, T, ce_alpha, r1*100, final*100, best*100, delta*100,
            avg_part*100, avg_eps, tag))

    # Key comparisons
    print()
    print("  KEY COMPARISONS:")
    
    # D1 vs D2: full fix
    if "D1_C2_T3_g5" in results["runs"] and "D2_C5_T1_g5" in results["runs"]:
        d1_final = results["runs"]["D1_C2_T3_g5"]["accuracies"][-1]
        d2_final = results["runs"]["D2_C5_T1_g5"]["accuracies"][-1]
        d1_r1 = results["runs"]["D1_C2_T3_g5"]["accuracies"][0]
        d2_r1 = results["runs"]["D2_C5_T1_g5"]["accuracies"][0]
        print("    C=2,T=3 -> C=5,T=1: R1 %.1f%%->%.1f%%, Final %.1f%%->%.1f%% (%+.1f%%)" % (
            d1_r1*100, d2_r1*100, d1_final*100, d2_final*100, (d2_final-d1_final)*100))

    # D5 vs D2: only C change
    if "D5_C2_T1_g5" in results["runs"] and "D2_C5_T1_g5" in results["runs"]:
        d5_final = results["runs"]["D5_C2_T1_g5"]["accuracies"][-1]
        d2_final = results["runs"]["D2_C5_T1_g5"]["accuracies"][-1]
        print("    C=2->5 (T=1 fixed): Final %.1f%% -> %.1f%% (%+.1f%%)" % (
            d5_final*100, d2_final*100, (d2_final-d5_final)*100))

    # D6 vs D2: only T change
    if "D6_C5_T3_g5" in results["runs"] and "D2_C5_T1_g5" in results["runs"]:
        d6_final = results["runs"]["D6_C5_T3_g5"]["accuracies"][-1]
        d2_final = results["runs"]["D2_C5_T1_g5"]["accuracies"][-1]
        print("    T=3->1 (C=5 fixed): Final %.1f%% -> %.1f%% (%+.1f%%)" % (
            d6_final*100, d2_final*100, (d2_final-d6_final)*100))

    # Gamma differentiation D8 vs D2 vs D7
    gamma_configs = ["D8_C5_T1_g3", "D2_C5_T1_g5", "D7_C5_T1_g10"]
    gamma_finals = []
    for gc in gamma_configs:
        if gc in results["runs"]:
            gamma_finals.append((gc, results["runs"][gc]["accuracies"][-1]))
    
    if len(gamma_finals) >= 2:
        best_g = max(gamma_finals, key=lambda x: x[1])
        worst_g = min(gamma_finals, key=lambda x: x[1])
        gap = best_g[1] - worst_g[1]
        print("    Gamma differentiation (C=5,T=1): gap = %.1f%% (%s=%.1f%% vs %s=%.1f%%)" % (
            gap*100, best_g[0], best_g[1]*100, worst_g[0], worst_g[1]*100))

    print()
    print("  Total time: %.0f seconds (%.1f hours)" % (total_time, total_time / 3600))

    # Save
    out_path = "results/experiments/v9_1_distill_fix.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2, default=str)
    print("\n  Saved to %s" % out_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v9.1: Distillation hyperparameter fix")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                        help="Config keys to run (default: all)")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--quick", action="store_true", help="10 rounds")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        args.rounds = 10

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    else:
        print("Using device: %s" % args.device)
        if args.device == "cuda":
            print("  GPU: %s" % torch.cuda.get_device_name(0))

    # Validate config keys
    for k in args.configs:
        if k not in CONFIGS:
            print("Unknown config: %s. Available: %s" % (k, list(CONFIGS.keys())))
            sys.exit(1)

    print()
    print("=" * 70)
    print("  v9.1: Distillation Hyperparameter Fix")
    print("=" * 70)
    print()
    print("  Problem: C=2,T=3 -> teacher signal near-uniform (1.93%% vs 1%% uniform)")
    print("  Fix: C=5,T=1 -> teacher signal strong (32.5%% vs 1%% uniform)")
    print()
    print("  Configs: %s" % args.configs)
    print("  Rounds: %d" % args.rounds)
    print("  Seed: %d" % args.seed)
    print()

    run_distill_fix_sweep(
        configs_to_run=args.configs,
        n_rounds=args.rounds,
        seed=args.seed,
        device=args.device,
    )

    print("\nv9.1 sweep complete!")
