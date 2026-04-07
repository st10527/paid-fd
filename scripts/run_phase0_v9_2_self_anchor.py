#!/usr/bin/env python3
"""
Phase 0 v9.2: Self-Anchor Distillation
=======================================

v9.1 diagnosis: All pure KL configs degrade. Degradation ∝ C (noise scale).
Only CE anchor is stable, but CE makes γ irrelevant (structural dilemma).

Route 2 fix: Self-anchor — before distillation, compute server's own logits
as an anchor. Loss = α_sa × KL(teacher) + (1-α_sa) × KL(self).
No ground truth needed. Prevents noise drift while keeping γ relevant.

Sweep design:
  Phase A: α_sa sweep {0.3, 0.5, 0.7} at γ=5, C=5, T=1
           + baseline (α_sa=0) and CE anchor for comparison
  Phase B: γ sweep {3, 5, 10} at best α_sa from Phase A

Expected outcome:
  - α_sa > 0 should reduce/eliminate degradation (Δ ≈ 0 instead of -4.4%)
  - γ differentiation should increase (gap > 0.9% from v9.1)
  - Higher γ → higher accuracy (γ controls teacher quality)

Usage:
  python scripts/run_phase0_v9_2_self_anchor.py                  # full (50 rounds)
  python scripts/run_phase0_v9_2_self_anchor.py --quick           # quick (10 rounds)
  python scripts/run_phase0_v9_2_self_anchor.py --rounds 100      # longer
  python scripts/run_phase0_v9_2_self_anchor.py --phase A         # only alpha sweep
  python scripts/run_phase0_v9_2_self_anchor.py --phase B         # only gamma sweep
"""

import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_all_experiments import run_single_experiment


# Format: (C, T, ce_alpha, self_anchor_alpha, gamma, label)
CONFIGS = {
    # --- Phase A: Self-anchor alpha sweep (γ=5, C=5, T=1) ---
    "A0_baseline":      (5.0, 1.0, 0.0, 0.0, 5,  "No anchor (v9.1 baseline)"),
    "A1_sa03":          (5.0, 1.0, 0.0, 0.3, 5,  "Self-anchor α=0.3"),
    "A2_sa05":          (5.0, 1.0, 0.0, 0.5, 5,  "Self-anchor α=0.5"),
    "A3_sa07":          (5.0, 1.0, 0.0, 0.7, 5,  "Self-anchor α=0.7"),
    "A4_ce03":          (5.0, 1.0, 0.3, 0.0, 5,  "CE anchor α=0.3 (reference)"),

    # --- Phase B: Gamma sweep at best α_sa (default α=0.5, C=5, T=1) ---
    "B1_g3_sa05":       (5.0, 1.0, 0.0, 0.5, 3,  "γ=3 + self-anchor 0.5"),
    "B2_g5_sa05":       (5.0, 1.0, 0.0, 0.5, 5,  "γ=5 + self-anchor 0.5 (=A2)"),
    "B3_g10_sa05":      (5.0, 1.0, 0.0, 0.5, 10, "γ=10 + self-anchor 0.5"),
    "B4_g3_noanchor":   (5.0, 1.0, 0.0, 0.0, 3,  "γ=3 no anchor (v9.1 ref)"),
    "B5_g10_noanchor":  (5.0, 1.0, 0.0, 0.0, 10, "γ=10 no anchor (v9.1 ref)"),
}

PHASE_A = ["A0_baseline", "A1_sa03", "A2_sa05", "A3_sa07", "A4_ce03"]
PHASE_B = ["B1_g3_sa05", "B2_g5_sa05", "B3_g10_sa05", "B4_g3_noanchor", "B5_g10_noanchor"]


def run_self_anchor_sweep(configs_to_run, n_rounds=50, seed=42, device="cuda"):
    """Run self-anchor distillation sweep."""

    results = {
        "experiment": "v9_2_self_anchor",
        "description": "Self-anchor distillation: α_sa sweep + γ sweep",
        "version": "v9.2",
        "seed": seed,
        "n_rounds": n_rounds,
        "configs": {k: CONFIGS[k] for k in configs_to_run},
        "runs": {},
    }

    total_start = time.time()

    for config_key in configs_to_run:
        C, T, ce_alpha, sa_alpha, gamma, desc = CONFIGS[config_key]
        print("\n" + "=" * 70)
        print("  %s: %s | γ=%d | α_sa=%.1f | %d rounds" % (
            config_key, desc, gamma, sa_alpha, n_rounds))
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
                "self_anchor_alpha": sa_alpha,
                "use_denoising": False,
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
    print("  v9.2 SELF-ANCHOR SWEEP — SUMMARY")
    print("  (%d rounds, seed=%d)" % (n_rounds, seed))
    print("=" * 85)
    print("  {:<20s} {:>4s} {:>4s} {:>4s} {:>4s} {:>6s} {:>6s} {:>6s} {:>7s} {:>5s} {:>5s}".format(
        "Config", "C", "T", "αce", "αsa", "R1", "Final", "Best", "Delta", "Part", "Eps"))
    print("  " + "-" * 85)

    for config_key in configs_to_run:
        C, T, ce_alpha, sa_alpha, gamma, desc = CONFIGS[config_key]
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
        print("  {:<20s} {:4.0f} {:4.0f} {:4.1f} {:4.1f} {:5.1f}% {:5.1f}% {:5.1f}% {:+5.1f}% {:4.0f}% {:5.2f} {}".format(
            config_key, C, T, ce_alpha, sa_alpha, r1*100, final*100, best*100, delta*100,
            avg_part*100, avg_eps, tag))

    # Key comparisons
    print()
    print("  KEY COMPARISONS:")

    # Q1: Does self-anchor stop degradation?
    if "A0_baseline" in results["runs"] and "A2_sa05" in results["runs"]:
        a0 = results["runs"]["A0_baseline"]["accuracies"]
        a2 = results["runs"]["A2_sa05"]["accuracies"]
        a0_delta = a0[-1] - a0[0]
        a2_delta = a2[-1] - a2[0]
        print("    Q1: Self-anchor stops degradation?")
        print("        No anchor: R1=%.1f%% Final=%.1f%% (Δ=%+.1f%%)" % (
            a0[0]*100, a0[-1]*100, a0_delta*100))
        print("        α_sa=0.5:  R1=%.1f%% Final=%.1f%% (Δ=%+.1f%%)" % (
            a2[0]*100, a2[-1]*100, a2_delta*100))
        if a2_delta > a0_delta + 0.01:
            print("        → YES! Degradation reduced by %.1f pp" % ((a2_delta - a0_delta)*100))
        else:
            print("        → NO: still degrading")

    # Q2: Which α_sa is best?
    alpha_configs = ["A1_sa03", "A2_sa05", "A3_sa07"]
    alpha_results = []
    for ac in alpha_configs:
        if ac in results["runs"]:
            accs = results["runs"][ac]["accuracies"]
            alpha_results.append((ac, accs[-1], accs[-1] - accs[0]))
    if alpha_results:
        print()
        print("    Q2: Best α_sa?")
        for ac, final, delta in alpha_results:
            sa = CONFIGS[ac][3]
            print("        α_sa=%.1f: Final=%.1f%% (Δ=%+.1f%%)" % (sa, final*100, delta*100))
        best_a = max(alpha_results, key=lambda x: x[1])
        print("        → Best: %s (Final=%.1f%%)" % (best_a[0], best_a[1]*100))

    # Q3: Self-anchor vs CE anchor?
    if "A2_sa05" in results["runs"] and "A4_ce03" in results["runs"]:
        sa = results["runs"]["A2_sa05"]["accuracies"]
        ce = results["runs"]["A4_ce03"]["accuracies"]
        print()
        print("    Q3: Self-anchor vs CE anchor?")
        print("        Self α=0.5: Final=%.1f%% (Δ=%+.1f%%)" % (sa[-1]*100, (sa[-1]-sa[0])*100))
        print("        CE α=0.3:   Final=%.1f%% (Δ=%+.1f%%)" % (ce[-1]*100, (ce[-1]-ce[0])*100))

    # Q4: Gamma differentiation with self-anchor?
    gamma_sa = [("B1_g3_sa05", 3), ("B2_g5_sa05", 5), ("B3_g10_sa05", 10)]
    gamma_no = [("B4_g3_noanchor", 3), ("A0_baseline", 5), ("B5_g10_noanchor", 10)]
    
    gamma_sa_results = []
    for gc, g in gamma_sa:
        if gc in results["runs"]:
            gamma_sa_results.append((g, results["runs"][gc]["accuracies"][-1]))
    
    gamma_no_results = []
    for gc, g in gamma_no:
        if gc in results["runs"]:
            gamma_no_results.append((g, results["runs"][gc]["accuracies"][-1]))

    if len(gamma_sa_results) >= 2:
        print()
        print("    Q4: γ differentiation with self-anchor?")
        for g, final in sorted(gamma_sa_results):
            print("        γ=%d + self-anchor: Final=%.1f%%" % (g, final*100))
        gap_sa = max(f for _, f in gamma_sa_results) - min(f for _, f in gamma_sa_results)
        print("        Self-anchor γ gap: %.1f%%" % (gap_sa*100))
        
        if gamma_no_results:
            gap_no = max(f for _, f in gamma_no_results) - min(f for _, f in gamma_no_results)
            print("        No-anchor γ gap:   %.1f%% (v9.1 was 0.9%%)" % (gap_no*100))
            if gap_sa > gap_no + 0.005:
                print("        → YES! Self-anchor improves γ differentiation")
            else:
                print("        → NO: γ gap not improved")

    print()
    print("  Total time: %.0f seconds (%.1f hours)" % (total_time, total_time / 3600))

    # Save
    out_path = "results/experiments/v9_2_self_anchor.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2, default=str)
    print("\n  Saved to %s" % out_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v9.2: Self-Anchor Distillation")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Config keys to run (default: all)")
    parser.add_argument("--phase", type=str, default=None, choices=["A", "B"],
                        help="Run only Phase A (alpha sweep) or Phase B (gamma sweep)")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--quick", action="store_true", help="10 rounds")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        args.rounds = 10

    # Determine which configs to run
    if args.configs:
        configs_to_run = args.configs
    elif args.phase == "A":
        configs_to_run = PHASE_A
    elif args.phase == "B":
        configs_to_run = PHASE_B
    else:
        configs_to_run = PHASE_A + PHASE_B

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    else:
        print("Using device: %s" % args.device)
        if args.device == "cuda":
            print("  GPU: %s" % torch.cuda.get_device_name(0))

    # Validate config keys
    for k in configs_to_run:
        if k not in CONFIGS:
            print("Unknown config: %s. Available: %s" % (k, list(CONFIGS.keys())))
            sys.exit(1)

    print()
    print("=" * 70)
    print("  v9.2: Self-Anchor Distillation")
    print("=" * 70)
    print()
    print("  Problem: Pure KL degrades (Δ=-4.4%), CE anchor masks γ effect")
    print("  Fix: Self-anchor = KL with server's own previous logits")
    print("       loss = α_sa × KL(teacher) + (1-α_sa) × KL(self)")
    print()
    print("  Phase A: α_sa sweep {0, 0.3, 0.5, 0.7} + CE reference")
    print("  Phase B: γ sweep {3, 5, 10} with best α_sa")
    print()
    print("  Configs: %s" % configs_to_run)
    print("  Rounds: %d" % args.rounds)
    print("  Seed: %d" % args.seed)
    print()

    run_self_anchor_sweep(
        configs_to_run=configs_to_run,
        n_rounds=args.rounds,
        seed=args.seed,
        device=args.device,
    )

    print("\nv9.2 sweep complete!")
