#!/usr/bin/env python3
"""
Phase 0 v9.0: Gamma Sweep with FIXED Cubic Solver
=====================================================

THE BUG:
  The cubic equation f(ε) = λε³ + λε² + (1-p)ε + 1 = 0 has TWO positive
  roots when p > p_threshold. The old solver (bisection from small ε upward)
  always found Root 1 (the SMALLER root), which:
  - Gave ε* ≈ 0.3-0.8 → SNR < 1 → noise destroys signal
  - Made p↑ → ε*↓ (VIOLATING Proposition 2)
  - Produced identical degradation for all γ values

THE FIX:
  The new solver uses np.roots() to find ALL roots analytically, then picks
  the one that maximizes device utility. This gives Root 2 (the LARGER root):
  - ε* ≈ 2-8 → SNR = 6-433 → signal dominates
  - p↑ → ε*↑ (Proposition 2 holds perfectly)
  - Higher γ → higher price → more participants AND lower noise per device

SETUP (D4 config from v8.3, now with fixed solver):
  - Method: PAID-FD (Stackelberg game + BLUE + LDP)
  - Distillation: pure KL (ce_anchor_alpha=0.0)
  - Denoising: OFF (use_denoising=False)
  - Optimizer: fresh SGD per round (momentum=0.9, wd=5e-4)
  - T=3, distill_lr=0.001, clip_bound=2.0
  - 50 devices, CIFAR-100, Dirichlet α=0.5, ResNet-18

EXPECTED (from solver verification):
  γ=2:  p*=0.01 → 0% participation (below threshold)
  γ=3:  p*=2.22 → 22 devices, avg_ε=2.69, agg_var=0.17
  γ=5:  p*=2.97 → 41 devices, avg_ε=2.92, agg_var=0.07
  γ=7:  p*=3.63 → 48 devices, avg_ε=3.25, agg_var=0.05
  γ=10: p*=3.71 → 48 devices, avg_ε=3.32, agg_var=0.05

HYPOTHESIS:
  With SNR >> 1, distillation should IMPROVE accuracy (not degrade it).
  Higher γ → lower agg_var → cleaner distillation → higher accuracy.
  Gap between γ=3 and γ=10 should be > 3%.

Usage:
  python scripts/run_phase0_v9_gamma_sweep.py                   # full (100 rounds)
  python scripts/run_phase0_v9_gamma_sweep.py --quick            # quick test (5 rounds)
  python scripts/run_phase0_v9_gamma_sweep.py --rounds 50        # custom rounds
  python scripts/run_phase0_v9_gamma_sweep.py --gamma 3 5 10     # subset of gammas
"""

import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_all_experiments import run_single_experiment


def run_gamma_sweep(gammas, n_rounds=100, seed=42, device="cuda"):
    """Run PAID-FD (D4 config + fixed solver) across multiple γ values."""

    results = {
        "experiment": "v9_0_phase0_gamma_sweep_fixed_solver",
        "description": "Gamma sweep with FIXED cubic solver (Root 2). Pure BLUE, no denoise, no CE anchor.",
        "version": "v9.0",
        "bug_fix": "Cubic solver now finds utility-maximizing root (Root 2, larger eps*)",
        "seed": seed,
        "n_rounds": n_rounds,
        "gammas": gammas,
        "runs": {},
    }

    total_start = time.time()

    for gamma in gammas:
        label = "gamma=%s" % gamma
        print("\n" + "=" * 70)
        print("  v9.0 PAID-FD (fixed solver) | gamma=%s | seed=%d | %d rounds" % (gamma, seed, n_rounds))
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
            "temperature": 3.0,
            "public_samples": 20000,
            "synthetic": False,
            "heterogeneity": {
                "config_file": "config/devices/heterogeneity.yaml",
            },
            "method_config": {
                "gamma": float(gamma),
                "delta": 0.01,
                "clip_bound": 2.0,
                "distill_lr": 0.001,
                "distill_epochs": 1,
                "temperature": 3.0,
                "ce_anchor_alpha": 0.0,      # Pure KL — no CE anchor
                "use_denoising": False,       # No denoising — trust BLUE only
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

        # Print trajectory summary
        accs = run["accuracies"]
        r1 = accs[0]
        final = accs[-1]
        best = max(accs)
        best_round = accs.index(best)
        parts = run.get("participation_rates", [])
        avg_part = sum(parts) / len(parts) if parts else 0
        eps_list = run.get("avg_eps", [])
        avg_eps = sum(eps_list) / len(eps_list) if eps_list else 0

        print("\n  R1=%.4f  Final=%.4f  Best=%.4f (R%d)  Part=%.0f%%  Eps=%.3f  Time=%.0fs"
              % (r1, final, best, best_round, avg_part * 100, avg_eps, run["elapsed_sec"]))

        # Print every 10th round
        print("  Trajectory (every 10 rounds):")
        for i in range(0, len(accs), 10):
            print("    R%3d: %.4f" % (i, accs[i]))
        if (len(accs) - 1) % 10 != 0:
            print("    R%3d: %.4f (final)" % (len(accs) - 1, accs[-1]))

    total_time = time.time() - total_start

    # ====== SUMMARY TABLE ======
    print("\n" + "=" * 75)
    print("  v9.0 GAMMA SWEEP — FIXED CUBIC SOLVER — SUMMARY")
    print("  (Pure KL, no denoise, fresh SGD, %d rounds, seed=%d)" % (n_rounds, seed))
    print("=" * 75)
    print("  %-12s %6s %6s %6s %7s %5s %5s %6s" % (
        "Config", "R1", "Final", "Best", "Delta", "Part", "Eps", "Time"))
    print("  %s" % ("-" * 68))

    finals = []
    for label, run in results["runs"].items():
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
        print("  %-12s %5.1f%% %5.1f%% %5.1f%% %+5.1f%% %4.0f%% %5.3f %5.0fs %s" % (
            label, r1 * 100, final * 100, best * 100, delta * 100,
            avg_part * 100, avg_eps, run["elapsed_sec"], tag))
        finals.append((label, final, best, avg_part, avg_eps))

    # ====== GAP ANALYSIS ======
    print()
    active_finals = [(l, f, b, p, e) for l, f, b, p, e in finals if p > 0.05]

    if len(active_finals) >= 2:
        best_label, best_final = max(active_finals, key=lambda x: x[1])[:2]
        worst_label, worst_final = min(active_finals, key=lambda x: x[1])[:2]
        gap_final = best_final - worst_final

        best_best_label, best_best = max(active_finals, key=lambda x: x[2])[:2]
        worst_best_label, worst_best = min(active_finals, key=lambda x: x[2])[:2]
        gap_best = best_best - worst_best

        print("  GAMMA DIFFERENTIATION (excluding 0%% participation):")
        print("    Final accuracy gap: %.1f%% (%s=%.1f%% vs %s=%.1f%%)" % (
            gap_final * 100, best_label, best_final * 100, worst_label, worst_final * 100))
        print("    Best accuracy gap:  %.1f%% (%s=%.1f%% vs %s=%.1f%%)" % (
            gap_best * 100, best_best_label, best_best * 100, worst_best_label, worst_best * 100))

        print()
        if gap_final > 0.03:
            print("  ✅ VERDICT: GAP > 3%% — Solver fix resolved γ differentiation!")
            print("  The bug was finding the wrong cubic root (Root 1 instead of Root 2).")
            print("  With correct root: SNR >> 1, Proposition 2 holds, γ differentiates.")
            print("  Next: Full Phase 1-3 experiments with 3 seeds, all methods.")
        elif gap_final > 0.01:
            print("  ⚠️  VERDICT: Gap 1-3%% — marginal improvement")
            print("  Solver fix helped but may need CE anchor or more rounds.")
        else:
            print("  ❌ VERDICT: Gap < 1%% — solver fix alone insufficient")
            print("  Other factors (distillation dynamics?) may dominate.")
    else:
        print("  WARNING: Too few active configs for gap analysis")

    # Ordering check
    ordered = sorted(active_finals, key=lambda x: float(x[0].split("=")[1]))
    acc_ordered = [f for _, f, _, _, _ in ordered]
    is_monotone = all(acc_ordered[i] <= acc_ordered[i+1] for i in range(len(acc_ordered)-1))
    print()
    print("  Monotonicity (higher γ → higher accuracy): %s" % (
        "✅ YES" if is_monotone else "❌ NO"))
    for l, f, b, p, e in ordered:
        print("    %s: final=%.2f%%  best=%.2f%%  part=%.0f%%  eps=%.3f" % (
            l, f * 100, b * 100, p * 100, e))

    # Compare vs v8.3 (wrong root) results if available
    v8_3_path = "results/experiments/v8_3_phase0_gamma_sweep.json"
    if os.path.exists(v8_3_path):
        print()
        print("  COMPARISON vs v8.3 (wrong root):")
        with open(v8_3_path) as f:
            v8_3 = json.load(f)
        for label, run in results["runs"].items():
            if label in v8_3.get("runs", {}):
                old_final = v8_3["runs"][label]["accuracies"][-1]
                new_final = run["accuracies"][-1]
                diff = new_final - old_final
                print("    %s: v8.3=%.2f%% → v9.0=%.2f%% (%+.1f%%)" % (
                    label, old_final * 100, new_final * 100, diff * 100))

    print()
    print("  Total time: %.0f seconds (%.1f hours)" % (total_time, total_time / 3600))

    # ====== SAVE ======
    out_path = "results/experiments/v9_0_phase0_gamma_sweep_fixed_solver.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n  Saved to %s" % out_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="v9.0: Gamma sweep with FIXED cubic solver (100 rounds)")
    parser.add_argument("--gamma", type=float, nargs="+", default=[2, 3, 5, 7, 10],
                        help="Gamma values to sweep (default: 2 3 5 7 10)")
    parser.add_argument("--rounds", type=int, default=100,
                        help="Rounds per config (default: 100)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (5 rounds)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        args.rounds = 5

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    else:
        print("Using device: %s" % args.device)
        if args.device == "cuda":
            print("  GPU: %s" % torch.cuda.get_device_name(0))

    print()
    print("═" * 70)
    print("  v9.0: Gamma Sweep with FIXED Cubic Solver")
    print("═" * 70)
    print()
    print("  BUG FIXED: Cubic solver was finding Root 1 (small ε*, SNR < 1)")
    print("  NOW FINDS: Root 2 (large ε*, SNR >> 1, Proposition 2 holds)")
    print()
    print("  Config: D4 (pure KL, no denoise, fresh SGD)")
    print("  Gammas: %s" % args.gamma)
    print("  Rounds: %d" % args.rounds)
    print("  Seed: %d" % args.seed)
    print()

    run_gamma_sweep(
        gammas=args.gamma,
        n_rounds=args.rounds,
        seed=args.seed,
        device=args.device,
    )

    print("\nv9.0 sweep complete!")
