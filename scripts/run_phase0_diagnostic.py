#!/usr/bin/env python3
"""
Phase 0.1: SNR Diagnostic — find parameter combo that makes v8 distillation work.
====================================================================================

Problem diagnosed from Phase 0:
  - Pretrain gives ~44% accuracy
  - ANY distillation round DESTROYS it (drops to ~25%)
  - Root cause: SNR < 1 after aggregation → teacher logits are noise
  - Contributing: Adam lr=0.001 × T²=9 → too aggressive for noisy teacher

This script tests 6 parameter configs with γ=5, 20 rounds each.
Expected total time: ~2h on RTX 5070 Ti.

Usage:
  python scripts/run_phase0_diagnostic.py                # all 6 configs
  python scripts/run_phase0_diagnostic.py --config 1 2   # specific configs
  python scripts/run_phase0_diagnostic.py --quick         # 5 rounds only
"""

import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_all_experiments import run_single_experiment


# ============================================================================
# 6 diagnostic configurations
# ============================================================================
# Each tuple: (label, method_config_overrides, heterogeneity_overrides)

DIAG_CONFIGS = {
    # --- Group 1: Fix the noise (reduce C) ---
    1: {
        "label": "C=1.0 (halve sensitivity)",
        "desc": "4x less noise variance, same lr",
        "method_config": {
            "gamma": 5.0, "delta": 0.01,
            "clip_bound": 1.0,
            "distill_lr": 0.001,
            "distill_epochs": 1,
        },
        "het_overrides": None,
    },

    # --- Group 2: Fix the learning (conservative distill) ---
    2: {
        "label": "lr=0.0001 (10x smaller)",
        "desc": "Same noise, but very gentle distillation",
        "method_config": {
            "gamma": 5.0, "delta": 0.01,
            "clip_bound": 2.0,
            "distill_lr": 0.0001,
            "distill_epochs": 1,
        },
        "het_overrides": None,
    },

    # --- Group 3: Both fixes ---
    3: {
        "label": "C=1.0 + lr=0.0001",
        "desc": "4x less noise + gentle lr",
        "method_config": {
            "gamma": 5.0, "delta": 0.01,
            "clip_bound": 1.0,
            "distill_lr": 0.0001,
            "distill_epochs": 1,
        },
        "het_overrides": None,
    },

    # --- Group 4: Higher ε regime via lower λ ---
    4: {
        "label": "λ×0.1 (higher ε regime)",
        "desc": "Game produces ε≈2-5 instead of 0.3-0.8",
        "method_config": {
            "gamma": 5.0, "delta": 0.01,
            "clip_bound": 2.0,
            "distill_lr": 0.001,
            "distill_epochs": 1,
        },
        "het_overrides": {"privacy_sensitivity": {"lambda_mult": 0.1}},
    },

    # --- Group 5: Higher ε + all fixes ---
    5: {
        "label": "λ×0.1 + C=1.0 + lr=0.0001",
        "desc": "Higher ε + lower noise + gentle lr (best combo?)",
        "method_config": {
            "gamma": 5.0, "delta": 0.01,
            "clip_bound": 1.0,
            "distill_lr": 0.0001,
            "distill_epochs": 1,
        },
        "het_overrides": {"privacy_sensitivity": {"lambda_mult": 0.1}},
    },

    # --- Group 6: FedMD oracle (no noise) to see v8 upper bound ---
    6: {
        "label": "FedMD oracle (no noise, v8 ceiling)",
        "desc": "v8 FD without any noise — what's the best possible?",
        "method_config": {},
        "het_overrides": None,
        "method_name": "FedMD",
    },
}


def run_diagnostic(config_ids, n_rounds=20, seed=42, device="cuda", quick=False):
    """Run diagnostic configs."""
    if quick:
        n_rounds = 5

    results = {
        "experiment": "v8_phase0_diagnostic",
        "version": "v8",
        "seed": seed,
        "n_rounds": n_rounds,
        "runs": {},
    }

    for cid in config_ids:
        dc = DIAG_CONFIGS[cid]
        label = f"C{cid}: {dc['label']}"
        method_name = dc.get("method_name", "PAID-FD")

        print(f"\n{'='*70}")
        print(f"  Config {cid}: {dc['label']}")
        print(f"  {dc['desc']}")
        print(f"  Method: {method_name}, {n_rounds} rounds, seed={seed}")
        print(f"{'='*70}")

        het_cfg = {"config_file": "config/devices/heterogeneity.yaml"}
        if dc["het_overrides"]:
            het_cfg["overrides"] = dc["het_overrides"]

        config = {
            "n_devices": 50,
            "gamma": 5.0,
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
            "heterogeneity": het_cfg,
            "method_config": dc["method_config"],
        }

        run = run_single_experiment(
            method_name=method_name,
            config=config,
            seed=seed,
            device=device,
            n_rounds=n_rounds,
        )
        results["runs"][label] = run

        # Print round-by-round
        accs = run['accuracies']
        print(f"\n  Round-by-round accuracy:")
        for i, a in enumerate(accs):
            marker = ""
            if i == 0:
                marker = " ← first distill"
            elif a > accs[0]:
                marker = " ↑"
            elif a < accs[0] - 0.02:
                marker = " ↓↓"
            print(f"    R{i:2d}: {a:.4f}{marker}")

        diff = accs[-1] - accs[0]
        print(f"\n  Final: {accs[-1]:.4f}, Best: {max(accs):.4f}, "
              f"Δ(final-R0): {diff:+.4f}, Time: {run['elapsed_sec']:.0f}s")

    # ====== Summary ======
    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Config':<40s} {'R0':>6s} {'Final':>6s} {'Best':>6s} {'Δ':>7s}")
    print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")

    for label, run in results["runs"].items():
        accs = run['accuracies']
        r0 = accs[0]
        final = accs[-1]
        best = max(accs)
        delta = final - r0
        status = "✅" if delta > 0 else "⚠️" if delta > -0.02 else "❌"
        print(f"  {label:<40s} {r0:>5.1%} {final:>5.1%} {best:>5.1%} {delta:>+6.1%} {status}")

    print(f"\n  Key: ✅ improving  ⚠️ stable  ❌ degrading")
    print(f"  Pretrain baseline: ~44.4%")
    print(f"  If ALL configs show ❌, the noise regime is fundamentally broken.")
    print(f"  If Config 6 (FedMD) shows ❌, the v8 flow itself has a problem.")

    # Save
    out_path = "results/experiments/v8_phase0_diagnostic.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ Saved to {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0.1: SNR Diagnostic")
    parser.add_argument("--config", type=int, nargs="+", default=list(range(1, 7)),
                        help="Config IDs to run (1-6)")
    parser.add_argument("--rounds", type=int, default=20,
                        help="Rounds per config (default: 20)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (5 rounds)")
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

    print(f"\n📋 Configs to test: {args.config}")
    print(f"   Rounds per config: {args.rounds}")
    print(f"   This tests WHY distillation destroys pretrain knowledge.\n")

    run_diagnostic(
        config_ids=args.config,
        n_rounds=args.rounds,
        seed=args.seed,
        device=args.device,
        quick=args.quick,
    )
    print("\n🏁 Diagnostic complete!")
