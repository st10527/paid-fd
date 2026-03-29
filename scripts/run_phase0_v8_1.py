#!/usr/bin/env python3
"""
Phase 0.2: v8.1 Verification — Test CE-anchored distillation fixes.
=====================================================================

v8.0 diagnosis (Phase 0.1): ALL configs degraded monotonically.
Root causes: (1) Adam state leak, (2) no CE anchor, (3) cumulative drift.

v8.1 fixes: Fresh SGD per round + CE anchor (α * CE + (1-α) * KL).

This script tests 6 configs to verify the fix works:
  V1: PAID-FD v8.1 default (α=0.5, SGD, γ=5)        — balanced anchor
  V2: PAID-FD v8.1 α=0.3                              — more KL weight
  V3: PAID-FD v8.1 α=0.7                              — stronger anchor
  V4: PAID-FD v8.1 α=0.0 (no anchor, SGD only)        — isolate SGD fix
  V5: PAID-FD v8.1 α=1.0 (pure CE, no KL)             — ceiling/sanity
  V6: FedMD oracle v8.1 (α=0.5, no noise)             — upper bound

Expected:
  - V1-V3 should IMPROVE or at least stay stable (not degrade)
  - V4 should degrade less than v8.0 (SGD fix alone helps somewhat)
  - V5 should stay at pretrain level (pure CE = retrain on public)
  - V6 should be best (no noise + CE anchor)
  - If V1/V2 > V5: the KL component adds private data knowledge! 

Usage:
  python scripts/run_phase0_v8_1.py                # all 6 configs, 20 rounds
  python scripts/run_phase0_v8_1.py --quick         # 5 rounds
  python scripts/run_phase0_v8_1.py --config 1 6    # specific configs
"""

import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_all_experiments import run_single_experiment


VERIFY_CONFIGS = {
    1: {
        "label": "v8.1 default (α=0.5, γ=5)",
        "desc": "Balanced CE anchor + KL distillation",
        "method_config": {
            "gamma": 5.0, "delta": 0.01,
            "clip_bound": 2.0,
            "distill_lr": 0.001,
            "distill_epochs": 1,
            "temperature": 3.0,
            "ce_anchor_alpha": 0.5,
        },
    },
    2: {
        "label": "v8.1 α=0.3 (more KL weight)",
        "desc": "30% CE anchor, 70% KL — more teacher influence",
        "method_config": {
            "gamma": 5.0, "delta": 0.01,
            "clip_bound": 2.0,
            "distill_lr": 0.001,
            "distill_epochs": 1,
            "temperature": 3.0,
            "ce_anchor_alpha": 0.3,
        },
    },
    3: {
        "label": "v8.1 α=0.7 (strong anchor)",
        "desc": "70% CE anchor, 30% KL — conservative distillation",
        "method_config": {
            "gamma": 5.0, "delta": 0.01,
            "clip_bound": 2.0,
            "distill_lr": 0.001,
            "distill_epochs": 1,
            "temperature": 3.0,
            "ce_anchor_alpha": 0.7,
        },
    },
    4: {
        "label": "v8.1 α=0.0 (SGD only, no anchor)",
        "desc": "Pure KL + fresh SGD — isolate optimizer fix",
        "method_config": {
            "gamma": 5.0, "delta": 0.01,
            "clip_bound": 2.0,
            "distill_lr": 0.001,
            "distill_epochs": 1,
            "temperature": 3.0,
            "ce_anchor_alpha": 0.0,
        },
    },
    5: {
        "label": "v8.1 α=1.0 (pure CE, no KL)",
        "desc": "Only CE on public data — baseline ceiling",
        "method_config": {
            "gamma": 5.0, "delta": 0.01,
            "clip_bound": 2.0,
            "distill_lr": 0.001,
            "distill_epochs": 1,
            "temperature": 3.0,
            "ce_anchor_alpha": 1.0,
        },
    },
    6: {
        "label": "FedMD oracle v8.1 (α=0.5, no noise)",
        "desc": "Noise-free FD with CE anchor — upper bound",
        "method_config": {
            "ce_anchor_alpha": 0.5,
        },
        "method_name": "FedMD",
    },
}


def run_verification(config_ids, n_rounds=20, seed=42, device="cuda", quick=False):
    """Run v8.1 verification configs."""
    if quick:
        n_rounds = 5

    results = {
        "experiment": "v8_1_phase0_verification",
        "version": "v8.1",
        "seed": seed,
        "n_rounds": n_rounds,
        "runs": {},
    }

    for cid in config_ids:
        dc = VERIFY_CONFIGS[cid]
        label = "V%d: %s" % (cid, dc["label"])
        method_name = dc.get("method_name", "PAID-FD")

        print("\n" + "=" * 70)
        print("  Config V%d: %s" % (cid, dc["label"]))
        print("  %s" % dc["desc"])
        print("  Method: %s, %d rounds, seed=%d" % (method_name, n_rounds, seed))
        print("=" * 70)

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
            "heterogeneity": {
                "config_file": "config/devices/heterogeneity.yaml"
            },
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
        accs = run["accuracies"]
        print("\n  Round-by-round accuracy:")
        for i, a in enumerate(accs):
            marker = ""
            if i == 0:
                marker = " <- first distill"
            elif a > accs[0] + 0.01:
                marker = " ^ improved"
            elif a < accs[0] - 0.02:
                marker = " v degraded"
            print("    R%2d: %.4f%s" % (i, a, marker))

        diff = accs[-1] - accs[0]
        print("\n  Final: %.4f, Best: %.4f, delta(final-R0): %+.4f, Time: %.0fs"
              % (accs[-1], max(accs), diff, run["elapsed_sec"]))

    # ====== Summary ======
    print("\n" + "=" * 70)
    print("  v8.1 VERIFICATION SUMMARY")
    print("=" * 70)
    print("  %-38s %6s %6s %6s %7s" % ("Config", "R0", "Final", "Best", "Delta"))
    print("  %s %s %s %s %s" % ("-" * 38, "-" * 6, "-" * 6, "-" * 6, "-" * 7))

    for label, run in results["runs"].items():
        accs = run["accuracies"]
        r0 = accs[0]
        final = accs[-1]
        best = max(accs)
        delta = final - r0
        if delta > 0.01:
            status = "OK"
        elif delta > -0.02:
            status = "~"
        else:
            status = "X"
        print("  %-38s %5.1f%% %5.1f%% %5.1f%% %+5.1f%% %s"
              % (label[:38], r0 * 100, final * 100, best * 100, delta * 100, status))

    print()
    print("  Key: OK=improving  ~=stable  X=still degrading")
    print("  Pretrain baseline: ~44.4%%")
    print()
    print("  Expected results:")
    print("    V1-V3: Should improve or stay stable (CE anchor works)")
    print("    V4: May still degrade (SGD fix alone insufficient)")
    print("    V5: Should stay ~pretrain level (pure CE retraining)")
    print("    V6: Best overall (no noise + CE anchor)")
    print("    If V1 > V5: KL component successfully adds private knowledge!")

    # Save
    out_path = "results/experiments/v8_1_phase0_verification.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved to %s" % out_path)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0.2: v8.1 Verification")
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
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    else:
        print("Using device: %s" % args.device)
        if args.device == "cuda":
            print("  GPU: %s" % torch.cuda.get_device_name(0))

    print("\nConfigs to test: %s" % args.config)
    print("Rounds per config: %d" % args.rounds)
    print("Testing v8.1 fixes: fresh SGD + CE anchor\n")

    run_verification(
        config_ids=args.config,
        n_rounds=args.rounds,
        seed=args.seed,
        device=args.device,
        quick=args.quick,
    )
    print("\nv8.1 verification complete!")
