#!/usr/bin/env python3
"""
Phase 0.2: v8.2 Verification — Class-Conditional Denoising
============================================================

v8.0: Pure KL, no denoising → catastrophic drift (ALL configs → 7-16%)
v8.1: CE anchor (α=0.5) → CE dominates, KL is noise, γ irrelevant
v8.2: Class-conditional denoising → SNR ÷ √n_c → pure KL works

Key insight: 20K public samples, 100 classes → ~200 samples/class.
Class averaging reduces noise std by √200 ≈ 14×.
SNR goes from ~0.8 (unusable) to ~11 (clean enough for pure KL).

This script tests 8 configs to verify:
  D1: v8.2 default (denoising + pure KL, γ=5)        ← THE FIX
  D2: v8.2 denoising + pure KL, γ=3                   ← lower participation
  D3: v8.2 denoising + pure KL, γ=10                  ← higher participation
  D4: v8.0 baseline (no denoising, no anchor, γ=5)    ← should still fail
  D5: v8.1 CE anchor (α=0.5, no denoising, γ=5)      ← CE-propped baseline
  D6: v8.2 denoising + CE anchor (α=0.3, γ=5)        ← belt+suspenders
  D7: FedMD oracle (no noise, no denoising)            ← upper bound
  D8: FedMD oracle + denoising                         ← should match D7

Expected:
  D1 > D4 (denoising helps)
  D1 ≈ D5 or D1 > D5 (denoising > CE anchor)
  D1 accuracy should IMPROVE or stabilize over rounds
  D2 < D1 < D3 (γ differentiation via SNR!)
  D7 ≈ D8 (denoising is no-op for clean logits)

If D2 < D1 < D3: GAME MECHANISM WORKS! γ affects accuracy through SNR.

Usage:
  python scripts/run_phase0_v8_2.py                # all 8 configs, 20 rounds
  python scripts/run_phase0_v8_2.py --quick         # 5 rounds
  python scripts/run_phase0_v8_2.py --config 1 4 7  # specific configs
"""

import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_all_experiments import run_single_experiment


VERIFY_CONFIGS = {
    1: {
        "label": "v8.2 denoising + pure KL (gamma=5)",
        "desc": "THE FIX: class-cond denoise, no CE anchor",
        "method_config": {
            "gamma": 5.0, "delta": 0.01, "clip_bound": 2.0,
            "distill_lr": 0.001, "distill_epochs": 1, "temperature": 3.0,
            "ce_anchor_alpha": 0.0, "use_denoising": True,
        },
    },
    2: {
        "label": "v8.2 denoising + pure KL (gamma=3)",
        "desc": "Lower gamma = fewer participants = higher per-device noise",
        "method_config": {
            "gamma": 3.0, "delta": 0.01, "clip_bound": 2.0,
            "distill_lr": 0.001, "distill_epochs": 1, "temperature": 3.0,
            "ce_anchor_alpha": 0.0, "use_denoising": True,
        },
    },
    3: {
        "label": "v8.2 denoising + pure KL (gamma=10)",
        "desc": "Higher gamma = more participants = lower noise",
        "method_config": {
            "gamma": 10.0, "delta": 0.01, "clip_bound": 2.0,
            "distill_lr": 0.001, "distill_epochs": 1, "temperature": 3.0,
            "ce_anchor_alpha": 0.0, "use_denoising": True,
        },
    },
    4: {
        "label": "v8.0 baseline (no denoise, no anchor, gamma=5)",
        "desc": "Control: pure noisy KL, should still degrade",
        "method_config": {
            "gamma": 5.0, "delta": 0.01, "clip_bound": 2.0,
            "distill_lr": 0.001, "distill_epochs": 1, "temperature": 3.0,
            "ce_anchor_alpha": 0.0, "use_denoising": False,
        },
    },
    5: {
        "label": "v8.1 CE anchor alpha=0.5 (no denoise, gamma=5)",
        "desc": "CE-propped baseline from v8.1",
        "method_config": {
            "gamma": 5.0, "delta": 0.01, "clip_bound": 2.0,
            "distill_lr": 0.001, "distill_epochs": 1, "temperature": 3.0,
            "ce_anchor_alpha": 0.5, "use_denoising": False,
        },
    },
    6: {
        "label": "v8.2 denoise + CE anchor alpha=0.3 (gamma=5)",
        "desc": "Belt+suspenders: denoising AND CE anchor",
        "method_config": {
            "gamma": 5.0, "delta": 0.01, "clip_bound": 2.0,
            "distill_lr": 0.001, "distill_epochs": 1, "temperature": 3.0,
            "ce_anchor_alpha": 0.3, "use_denoising": True,
        },
    },
    7: {
        "label": "FedMD oracle (no noise, no denoise)",
        "desc": "Upper bound: clean logits, pure KL",
        "method_config": {
            "ce_anchor_alpha": 0.0, "use_denoising": False,
        },
        "method_name": "FedMD",
    },
    8: {
        "label": "FedMD oracle + denoising",
        "desc": "Sanity: denoising on clean logits (should ≈ D7)",
        "method_config": {
            "ce_anchor_alpha": 0.0, "use_denoising": True,
        },
        "method_name": "FedMD",
    },
}


def run_verification(config_ids, n_rounds=20, seed=42, device="cuda", quick=False):
    if quick:
        n_rounds = 5

    results = {
        "experiment": "v8_2_phase0_verification",
        "version": "v8.2",
        "seed": seed,
        "n_rounds": n_rounds,
        "runs": {},
    }

    for cid in config_ids:
        dc = VERIFY_CONFIGS[cid]
        label = "D%d: %s" % (cid, dc["label"])
        method_name = dc.get("method_name", "PAID-FD")

        print("\n" + "=" * 70)
        print("  Config D%d: %s" % (cid, dc["label"]))
        print("  %s" % dc["desc"])
        print("  Method: %s, %d rounds, seed=%d" % (method_name, n_rounds, seed))
        print("=" * 70)

        gamma_val = dc["method_config"].get("gamma", 5.0)

        config = {
            "n_devices": 50,
            "gamma": gamma_val,
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

        # Print trajectory
        accs = run["accuracies"]
        print("\n  Round-by-round accuracy:")
        for i, a in enumerate(accs):
            marker = ""
            if i == 0:
                marker = " <- R1"
            elif a > accs[0] + 0.01:
                marker = " ^ improved"
            elif a < accs[0] - 0.02:
                marker = " v degraded"
            print("    R%2d: %.4f%s" % (i, a, marker))

        diff = accs[-1] - accs[0]
        print("\n  Final: %.4f, Best: %.4f, delta: %+.4f, Time: %.0fs"
              % (accs[-1], max(accs), diff, run["elapsed_sec"]))

    # ====== Summary ======
    print("\n" + "=" * 75)
    print("  v8.2 DENOISING VERIFICATION SUMMARY")
    print("=" * 75)
    print("  %-48s %6s %6s %6s %7s" % ("Config", "R1", "Final", "Best", "Delta"))
    print("  %s %s %s %s %s" % ("-" * 48, "-" * 6, "-" * 6, "-" * 6, "-" * 7))

    for label, run in results["runs"].items():
        accs = run["accuracies"]
        r1 = accs[0]
        final = accs[-1]
        best = max(accs)
        delta = final - r1
        if delta > 0.01:
            status = "OK"
        elif delta > -0.02:
            status = "~"
        else:
            status = "X"
        print("  %-48s %5.1f%% %5.1f%% %5.1f%% %+5.1f%% %s"
              % (label[:48], r1 * 100, final * 100, best * 100, delta * 100, status))

    print()
    print("  Interpretation guide:")
    print("    D1 > D4: Denoising helps (core hypothesis)")
    print("    D1 > D5: Denoising > CE anchor (correct fix > band-aid)")
    print("    D2 < D1 < D3: GAMMA DIFFERENTIATION WORKS!")
    print("    D7 ~ D8: Denoising no-op on clean logits (sanity)")
    print("    D1 OK: Pure KL works with denoised teacher")

    # Save
    out_path = "results/experiments/v8_2_phase0_verification.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved to %s" % out_path)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0.2: v8.2 Denoising Verification")
    parser.add_argument("--config", type=int, nargs="+", default=list(range(1, 9)),
                        help="Config IDs to run (1-8)")
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
    print("Rounds: %d" % args.rounds)
    print("\nv8.2 hypothesis: class-conditional denoising makes pure KL viable")
    print("SNR improvement: 20K samples / 100 classes = 200/class -> noise / sqrt(200)\n")

    run_verification(
        config_ids=args.config,
        n_rounds=args.rounds,
        seed=args.seed,
        device=args.device,
        quick=args.quick,
    )
    print("\nv8.2 verification complete!")
