#!/usr/bin/env python3
"""
run_phase0_task.py — run a single Phase-0 (main γ sweep) PAID-FD experiment.

Called by run_v2_experiments.py --phase 0 --task-id N.
Results saved to TMC_V2_OUTDIR (or results/experiments/tmc_v2/ by default).
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTDIR = Path(os.environ.get("TMC_V2_OUTDIR",
                             str(ROOT / "results/experiments/tmc_v2")))

# Import phase-0 config list from the batch runner
sys.path.insert(0, str(ROOT / "scripts"))
from run_v2_experiments import PHASE0_CONFIGS  # noqa: E402

# Import experiment runner
from scripts.run_all_experiments import run_single_experiment  # noqa: E402


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rounds", type=int, default=100)
    args = parser.parse_args()

    if args.task_id < 0 or args.task_id >= len(PHASE0_CONFIGS):
        print(f"ERROR: task-id {args.task_id} out of range "
              f"[0, {len(PHASE0_CONFIGS)})")
        sys.exit(1)

    spec = PHASE0_CONFIGS[args.task_id]
    label = spec["label"]
    outfile = OUTDIR / f"{label}.json"

    if outfile.exists():
        print(f"[SKIP] {label} already done: {outfile}")
        return

    print("=" * 70)
    print(f"  Phase 0 Task {args.task_id+1}/{len(PHASE0_CONFIGS)}")
    print(f"  Label : {label}")
    print(f"  Desc  : {spec['desc']}")
    print(f"  Device: {args.device}  Rounds: {args.rounds}")
    print("=" * 70)

    t0 = time.time()
    run = run_single_experiment(
        method_name=spec["method"],
        config=spec["config"],
        seed=spec["seed"],
        device=args.device,
        n_rounds=args.rounds,
        save_decisions=False,
        verbose=True,
    )
    elapsed = time.time() - t0

    accs = run.get("accuracies", [])
    parts = run.get("participation_rates", [])
    prices = run.get("prices", [])
    avg_eps_list = run.get("avg_eps", [])
    extras = run.get("extras", [])

    import numpy as np
    final_acc = accs[-1] if accs else 0
    best_acc = max(accs) if accs else 0
    avg_part = float(np.mean(parts)) if parts else 0
    avg_price = float(np.mean(prices)) if prices else 0
    avg_eps = float(np.mean(avg_eps_list)) if avg_eps_list else 0
    last_extra = extras[-1] if extras else {}
    cum_payment = last_extra.get("cumulative_payment", 0)
    avg_priv = last_extra.get("avg_privacy_spent", 0)
    max_priv = last_extra.get("max_privacy_spent", 0)

    targets = [0.45, 0.50, 0.55, 0.58, 0.60]
    ttt = {}
    for t in targets:
        reached = [i for i, a in enumerate(accs) if a >= t]
        ttt[str(t)] = reached[0] if reached else None

    summary = {
        "label": label, "exp": "0", "method": spec["method"],
        "seed": spec["seed"], "desc": spec["desc"],
        "final_acc": final_acc, "best_acc": best_acc,
        "avg_participation": avg_part, "avg_price": avg_price,
        "avg_eps_per_round": avg_eps,
        "cumulative_payment": cum_payment,
        "avg_privacy_spent": avg_priv, "max_privacy_spent": max_priv,
        "time_to_targets": ttt, "elapsed_sec": elapsed,
    }

    result = {
        "label": label, "phase": 0, "exp": "0",
        "method": spec["method"], "seed": spec["seed"],
        "desc": spec["desc"], "dataset": "cifar100",
        "summary": summary,
        "accuracies": accs,
        "participation_rates": parts,
        "prices": prices,
        "avg_eps": avg_eps_list,
        "config": spec["config"],
        "completed_at": datetime.now().isoformat(),
        "elapsed_sec": elapsed,
    }
    save_json(result, str(outfile))

    print("\n" + "=" * 70)
    print(f"  DONE: {label}")
    print(f"  best={best_acc*100:.2f}%  final={final_acc*100:.2f}%  "
          f"part={avg_part:.1%}  ε*={avg_eps:.3f}")
    print(f"  time={elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  saved: {outfile}")
    print("=" * 70)


if __name__ == "__main__":
    main()
