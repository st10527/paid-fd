#!/usr/bin/env python3
"""
run_v2_experiments.py — Post-cubic-fix batch experiment launcher.

After the cubic-solver bug fix (coeffs = [lam, lam, c-p, c] instead of
[lam, lam, 1-p, 1]), all PAID-FD experiments must be re-run.  This script
wraps run_tmc_experiment.py and saves results to results/experiments/tmc_v2/
so old (tmc/) and new (tmc_v2/) results are preserved side-by-side.

Usage
-----
# Dry-run: show what would be launched (no GPU required)
python scripts/run_v2_experiments.py --list
python scripts/run_v2_experiments.py --phase 1 --dry-run

# Run a single phase sequentially (one task at a time)
python scripts/run_v2_experiments.py --phase 1

# Run a specific task from a phase (for SLURM / manual retry)
python scripts/run_v2_experiments.py --phase 1 --task-id 0

# Run all phases in priority order
python scripts/run_v2_experiments.py --all

# Run T=10 quick diagnostic (verifies epsilon shift, ~15 min)
python scripts/run_v2_experiments.py --diagnostic

Priority order for GPU time
---------------------------
Phase 1 (VI-B/C)   : 33 runs — γ sweep, N sweep, ablation (core paper claims)
Phase 4 (VI-F/G/H) : 12 runs — Fair Fixed-ε, privacy-utility curve, BLUE valid.
Phase 3 (VI-E)     : 12 runs — Non-IID α sweep
Phase 2 (VI-D)     : 9 runs  — CIFAR-10 cross-dataset
Phase 5 (VI-I)     : 3 runs  — Pipeline component ablation
                     -------
Total               : 69 runs  (~7–8 GPU-days on RTX 5070 Ti)

Key difference from tmc/ runs
------------------------------
- Cubic solver fix: ε* now ~+0.3–1.5 higher across all (p,c,λ) combos
- Expected avg ε̄* at γ=5: ~3.3 (was ~2.84)
- s* nearly unchanged (compensating shift): ~13 logit vectors/device
- Output accuracy expected ≥ old (higher ε → lower noise)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "run_tmc_experiment.py"
V2_OUTDIR = ROOT / "results" / "experiments" / "tmc_v2"
LOG_DIR = ROOT / "results" / "logs"

# ── phase sizes (must match run_tmc_experiment.py) ─────────────────────────
PHASE_SIZES = {1: 33, 2: 9, 3: 12, 4: 12, 5: 3}

# ── priority order for --all ───────────────────────────────────────────────
ALL_PHASES_ORDERED = [1, 4, 3, 2, 5]


# ────────────────────────────────────────────────────────────────────────────
def outfile_for(phase: int, task_id: int, label: str) -> Path:
    """Return the expected output path for a run."""
    return V2_OUTDIR / f"{label}.json"


def label_for(phase: int, task_id: int) -> str | None:
    """Ask the runner for the label of a task via --dry-run."""
    cmd = [
        sys.executable, str(RUNNER),
        "--phase", str(phase),
        "--task-id", str(task_id),
        "--dry-run",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL,
                                      cwd=str(ROOT))
        for line in out.decode().splitlines():
            if line.strip().startswith("Label:"):
                return line.split(":", 1)[1].strip()
    except subprocess.CalledProcessError:
        pass
    return None


def list_phase(phase: int) -> None:
    """Print task list for a phase."""
    cmd = [sys.executable, str(RUNNER), "--phase", str(phase), "--list"]
    subprocess.run(cmd, cwd=str(ROOT))


def run_task(phase: int, task_id: int, device: str, rounds: int,
             dry: bool = False, seed_override: int | None = None) -> bool:
    """
    Launch one task.  Returns True if the run completed (or was skipped),
    False if it failed.
    """
    # Build base command
    cmd = [
        sys.executable, str(RUNNER),
        "--phase", str(phase),
        "--task-id", str(task_id),
        "--device", device,
        "--rounds", str(rounds),
    ]
    if phase == 5 and seed_override is not None:
        cmd += ["--seed", str(seed_override)]
    if dry:
        cmd.append("--dry-run")

    # Patch OUTDIR so results land in tmc_v2 instead of tmc
    env = os.environ.copy()
    env["TMC_V2_OUTDIR"] = str(V2_OUTDIR)

    print(f"\n{'─'*68}")
    print(f"  Phase {phase}  Task {task_id:2d}/{PHASE_SIZES[phase]-1}",
          end="")
    if dry:
        print("  [DRY-RUN]")
    else:
        print()

    t0 = time.time()
    ret = subprocess.run(cmd, cwd=str(ROOT), env=env)
    elapsed = time.time() - t0

    if ret.returncode == 0:
        if not dry:
            print(f"  ✓ completed in {elapsed/60:.1f} min")
        return True
    else:
        print(f"  ✗ FAILED (code {ret.returncode})")
        return False


# ────────────────────────────────────────────────────────────────────────────
def run_diagnostic(device: str = "cuda") -> None:
    """
    Quick T=10 sanity-check: PAID-FD γ=5 seed=42 for 10 rounds only.

    Verifies:
      • No crash after cubic fix
      • ε* shift is in the expected direction (+0.3 to +1.5)
      • Participation/payment structure looks sane
      • ~15 min wall-clock on RTX 5070 Ti
    """
    print("\n" + "=" * 68)
    print("  T=10 DIAGNOSTIC — verifying cubic-fix post-conditions")
    print("=" * 68)

    cmd = [
        sys.executable, str(RUNNER),
        "--phase", "1",
        "--task-id", "0",       # expA_fixedeps1_s42 — lightweight sanity run
        "--device", device,
        "--rounds", "10",
    ]
    # Use a temp label so it doesn't clobber real results
    env = os.environ.copy()
    env["TMC_V2_OUTDIR"] = str(V2_OUTDIR / "diagnostic")

    print(f"  Command: {' '.join(cmd)}")
    print(f"  Output : {V2_OUTDIR / 'diagnostic'}")
    print()

    t0 = time.time()
    ret = subprocess.run(cmd, cwd=str(ROOT), env=env)
    elapsed = time.time() - t0

    if ret.returncode == 0:
        print(f"\n  ✓ Diagnostic completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")

        # Parse the output file and show key stats
        diag_dir = V2_OUTDIR / "diagnostic"
        results = sorted(diag_dir.glob("*.json")) if diag_dir.exists() else []
        if results:
            data = json.loads(results[-1].read_text())
            s = data.get("summary", {})
            accs = data.get("accuracies", [])
            avg_eps_list = data.get("avg_eps", [])
            print(f"\n  Diagnostic summary:")
            print(f"    final_acc       = {(s.get('final_acc', 0)*100):.2f}%")
            print(f"    avg_participation = {s.get('avg_participation', 0):.1%}")
            print(f"    avg_eps/round   = {s.get('avg_eps_per_round', 0):.3f}")
            if avg_eps_list:
                print(f"    ε* range        = [{min(avg_eps_list):.3f}, "
                      f"{max(avg_eps_list):.3f}]")
            if accs:
                print(f"    acc @ T=1       = {accs[0]*100:.2f}%")
                print(f"    acc @ T=10      = {accs[-1]*100:.2f}%")
        else:
            print("  (no output file found — check V2_OUTDIR patch)")
    else:
        print(f"\n  ✗ Diagnostic FAILED (code {ret.returncode})")
        sys.exit(1)


# ────────────────────────────────────────────────────────────────────────────
def run_phase(phase: int, device: str, rounds: int,
              dry: bool = False, start_task: int = 0,
              task_id: int | None = None) -> None:
    """Run all tasks in a phase, skipping completed ones."""
    n = PHASE_SIZES[phase]
    V2_OUTDIR.mkdir(parents=True, exist_ok=True)

    tasks = [task_id] if task_id is not None else list(range(start_task, n))

    total = len(tasks)
    done = 0
    failed: list[int] = []
    skipped = 0

    print(f"\n{'═'*68}")
    print(f"  Phase {phase} — {n} total tasks, running {total}")
    print(f"  Output dir: {V2_OUTDIR}")
    print(f"{'═'*68}")

    for tid in tasks:
        # Probe label to check if already complete
        lbl = label_for(phase, tid)
        if lbl is not None:
            dest = V2_OUTDIR / f"{lbl}.json"
            if dest.exists() and not dry:
                print(f"  [skip] {lbl} — already done")
                skipped += 1
                continue

        ok = run_task(phase, tid, device, rounds, dry=dry)
        if ok:
            done += 1
        else:
            failed.append(tid)

    print(f"\n{'═'*68}")
    print(f"  Phase {phase} complete — done={done}  skipped={skipped}  "
          f"failed={len(failed)}")
    if failed:
        print(f"  Failed task-ids: {failed}")
    print(f"{'═'*68}")


# ────────────────────────────────────────────────────────────────────────────
def run_all(device: str, rounds: int, dry: bool = False) -> None:
    """Run all phases in priority order."""
    total_phases = len(ALL_PHASES_ORDERED)
    for i, phase in enumerate(ALL_PHASES_ORDERED):
        print(f"\n\n{'#'*68}")
        print(f"  PHASE {phase}  ({i+1}/{total_phases})")
        print(f"{'#'*68}")
        run_phase(phase, device, rounds, dry=dry)

    # Print global summary
    all_results = sorted(V2_OUTDIR.glob("*.json"))
    print(f"\n{'═'*68}")
    print(f"  ALL PHASES DONE — {len(all_results)} results in {V2_OUTDIR}")
    print(f"{'═'*68}")


# ────────────────────────────────────────────────────────────────────────────
def patch_runner_outdir() -> None:
    """
    Monkey-patch the OUTDIR in run_tmc_experiment.py via env var.

    run_tmc_experiment.py uses a hardcoded OUTDIR = "results/experiments/tmc".
    We override it by injecting TMC_V2_OUTDIR into the subprocess environment.

    NOTE: This requires a one-line patch to run_tmc_experiment.py:
        OUTDIR = os.environ.get("TMC_V2_OUTDIR", "results/experiments/tmc")

    If the patch is not yet applied, this function applies it automatically.
    """
    runner_text = RUNNER.read_text()
    old = 'OUTDIR = "results/experiments/tmc"'
    new = 'OUTDIR = os.environ.get("TMC_V2_OUTDIR", "results/experiments/tmc")'

    if old in runner_text:
        print(f"  Patching {RUNNER.name}: OUTDIR → env-configurable")
        # Also ensure 'import os' is present
        if "import os" not in runner_text:
            runner_text = "import os\n" + runner_text
        runner_text = runner_text.replace(old, new)
        RUNNER.write_text(runner_text)
        print("  ✓ Patch applied")
    elif new in runner_text:
        print(f"  {RUNNER.name} already patched — OK")
    else:
        print(f"  WARNING: Could not patch {RUNNER.name} — "
              "results will go to tmc/ not tmc_v2/")


# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-cubic-fix batch launcher for all TMC experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run a specific phase (1–5)")
    parser.add_argument("--task-id", type=int, default=None,
                        help="Run a single task within --phase")
    parser.add_argument("--start-task", type=int, default=0,
                        help="Resume from this task-id (skip earlier tasks)")
    parser.add_argument("--all", action="store_true",
                        help="Run all phases in priority order")
    parser.add_argument("--diagnostic", action="store_true",
                        help="Quick T=10 sanity check (no full rerun needed)")
    parser.add_argument("--list", action="store_true",
                        help="List tasks in --phase (or all phases if --all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show commands without executing")
    parser.add_argument("--device", default="cuda",
                        help="torch device (default: cuda)")
    parser.add_argument("--rounds", type=int, default=100,
                        help="Training rounds (default: 100)")
    parser.add_argument("--no-patch", action="store_true",
                        help="Skip auto-patching OUTDIR in run_tmc_experiment.py")
    args = parser.parse_args()

    # ── sanity checks ───────────────────────────────────────────────────────
    if not RUNNER.exists():
        print(f"ERROR: runner not found: {RUNNER}")
        sys.exit(1)

    # ── print header ────────────────────────────────────────────────────────
    print("=" * 68)
    print("  PAID-FD v2 Batch Experiment Launcher")
    print(f"  cubic fix: coeffs = [lam, lam, c-p, c]  (was [lam, lam, 1-p, 1])")
    print(f"  output   : {V2_OUTDIR}")
    print(f"  device   : {args.device}   rounds: {args.rounds}")
    print("=" * 68)

    # ── auto-patch OUTDIR ───────────────────────────────────────────────────
    if not args.no_patch:
        patch_runner_outdir()

    V2_OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── dispatch ────────────────────────────────────────────────────────────
    if args.diagnostic:
        run_diagnostic(device=args.device)

    elif args.list:
        if args.all or args.phase is None:
            for p in ALL_PHASES_ORDERED:
                list_phase(p)
        else:
            list_phase(args.phase)

    elif args.all:
        run_all(args.device, args.rounds, dry=args.dry_run)

    elif args.phase is not None:
        run_phase(args.phase, args.device, args.rounds,
                  dry=args.dry_run,
                  start_task=args.start_task,
                  task_id=args.task_id)

    else:
        parser.print_help()
        print("\nHint: use --phase N to run a specific phase, "
              "or --all to run all phases in priority order.")
        sys.exit(1)


if __name__ == "__main__":
    main()
