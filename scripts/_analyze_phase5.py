#!/usr/bin/env python3
"""
Phase 5 Pipeline Ablation — Multi-Seed Analysis
=================================================
Reads all expI_* result files (3 ablations × up to 3 seeds) and the
v10.1 baseline (γ=5, 3 seeds from the combined sweep file) to compute:

  • Per-ablation mean ± std accuracy (final and best)
  • Δ vs v10.1 baseline in percentage points
  • Classification: negligible / minor / significant

Outputs:
  results/analysis/phase5_summary.md  — human-readable Markdown table
  results/analysis/phase5_summary.csv — CSV for paper tables

Classification thresholds:
  |Δ| < 0.5 pp  → negligible
  0.5–2 pp      → minor
  > 2 pp        → significant

Usage:
  cd /path/to/paid_fd
  python scripts/_analyze_phase5.py [--best-acc]   # default: final_acc
"""

import sys, os, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
TMC_DIR    = Path("results/experiments/tmc")
COMBINED   = Path("results/experiments/v10_1_combined_20260409_2304.json")
OUT_DIR    = Path("results/analysis")

# ──────────────────────────────────────────────────────────────────────────────
# v10.1 baseline: γ=5, seeds 42/123/456  (from combined sweep)
# ──────────────────────────────────────────────────────────────────────────────
def load_baseline(metric: str) -> dict:
    """Return {seed: acc} for v10.1 γ=5 baseline."""
    data = json.loads(COMBINED.read_text())
    summaries = data["summaries"]
    seeds = [42, 123, 456]
    result = {}
    for s in seeds:
        key = "g5_s%d" % s
        if key in summaries:
            result[s] = summaries[key][metric]
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Ablation result loader
# ──────────────────────────────────────────────────────────────────────────────
ABLATIONS = [
    ("noEMA",        "No EMA",              "expI_noEMA_s{seed}.json"),
    ("noMixedLoss",  "No Mixed Loss",       "expI_noMixedLoss_s{seed}.json"),
    ("noPersistent", "No Persistent Models","expI_noPersistent_s{seed}.json"),
]

def load_ablation(pattern: str, metric: str) -> dict:
    """Return {seed: acc} for an ablation across all available seeds."""
    result = {}
    for seed in [42, 123, 456]:
        fname = TMC_DIR / pattern.format(seed=seed)
        if fname.exists():
            d = json.loads(fname.read_text())
            result[seed] = d["summary"][metric]
        else:
            result[seed] = None   # not yet run
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Classification
# ──────────────────────────────────────────────────────────────────────────────
def classify(delta_pp: float) -> str:
    a = abs(delta_pp)
    if a < 0.5:
        return "negligible"
    elif a < 2.0:
        return "minor"
    else:
        return "**significant**"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 5 Analysis")
    parser.add_argument("--best-acc", action="store_true",
                        help="Use best_acc instead of final_acc")
    args = parser.parse_args()
    metric = "best_acc" if args.best_acc else "final_acc"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Baseline ──────────────────────────────────────────────────────────────
    baseline_vals = load_baseline(metric)
    base_seeds = [v for v in baseline_vals.values() if v is not None]
    base_mean  = np.mean(base_seeds) * 100
    base_std   = np.std(base_seeds, ddof=1) * 100 if len(base_seeds) > 1 else 0.0

    print("v10.1 Baseline (γ=5, %s):" % metric)
    for s, v in baseline_vals.items():
        status = "%.4f" % v if v is not None else "MISSING"
        print("  seed=%d : %s" % (s, status))
    print("  mean=%.2f%%  std=%.2f%%  (n=%d seeds)\n" % (
        base_mean, base_std, len(base_seeds)))

    # ── Ablations ─────────────────────────────────────────────────────────────
    rows = []
    for key, desc, pattern in ABLATIONS:
        vals = load_ablation(pattern, metric)
        seed_accs  = [(s, v) for s, v in vals.items() if v is not None]
        n_complete  = len(seed_accs)
        accs_pct    = [v * 100 for _, v in seed_accs]
        mean_pct    = np.mean(accs_pct) if accs_pct else float("nan")
        std_pct     = np.std(accs_pct, ddof=1) if len(accs_pct) > 1 else float("nan")
        delta_pp    = mean_pct - base_mean
        cls         = classify(delta_pp) if not np.isnan(delta_pp) else "—"

        per_seed_str = "  ".join(
            "s%d=%.2f%%" % (s, v * 100) if v is not None else "s%d=MISSING" % s
            for s, v in vals.items()
        )

        rows.append({
            "key":       key,
            "desc":      desc,
            "n":         n_complete,
            "mean":      mean_pct,
            "std":       std_pct,
            "delta":     delta_pp,
            "cls":       cls,
            "per_seed":  per_seed_str,
        })

        # Console output
        print("Ablation: %s (%s)" % (key, desc))
        print("  Seeds run: %d/3" % n_complete)
        print("  Per-seed:  %s" % per_seed_str)
        if n_complete > 0:
            std_str = "±%.2f%%" % std_pct if not np.isnan(std_pct) else "±n/a"
            print("  Mean:      %.2f%% %s" % (mean_pct, std_str))
            print("  Δ vs base: %+.2f pp  → %s" % (delta_pp, cls))
        print()

    # ── Markdown output ───────────────────────────────────────────────────────
    md_lines = [
        "# Phase 5 Pipeline Ablation — Summary",
        "",
        "**Metric**: `%s`" % metric,
        "",
        "## v10.1 Baseline (PAID-FD, γ=5)",
        "",
        "| Seed | Accuracy |",
        "|------|----------|",
    ]
    for s, v in baseline_vals.items():
        md_lines.append("| %d | %.2f%% |" % (s, v * 100) if v else "| %d | MISSING |" % s)
    md_lines += [
        "",
        "**Mean = %.2f%% ± %.2f%%** (%d seeds)" % (base_mean, base_std, len(base_seeds)),
        "",
        "---",
        "",
        "## Ablation Results",
        "",
        "| Ablation | n seeds | Mean (%s) | Std | Δ (pp) | Classification |" % metric,
        "|----------|---------|-----------|-----|--------|----------------|",
    ]
    for r in rows:
        if r["n"] > 0:
            std_str = "%.2f%%" % r["std"] if not np.isnan(r["std"]) else "—"
            md_lines.append(
                "| %s | %d/3 | %.2f%% | %s | %+.2f | %s |" % (
                    r["desc"], r["n"], r["mean"], std_str, r["delta"], r["cls"]
                )
            )
        else:
            md_lines.append("| %s | 0/3 | — | — | — | NOT RUN |" % r["desc"])

    md_lines += [
        "",
        "---",
        "",
        "## Per-Seed Detail",
        "",
    ]
    for r in rows:
        md_lines.append("**%s**: %s" % (r["desc"], r["per_seed"]))

    md_lines += [
        "",
        "---",
        "",
        "## Classification Thresholds",
        "",
        "| Range | Label |",
        "|-------|-------|",
        "| \\|Δ\\| < 0.5 pp | negligible |",
        "| 0.5 ≤ \\|Δ\\| < 2 pp | minor |",
        "| \\|Δ\\| ≥ 2 pp | **significant** |",
        "",
    ]

    md_path = OUT_DIR / "phase5_summary.md"
    md_path.write_text("\n".join(md_lines))
    print("Wrote: %s" % md_path)

    # ── CSV output ────────────────────────────────────────────────────────────
    import csv
    csv_path = OUT_DIR / "phase5_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ablation", "n_seeds", "mean_pct", "std_pct",
            "delta_pp", "classification",
            "s42_pct", "s123_pct", "s456_pct",
        ])
        writer.writeheader()
        # baseline row
        writer.writerow({
            "ablation": "v10.1-baseline-g5",
            "n_seeds": len(base_seeds),
            "mean_pct": round(base_mean, 4),
            "std_pct": round(base_std, 4),
            "delta_pp": 0.0,
            "classification": "baseline",
            "s42_pct":  round(baseline_vals.get(42, float("nan")) * 100, 4),
            "s123_pct": round(baseline_vals.get(123, float("nan")) * 100, 4),
            "s456_pct": round(baseline_vals.get(456, float("nan")) * 100, 4),
        })
        # ablation rows
        for r in rows:
            abl_vals = load_ablation(
                [p for k, _, p in ABLATIONS if k == r["key"]][0], metric)
            def _fmt(v):
                return round(v * 100, 4) if v is not None else ""
            writer.writerow({
                "ablation": r["key"],
                "n_seeds": r["n"],
                "mean_pct": round(r["mean"], 4) if r["n"] > 0 else "",
                "std_pct":  round(r["std"], 4)  if r["n"] > 1 else "",
                "delta_pp": round(r["delta"], 4) if r["n"] > 0 else "",
                "classification": r["cls"],
                "s42_pct":  _fmt(abl_vals.get(42)),
                "s123_pct": _fmt(abl_vals.get(123)),
                "s456_pct": _fmt(abl_vals.get(456)),
            })
    print("Wrote: %s" % csv_path)


if __name__ == "__main__":
    main()
