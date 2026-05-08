#!/usr/bin/env python3
"""
Fig 3 v2 — Prop 2 Monotonicity (ε* vs γ)
==========================================
Inputs:  results/experiments/v10_1_combined_20260409_2304.json
Outputs:
  results/figures/tmc_fig3_monotonicity_v2.pdf
  results/figures/previews/tmc_fig3_monotonicity_v2.png
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"))

from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _tmc_style import apply_tmc_style, TMC_COLORS, SINGLE_COL, subplot_label

apply_tmc_style()

COMBINED = Path("results/experiments/v10_1_combined_20260409_2304.json")
OUT_PDF  = Path("results/figures/tmc_fig3_monotonicity_v2.pdf")
OUT_PNG  = Path("results/figures/previews/tmc_fig3_monotonicity_v2.png")
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
comb = json.load(open(COMBINED))["summaries"]
gamma_eps = defaultdict(list)
for k, v in comb.items():
    if k.startswith("g") and "_s" in k:
        g = int(k.split("_")[0][1:])
        gamma_eps[g].append(v.get("avg_eps_per_round", v.get("avg_eps", 0)))

gammas = sorted(gamma_eps.keys())
means  = [np.mean(gamma_eps[g]) for g in gammas]
stds   = [np.std(gamma_eps[g])  for g in gammas]

print(f"{'γ':>4}  {'mean ε*':>8}  {'std':>6}  n")
for g, m, s in zip(gammas, means, stds):
    print(f"{g:>4}  {m:>8.4f}  {s:>6.4f}  {len(gamma_eps[g])}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=SINGLE_COL)

ax.errorbar(gammas, means, yerr=stds,
            fmt="s-",
            color=TMC_COLORS["paid_fd"],
            capsize=3, capthick=0.9,
            markeredgecolor="white", markeredgewidth=0.5,
            markersize=5.5, lw=1.4, zorder=5)

ax.set_xlabel(r"Server valuation $\gamma$")
ax.set_ylabel(r"Equilibrium $\varepsilon^*$ (mean, per round)")
ax.set_xticks(gammas)
ax.set_ylim(2.5, 3.4)

fig.tight_layout(pad=0.4)
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=300)
print(f"\nSaved PDF : {OUT_PDF}")
print(f"Saved PNG : {OUT_PNG}")
