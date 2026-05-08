#!/usr/bin/env python3
"""
Fig 5 v2 — λ Sensitivity
=========================
Inputs:  results/experiments/v10_1_combined_20260409_2304.json
         results/experiments/v10_1_lambda_20260409_1921.json
Outputs:
  results/figures/tmc_fig5_lambda_sensitivity_v2.pdf
  results/figures/previews/tmc_fig5_lambda_sensitivity_v2.png
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
LAMBDA   = Path("results/experiments/v10_1_lambda_20260409_1921.json")
OUT_PDF  = Path("results/figures/tmc_fig5_lambda_sensitivity_v2.pdf")
OUT_PNG  = Path("results/figures/previews/tmc_fig5_lambda_sensitivity_v2.png")
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
comb = json.load(open(COMBINED))["summaries"]
lm_data = defaultdict(lambda: defaultdict(list))

# Load lm* keys from combined (has lm0.5, lm1.0, lm2.0 for all γ)
for k, v in comb.items():
    if k.startswith("lm"):
        parts = k.split("_")
        lm = float(parts[0][2:])
        g  = int(parts[1][1:])
        val = v.get("best_acc", v.get("final_acc", 0))
        lm_data[lm][g].append(val * 100)

# Load lambda sweep file (may have additional seeds not in combined)
if LAMBDA.exists():
    lam_raw = json.load(open(LAMBDA))
    for k, v in (lam_raw.get("summaries", lam_raw).items()):
        if k.startswith("lm"):
            parts = k.split("_")
            lm = float(parts[0][2:])
            g  = int(parts[1][1:])
            val = v.get("best_acc", v.get("final_acc", 0))
            lm_data[lm][g].append(val * 100)

# Fill lm=1.0 from baseline g*_s* keys in combined (multiple seeds)
for k, v in comb.items():
    if k.startswith("g") and "_s" in k:
        g = int(k.split("_")[0][1:])
        lm_data[1.0][g].append(v["best_acc"] * 100)

# gradient_blue palette: light → dark for lm 0.5 → 1.0 → 2.0
GB = TMC_COLORS["gradient_blue"]  # ['#c6dbef','#6baed6','#2171b5','#08306b']
styles = {
    0.5: (GB[1], "o",  r"$\lambda_\mathrm{mult}=0.5$"),
    1.0: (GB[2], "s",  r"$\lambda_\mathrm{mult}=1.0$ (default)"),
    2.0: (GB[3], "^",  r"$\lambda_\mathrm{mult}=2.0$"),
}

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=SINGLE_COL)

for lm in sorted(lm_data.keys()):
    if lm not in styles:
        continue
    color, mk, lab = styles[lm]
    gs    = sorted(lm_data[lm].keys())
    means = [np.mean(lm_data[lm][g]) for g in gs]
    stds  = [np.std(lm_data[lm][g])  for g in gs]
    ax.plot(gs, means, marker=mk, color=color, label=lab, lw=1.3,
            markeredgecolor="white", markeredgewidth=0.5, markersize=5, zorder=5)
    ax.errorbar(gs, means, yerr=stds, fmt="none", ecolor=color,
                elinewidth=0.8, capsize=2, zorder=4)

ax.set_xlabel(r"Server valuation $\gamma$")
ax.set_ylabel("Best Accuracy (%)")
ax.set_xticks([3, 5, 7, 10])
ax.set_ylim(60.0, 62.5)
ax.legend(loc="lower right", borderpad=0.4, fontsize=6.5)

fig.tight_layout(pad=0.4)
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=300)
print(f"Saved PDF : {OUT_PDF}")
print(f"Saved PNG : {OUT_PNG}")
