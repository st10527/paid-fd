#!/usr/bin/env python3
"""
Fig 8 v2 — Scalability (N sweep)
=================================
Inputs:  results/experiments/tmc/expB_n*_g*_s*.json
         results/experiments/v10_1_combined_20260409_2304.json  (N=50 baseline)
Outputs:
  results/figures/tmc_fig8_scalability_v2.pdf
  results/figures/previews/tmc_fig8_scalability_v2.png
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

TMC      = Path("results/experiments/tmc")
COMBINED = Path("results/experiments/v10_1_combined_20260409_2304.json")
OUT_PDF  = Path("results/figures/tmc_fig8_scalability_v2.pdf")
OUT_PNG  = Path("results/figures/previews/tmc_fig8_scalability_v2.png")
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

C = TMC_COLORS

# ── Data ──────────────────────────────────────────────────────────────────────
comb = json.load(open(COMBINED))["summaries"]
data = defaultdict(list)

for f in sorted(TMC.glob("expB_n*_g*_s*.json")):
    d = json.load(open(f))
    n = d["config"]["n_devices"]
    g = int(d["config"]["gamma"])
    data[(n, g)].append(d["summary"]["best_acc"] * 100)

# N=50 from combined baseline
for k, v in comb.items():
    if k.startswith("g") and "_s" in k:
        g = int(k.split("_")[0][1:])
        if g in [3, 10]:
            data[(50, g)].append(v["best_acc"] * 100)

# Baselines (N=50 reference points)
def load_tmc(label):
    p = TMC / f"{label}.json"
    return json.load(open(p)) if p.exists() else None

baselines = [
    ("FedGMKD",       "expAp_fedgmkd_s42",  C["fedgmkd"],  "D"),
    ("FedAvg",        "expAp_fedavg_s42",   C["fedavg"],   "P"),
    ("Old Fixed-ε=3", "expA_fixedeps3_s42", C["old_fixed"], "X"),
]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=SINGLE_COL)

ns    = [20, 50, 80]
x     = np.arange(len(ns))
width = 0.34

gamma_styles = [
    (3,  C["paid_fd"],     r"PAID-FD $\gamma=3$"),
    (10, C["fair_eps_1"],  r"PAID-FD $\gamma=10$"),
]

for gi, (gamma, color, label) in enumerate(gamma_styles):
    means = [np.mean(data[(n, gamma)]) if data[(n, gamma)] else float("nan") for n in ns]
    stds  = [np.std(data[(n, gamma)])  if len(data[(n, gamma)]) > 1 else 0      for n in ns]
    off   = (gi - 0.5) * width
    ax.bar(x + off, means, width * 0.9, yerr=stds, color=color,
           edgecolor="white", linewidth=0.5, capsize=2,
           error_kw={"linewidth": 0.7},
           label=label, zorder=5)

# Baselines as scatter markers at N=50 position
# Center = mean of the two bar centers for N=50
baseline_x = x[1]  # exactly at N=50 tick center
for bname, btag, bcolor, bmark in baselines:
    d = load_tmc(btag)
    if d:
        val = d["summary"]["best_acc"] * 100
        ax.scatter([baseline_x], [val], marker=bmark, s=38, color=bcolor,
                   edgecolor="white", linewidth=0.5,
                   label=f"{bname} (N=50)", zorder=7)

ax.set_xlabel("Number of Devices ($N$)")
ax.set_ylabel("Best Accuracy (%)")
ax.set_xticks(x)
ax.set_xticklabels([f"$N$={n}" for n in ns])
ax.set_ylim(35, 65)

# 2-row legend: PAID-FD pair on row 1, 3 baselines on row 2
# Achieved by placing a blank spacer handle between the two groups
import matplotlib.patches as mpatches
blank = mpatches.Patch(visible=False, label="")
handles, labels = ax.get_legend_handles_labels()
# handles[0:2] = PAID-FD bars, handles[2:5] = baselines
row1 = handles[:2] + [blank]
row2 = handles[2:]
ax.legend(row1 + row2, [labels[i] for i in range(2)] + [""] + labels[2:],
          loc="lower center", bbox_to_anchor=(0.5, 1.01),
          ncol=3, borderpad=0.3, fontsize=5.8,
          labelspacing=0.2, columnspacing=0.5, handletextpad=0.4)
fig.subplots_adjust(top=0.78)

fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=300)
print(f"Saved PDF : {OUT_PDF}")
print(f"Saved PNG : {OUT_PNG}")
