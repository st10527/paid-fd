#!/usr/bin/env python3
"""
Fig 9 v2 — Non-IID Robustness (α sweep)
=========================================
Inputs:  results/experiments/tmc/expE_a*_g*_s*.json
         results/experiments/v10_1_combined_20260409_2304.json  (α=0.5 baseline)
Outputs:
  results/figures/tmc_fig9_noniid_v2.pdf
  results/figures/previews/tmc_fig9_noniid_v2.png
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
OUT_PDF  = Path("results/figures/tmc_fig9_noniid_v2.pdf")
OUT_PNG  = Path("results/figures/previews/tmc_fig9_noniid_v2.png")
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

C = TMC_COLORS

# ── Data ──────────────────────────────────────────────────────────────────────
comb = json.load(open(COMBINED))["summaries"]
data = defaultdict(list)

alpha_map = {"a01": 0.1, "a10": 1.0}   # expE_a01 → 0.1, expE_a10 → 1.0

for f in sorted(TMC.glob("expE_a*_g*_s*.json")):
    d = json.load(open(f))
    alpha = d["config"].get("alpha", 0.5)
    gamma = int(d["config"]["gamma"])
    data[(alpha, gamma)].append(d["summary"]["best_acc"] * 100)

# α=0.5 from combined baseline
for k, v in comb.items():
    if k.startswith("g") and "_s" in k:
        g = int(k.split("_")[0][1:])
        if g in [3, 10]:
            data[(0.5, g)].append(v["best_acc"] * 100)

def load_tmc(label):
    p = TMC / f"{label}.json"
    return json.load(open(p)) if p.exists() else None

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.5, 2.8))

alphas = [0.1, 0.5, 1.0]
x      = np.arange(len(alphas))
width  = 0.34

gamma_styles = [
    (3,  C["paid_fd"],    r"PAID-FD $\gamma=3$"),
    (10, C["fair_eps_1"], r"PAID-FD $\gamma=10$"),
]

for gi, (gamma, color, label) in enumerate(gamma_styles):
    means = [np.mean(data[(a, gamma)]) if data[(a, gamma)] else float("nan") for a in alphas]
    stds  = [np.std(data[(a, gamma)])  if len(data[(a, gamma)]) > 1 else 0      for a in alphas]
    offset = (gi - 0.5) * width
    ax.bar(x + offset, means, width * 0.9, yerr=stds, color=color,
           edgecolor="white", linewidth=0.5, capsize=2,
           error_kw={"linewidth": 0.7},
           label=label, zorder=5)

# Reference lines (dashed horizontal)
old_d = load_tmc("expA_fixedeps3_s42")
gm_d  = load_tmc("expAp_fedgmkd_s42")

if old_d:
    old_val = old_d["summary"]["best_acc"] * 100
    ax.axhline(y=old_val, color=C["old_fixed"], ls="--", lw=1.1, zorder=10)
    ax.text(2.57, old_val + 0.3, f"Old Fixed-ε=3 ({old_val:.1f}%)",
            fontsize=5.5, color=C["old_fixed"], ha="left", va="bottom")

if gm_d:
    gm_val = gm_d["summary"]["best_acc"] * 100
    label_y = gm_val - 1.5 if old_d and abs(gm_val - old_val) < 2.5 else gm_val + 0.3
    ax.axhline(y=gm_val, color=C["fedgmkd"], ls="--", lw=1.1, zorder=10)
    ax.text(2.57, label_y, f"FedGMKD ({gm_val:.1f}%)",
            fontsize=5.5, color=C["fedgmkd"], ha="left", va="bottom")

ax.set_xlabel(r"Dirichlet $\alpha$ (non-IID severity)")
ax.set_ylabel("Best Accuracy (%)")
ax.set_xticks(x)
ax.set_xticklabels([r"$\alpha$=0.1" + "\n(extreme)",
                    r"$\alpha$=0.5" + "\n(moderate)",
                    r"$\alpha$=1.0" + "\n(mild)"])
ax.set_ylim(35, 65)

handles, lbls = ax.get_legend_handles_labels()
bar_h = [h for h, l in zip(handles, lbls) if "PAID" in l]
bar_l = [l for l in lbls if "PAID" in l]
ax.legend(bar_h, bar_l, loc="lower center", bbox_to_anchor=(0.5, 1.01),
          ncol=2, borderpad=0.3, fontsize=6.2,
          labelspacing=0.2, columnspacing=0.6)
fig.subplots_adjust(top=0.88, right=0.80)

fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=300)
print(f"Saved PDF : {OUT_PDF}")
print(f"Saved PNG : {OUT_PNG}")
