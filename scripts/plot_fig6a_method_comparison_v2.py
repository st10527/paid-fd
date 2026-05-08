#!/usr/bin/env python3
"""
Fig 6a v2 — Method Comparison Bar Chart (grouped)
===================================================
Four visual groups:
  A: PAID-FD (ours, highlighted)
  B: Same-pipeline baselines — Fair Fixed-ε = 1, 3, 5
  C: Different-pipeline LDP baselines — Old Fixed-ε=3, CSRA
  D: No-privacy baselines — FedGMKD, FedAvg, FedMD

Inputs:
  results/experiments/tmc/  (expF_, expA_, expAp_ files)
  results/experiments/v10_1_combined_20260409_2304.json  (PAID-FD g5)
Outputs:
  results/figures/tmc_fig6a_method_comparison_v2.pdf
  results/figures/previews/tmc_fig6a_method_comparison_v2.png
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
import matplotlib.patches as mpatches
from _tmc_style import apply_tmc_style, TMC_COLORS, SINGLE_COL

apply_tmc_style()

# ── Paths ─────────────────────────────────────────────────────────────────────
TMC      = Path("results/experiments/tmc")
V101COMB = Path("results/experiments/v10_1_combined_20260409_2304.json")
OUT_PDF  = Path("results/figures/tmc_fig6a_method_comparison_v2.pdf")
OUT_PNG  = Path("results/figures/previews/tmc_fig6a_method_comparison_v2.png")
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

def load_tmc(label):
    p = TMC / f"{label}.json"
    return json.load(open(p)) if p.exists() else None

# ── Data ──────────────────────────────────────────────────────────────────────
comb = json.load(open(V101COMB))["summaries"]

# PAID-FD: g5, 3 seeds
paid_accs = [comb[k]["best_acc"] * 100 for k in ["g5_s42","g5_s123","g5_s456"]]

def tmc_acc(labels):
    vals = []
    for lb in labels:
        d = load_tmc(lb)
        if d:
            vals.append(d["summary"]["best_acc"] * 100)
    return vals

fair1_accs = tmc_acc(["expF_faireps1_s42"])
fair3_accs = tmc_acc(["expF_faireps3_s42"])
fair5_accs = tmc_acc(["expF_faireps5_s42"])
old3_accs  = tmc_acc(["expA_fixedeps3_s42","expA_fixedeps3_s123","expA_fixedeps3_s456"])
csra_accs  = tmc_acc(["expA_csra_s42","expA_csra_s123","expA_csra_s456"])
fedgmkd    = tmc_acc(["expAp_fedgmkd_s42"])
fedavg     = tmc_acc(["expAp_fedavg_s42"])
fedmd      = tmc_acc(["expAp_fedmd_s42"])

# ── Bar spec: (short_label, mean, std, color, group) ─────────────────────────
def ms(vals):
    arr = np.array(vals)
    return arr.mean(), (arr.std(ddof=1) if len(arr) > 1 else 0.0)

C = TMC_COLORS
BARS = [
    # Group A
    ("PAID-FD\n(ours)",   *ms(paid_accs),  C["paid_fd"],       "A"),
    # Group B
    ("Fair\nε=1",         *ms(fair1_accs), C["fair_eps_1"],    "B"),
    ("Fair\nε=3",         *ms(fair3_accs), C["fair_eps_3"],    "B"),
    ("Fair\nε=5",         *ms(fair5_accs), C["fair_eps_5"],    "B"),
    # Group C
    ("Old\nFixed-ε=3",   *ms(old3_accs),  C["old_fixed"],     "C"),
    ("CSRA",              *ms(csra_accs),  C["csra"],          "C"),
    # Group D
    ("FedGMKD",           *ms(fedgmkd),   C["fedgmkd"],       "D"),
    ("FedAvg",            *ms(fedavg),    C["fedavg"],         "D"),
    ("FedMD",             *ms(fedmd),     C["fedmd"],          "D"),
]

labels  = [b[0] for b in BARS]
means   = np.array([b[1] for b in BARS])
stds    = np.array([b[2] for b in BARS])
colors  = [b[3] for b in BARS]
groups  = [b[4] for b in BARS]

# Group brackets: (group_id, left_idx, right_idx, label_text, label_color)
BRACKETS = [
    ("A", 0,  0,  "PAID-FD",              C["paid_fd"]),
    ("B", 1,  3,  "same pipeline, fixed ε", C["fair_eps_3"]),
    ("C", 4,  5,  "different pipeline",   C["old_fixed"]),
    ("D", 6,  8,  "no privacy",           C["neutral"]),
]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.5, 2.8))

x = np.arange(len(BARS))
bars = ax.bar(x, means, 0.65, yerr=stds, color=colors,
              edgecolor="white", linewidth=0.5,
              capsize=2, error_kw={"linewidth": 0.7}, zorder=5)

# Numeric labels above each bar
for i, (m, e) in enumerate(zip(means, stds)):
    y_top = m + e + 0.8
    ax.text(i, y_top, f"{m:.1f}", ha="center", va="bottom",
            fontsize=5.5, color="#333333")

# ── Group brackets ─────────────────────────────────────────────────────────
BRACKET_Y = 73      # y-coordinate of bracket line
TICK_H    = 0.8     # downward tick height
LABEL_Y   = BRACKET_Y + 1.2

for gid, li, ri, text, color in BRACKETS:
    left  = li - 0.35
    right = ri + 0.35
    mid   = (left + right) / 2
    # Horizontal bar
    ax.plot([left, right], [BRACKET_Y, BRACKET_Y],
            color=color, lw=0.9, clip_on=False)
    # End ticks
    ax.plot([left,  left],  [BRACKET_Y, BRACKET_Y - TICK_H], color=color, lw=0.9, clip_on=False)
    ax.plot([right, right], [BRACKET_Y, BRACKET_Y - TICK_H], color=color, lw=0.9, clip_on=False)
    # Label
    ax.text(mid, LABEL_Y, text, ha="center", va="bottom",
            fontsize=5.2, color=color, clip_on=False)

ax.set_ylabel("Best Accuracy (%)")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=5.8, rotation=20, ha="right")
ax.set_ylim(0, 70)
ax.set_xlim(-0.6, len(BARS) - 0.4)

# Highlight PAID-FD bar with a subtle edge
bars[0].set_edgecolor(C["paid_fd"])
bars[0].set_linewidth(1.2)

fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=300)
print(f"Saved PDF : {OUT_PDF}")
print(f"Saved PNG : {OUT_PNG}")

CAPTION = r"""
\caption{Best accuracy of all methods on CIFAR-100 ($\gamma=5$, $\alpha=0.5$).
Methods are grouped by pipeline design:
(A)~PAID-FD (proposed);
(B)~same-pipeline ablations with fixed $\varepsilon\in\{1,3,5\}$ (``Fair Fixed-$\varepsilon$''),
showing that the Stackelberg game provides non-trivial benefit over a static assignment;
(C)~different-pipeline LDP baselines (Old Fixed-$\varepsilon$=3, CSRA);
(D)~non-private references (FedGMKD, FedAvg, FedMD).
Error bars: standard deviation across available seeds.}
"""
print("\n" + "─"*65)
print("LaTeX caption draft — Fig 6a v2:")
print("─"*65)
print(CAPTION)
