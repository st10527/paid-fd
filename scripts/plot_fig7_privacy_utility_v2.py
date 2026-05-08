#!/usr/bin/env python3
"""
Fig 7 v2 — Privacy-Utility Curve
==================================
Inputs:  results/experiments/tmc/expG_eps*_s42.json
         results/experiments/v10_1_combined_20260409_2304.json  (operating point)
Outputs:
  results/figures/tmc_fig7_privacy_utility_v2.pdf
  results/figures/previews/tmc_fig7_privacy_utility_v2.png
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"))

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from _tmc_style import apply_tmc_style, TMC_COLORS, SINGLE_COL

apply_tmc_style()

TMC      = Path("results/experiments/tmc")
COMBINED = Path("results/experiments/v10_1_combined_20260409_2304.json")
OUT_PDF  = Path("results/figures/tmc_fig7_privacy_utility_v2.pdf")
OUT_PNG  = Path("results/figures/previews/tmc_fig7_privacy_utility_v2.png")
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

C = TMC_COLORS

# ── Data ──────────────────────────────────────────────────────────────────────
eps_data = []
for f in sorted(TMC.glob("expG_*.json")):
    d = json.load(open(f))
    eps_str = d["label"].split("_")[1].replace("eps", "").replace("p", ".")
    eps = float(eps_str)
    eps_data.append((eps, d["summary"]["best_acc"] * 100))
eps_data.sort()

epsilons = [e[0] for e in eps_data]
bests    = [e[1] for e in eps_data]

# PAID-FD operating point
comb = json.load(open(COMBINED))["summaries"]
paid_eps  = np.mean([comb[f"g5_s{s}"]["avg_eps_per_round"] for s in [42, 123, 456]])
paid_acc  = np.mean([comb[f"g5_s{s}"]["best_acc"] * 100 for s in [42, 123, 456]])

print(f"expG data points: {list(zip(epsilons, [f'{v:.2f}' for v in bests]))}")
print(f"PAID-FD operating point: ε̄*={paid_eps:.4f}, acc={paid_acc:.2f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=SINGLE_COL)

# Main curve
ax.plot(epsilons, bests, "o-",
        color=C["fair_eps_3"],
        markeredgecolor="white", markeredgewidth=0.5,
        markersize=5, lw=1.4,
        label=r"Fixed $\varepsilon$ (same pipeline)", zorder=5)

# Shaded plateau region (ε ≥ 0.5)
plateau_lo, plateau_hi = 60.5, 62.3
ax.axhspan(plateau_lo, plateau_hi, alpha=0.07, color=C["paid_fd"], zorder=1)
ax.text(1.5, plateau_hi + 0.25, "noise-resilient plateau",
        fontsize=5.8, color=C["paid_fd"], fontstyle="italic",
        ha="center", va="bottom", zorder=8,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

# PAID-FD operating point — red star
ax.plot(paid_eps, paid_acc, "*",
        markersize=10, color="#d62728",
        markeredgecolor="white", markeredgewidth=0.4,
        label=rf"PAID-FD ($\bar{{\varepsilon}}^*={paid_eps:.2f}$, mean over devices)",
        zorder=10)

# ε=0.1 breaking point label (simple text, no large arrow)
idx_01 = epsilons.index(0.1)
ax.annotate(f"$\\varepsilon$=0.1: {bests[idx_01]:.1f}%",
            xy=(0.1, bests[idx_01]),
            xytext=(0.22, bests[idx_01] - 2.2),
            fontsize=5.8, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.7))

ax.set_xlabel(r"Privacy budget $\varepsilon$ per round")
ax.set_ylabel("Best Accuracy (%)")
ax.set_xscale("log")
ax.set_xticks([0.1, 0.5, 1, 2, 5, 10])
ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.set_xlim(0.07, 15)
ax.set_ylim(50, 65)

ax.legend(loc="lower right", borderpad=0.4, fontsize=6.2,
          labelspacing=0.3, handletextpad=0.4)

fig.tight_layout(pad=0.4)
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=300)
print(f"\nSaved PDF : {OUT_PDF}")
print(f"Saved PNG : {OUT_PNG}")

print("\n" + "="*65)
print("LaTeX caption draft — Fig 7 v2:")
print("="*65)
print(rf"""\caption{{Privacy-utility trade-off on CIFAR-100 ($N=50$,
$\gamma=5$, seed=42). Each point is the \emph{{same}} PAID-FD pipeline
with a fixed per-round budget $\varepsilon$ replacing the Stackelberg
equilibrium. The shaded band marks the noise-resilient plateau
($\varepsilon\geq0.5$) where accuracy is largely invariant to the
privacy level. The red star marks the PAID-FD Stackelberg operating
point ($\bar{{\varepsilon}}^*={paid_eps:.2f}$, mean over participating
devices across three seeds), confirming that the game-theoretic
assignment selects a budget well within the plateau.}}""")
