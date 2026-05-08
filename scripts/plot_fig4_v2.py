#!/usr/bin/env python3
"""
Fig 4 v2 — Efficiency Frontier (1×4 double-column)
====================================================
Input : results/analysis/efficiency_frontier_data.csv
Output: results/figures/tmc_fig4_efficiency_frontier_v2.pdf
        results/figures/previews/tmc_fig4_efficiency_frontier_v2.png
"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Shared TMC style ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"))
from _tmc_style import apply_tmc_style
apply_tmc_style()
# Extra per-figure overrides on top of shared style
plt.rcParams.update({
    "axes.linewidth":     0.7,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "lines.linewidth":    1.4,
    "lines.markersize":   5,
    "errorbar.capsize":   3,
    "ps.fonttype":        42,
})

# ── Paths ────────────────────────────────────────────────────────────────────
CSV_PATH   = Path("results/analysis/efficiency_frontier_data.csv")
OUT_PDF    = Path("results/figures/tmc_fig4_efficiency_frontier_v3.pdf")
OUT_PNG    = Path("results/figures/previews/tmc_fig4_efficiency_frontier_v3.png")
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
rows = []
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        rows.append({k: float(row[k]) for k in row})

by_gamma = defaultdict(list)
for r in rows:
    by_gamma[int(r["gamma"])].append(r)

GAMMAS = [3, 5, 7, 10]
X      = np.array(GAMMAS)

def stats(gamma_list, key):
    vals = np.array([r[key] for r in gamma_list])
    return vals.mean(), vals.std(ddof=1)

acc_m,  acc_s  = zip(*[stats(by_gamma[g], "best_acc")          for g in GAMMAS])
pr_m,   pr_s   = zip(*[stats(by_gamma[g], "participation_rate") for g in GAMMAS])
mce_m,  mce_s  = zip(*[stats(by_gamma[g], "max_cum_eps")        for g in GAMMAS])
pay_m,  pay_s  = zip(*[stats(by_gamma[g], "total_payment")      for g in GAMMAS])

acc_m  = np.array(acc_m);  acc_s  = np.array(acc_s)
pr_m   = np.array(pr_m);   pr_s   = np.array(pr_s)
mce_m  = np.array(mce_m);  mce_s  = np.array(mce_s)
pay_m  = np.array(pay_m);  pay_s  = np.array(pay_s)

# Payment in thousands
pay_m_k = pay_m / 1000
pay_s_k = pay_s / 1000

# ── Shared style helpers ──────────────────────────────────────────────────────
EB_KW   = dict(fmt="-", capsize=3, capthick=0.8, elinewidth=0.8, linewidth=1.4)
GRID_KW = dict(axis="y", color="#cccccc", linewidth=0.5, alpha=0.5)
PANEL_W, PANEL_H = 3.2, 2.4

C_ACC  = "#1565C0"   # blue
C_PR   = "#1565C0"   # blue
C_EPS  = "#C62828"   # red
C_PAY  = "#2E7D32"   # green

def _finish(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(GAMMAS)
    ax.set_xlabel(r"$\gamma$")
    ax.grid(**GRID_KW)

def _save(fig, stem):
    pdf = OUT_PDF.parent / f"{stem}.pdf"
    png = OUT_PNG.parent / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved PDF : {pdf}")
    print(f"Saved PNG : {png}")

# ─────────────────────────────────────────────────────────────────────────────
# Panel (a) — Best Accuracy
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H), constrained_layout=True)
ax.errorbar(X, acc_m, yerr=acc_s, color=C_ACC, marker="s",
            markerfacecolor=C_ACC, **EB_KW)
ax.set_ylabel("Best Accuracy (%)")
ax.set_ylim(60.0, 62.5)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
_finish(ax)
_save(fig, "tmc_fig4_panel_a_accuracy")

# ─────────────────────────────────────────────────────────────────────────────
# Panel (b) — Participation Rate
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H), constrained_layout=True)
ax.errorbar(X, pr_m, yerr=pr_s, color=C_PR, marker="o",
            markerfacecolor=C_PR, **EB_KW)
ax.set_ylabel("Participation Rate")
ax.set_ylim(0, 1.12)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
ax.text(0.95, 0.08, "2.63\u00d7", transform=ax.transAxes,
        fontsize=9, fontweight="bold", color=C_PR,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=C_PR, lw=0.6, alpha=0.85))
_finish(ax)
_save(fig, "tmc_fig4_panel_b_participation")

# ─────────────────────────────────────────────────────────────────────────────
# Panel (c) — Worst-case Privacy Exposure
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H), constrained_layout=True)
ax.errorbar(X, mce_m, yerr=mce_s, color=C_EPS, marker="^",
            markerfacecolor=C_EPS, **EB_KW)
ax.set_ylabel(r"Max Cumulative $\varepsilon$")
ax.set_ylim(400, 1000)
ax.text(0.95, 0.08, "1.75\u00d7", transform=ax.transAxes,
        fontsize=9, fontweight="bold", color=C_EPS,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=C_EPS, lw=0.6, alpha=0.85))
_finish(ax)
_save(fig, "tmc_fig4_panel_c_cumulative")

# ─────────────────────────────────────────────────────────────────────────────
# Panel (d) — Total Payment
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H), constrained_layout=True)
ax.errorbar(X, pay_m_k, yerr=pay_s_k, color=C_PAY, marker="D",
            markerfacecolor=C_PAY, **EB_KW)
ax.set_ylabel(r"Total Payment ($\times 10^3$)")
ax.set_ylim(0, 65)
ax.text(0.95, 0.08, "6.35\u00d7", transform=ax.transAxes,
        fontsize=9, fontweight="bold", color=C_PAY,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=C_PAY, lw=0.6, alpha=0.85))
_finish(ax)
_save(fig, "tmc_fig4_panel_d_payment")

# ─────────────────────────────────────────────────────────────────────────────
# Caption
# ─────────────────────────────────────────────────────────────────────────────
CAPTION = r"""
\caption{Efficiency frontier of PAID-FD across $\gamma \in \{3, 5, 7, 10\}$.
(a)~Best accuracy stays within 0.25\,pp across the $\gamma$ range, indicating
a pipeline-determined ceiling.
(b)--(d)~The three cost dimensions scale substantially with $\gamma$:
participation expands 2.63$\times$, worst-case privacy exposure 1.75$\times$,
and total payment 6.35$\times$.
The Stackelberg game thus selects a deployment operating point on the accuracy
plateau, trading coverage (participation) and expense (payment) against
privacy risk exposure.
Error bars: standard deviation across 3 seeds.}
"""

print("\n" + "─" * 65)
print("LaTeX caption draft:")
print("─" * 65)
print(CAPTION)
