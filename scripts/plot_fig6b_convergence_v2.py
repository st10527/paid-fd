#!/usr/bin/env python3
"""
Fig 6b v2 — Convergence Curves CIFAR-100
==========================================
Uses _tmc_style color palette. New palette:
  PAID-FD      : blue solid
  Fair Fixed-ε=1: light-blue dashed
  Old Fixed-ε=3 : red dashed  (degrades annotation kept)
  FedGMKD       : purple solid
  FedAvg        : green solid

Inputs:  results/experiments/tmc/  +  v10_1_3seeds file
Outputs:
  results/figures/tmc_fig6b_convergence_cifar100_v2.pdf
  results/figures/previews/tmc_fig6b_convergence_cifar100_v2.png
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"))

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _tmc_style import apply_tmc_style, TMC_COLORS

apply_tmc_style()

# ── Paths ─────────────────────────────────────────────────────────────────────
TMC      = Path("results/experiments/tmc")
V101_3S  = Path("results/experiments/v10_1_3seeds_20260409_0922.json")
OUT_PDF  = Path("results/figures/tmc_fig6b_convergence_cifar100_v2.pdf")
OUT_PNG  = Path("results/figures/previews/tmc_fig6b_convergence_cifar100_v2.png")
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

def load_tmc(label):
    p = TMC / f"{label}.json"
    return json.load(open(p)) if p.exists() else None

# ── Data ──────────────────────────────────────────────────────────────────────
v3s    = json.load(open(V101_3S))
# Use g3_s42 (only complete run with full accuracy trace in 3seeds file)
paid_run = v3s["runs"]["g3_s42"]

C = TMC_COLORS
# (label, color, linestyle, linewidth, data, zorder)
CURVES = [
    ("PAID-FD",        C["paid_fd"],    "-",   1.8, paid_run,                        6),
    ("Fair Fixed-ε=1", C["fair_eps_1"], "--",  1.1, load_tmc("expF_faireps1_s42"),   4),
    ("Old Fixed-ε=3",  C["old_fixed"],  "-.",  1.1, load_tmc("expA_fixedeps3_s42"),  4),
    ("FedGMKD",        C["fedgmkd"],    ":",   1.2, load_tmc("expAp_fedgmkd_s42"),   3),
    ("FedAvg",         C["fedavg"],     "-",   0.9, load_tmc("expAp_fedavg_s42"),    3),
]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.5, 3.0))
rounds = np.arange(100)

for name, color, ls, lw, data, zord in CURVES:
    if data is None:
        continue
    accs = np.array(data["accuracies"][:100]) * 100
    ax.plot(rounds, accs, color=color, ls=ls, lw=lw, label=name, zorder=zord)

ax.set_xlabel("Communication Round")
ax.set_ylabel("Test Accuracy (%)")
ax.set_xlim(0, 100)
ax.set_ylim(0, 70)

# "degrades" annotation on Old Fixed-ε=3 line
old_d = load_tmc("expA_fixedeps3_s42")
if old_d:
    old_accs = np.array(old_d["accuracies"][:100]) * 100
    ax.annotate("degrades",
                xy=(90, old_accs[89]), xytext=(72, old_accs[89] - 8),
                fontsize=5.5, color=C["old_fixed"],
                arrowprops=dict(arrowstyle="->", color=C["old_fixed"], lw=0.7))

ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
          ncol=3, borderpad=0.3, labelspacing=0.2,
          handletextpad=0.4, fontsize=6.2, columnspacing=0.5)
fig.subplots_adjust(top=0.82)

fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=300)
print(f"Saved PDF : {OUT_PDF}")
print(f"Saved PNG : {OUT_PNG}")

CAPTION = r"""
\caption{Test accuracy convergence on CIFAR-100 over 100 communication rounds
($N=50$, $\gamma=3$, $\alpha=0.5$, seed=42).
PAID-FD converges to $\approx$61\% best accuracy while all LDP baselines 
either plateau lower (Old Fixed-$\varepsilon$=3) or lack privacy 
guarantees (FedGMKD, FedAvg).
Fair Fixed-$\varepsilon$=1 uses the same pipeline with a static 
$\varepsilon$ assignment, showing the game-theoretic selection provides 
benefit beyond the pipeline itself.}
"""
print("\n" + "─"*65)
print("LaTeX caption draft — Fig 6b v2:")
print("─"*65)
print(CAPTION)
