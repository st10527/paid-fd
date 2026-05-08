#!/usr/bin/env python3
"""
Fig 10 v2 — Privacy Composition (Basic only)
=============================================
Only plots Basic composition (full + selective).
Advanced composition omitted per Practical Privacy Regime positioning.

Inputs:  results/experiments/v10_1_3seeds_20260409_0922.json
Outputs:
  results/figures/tmc_fig10_privacy_composition_v2.pdf
  results/figures/previews/tmc_fig10_privacy_composition_v2.png
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"))

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _tmc_style import apply_tmc_style

apply_tmc_style()

# ── Paths ─────────────────────────────────────────────────────────────────────
V101_3S  = Path("results/experiments/v10_1_3seeds_20260409_0922.json")
OUT_PDF  = Path("results/figures/tmc_fig10_privacy_composition_v2.pdf")
OUT_PNG  = Path("results/figures/previews/tmc_fig10_privacy_composition_v2.png")
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
run   = json.load(open(V101_3S))["runs"]["g3_s42"]
avg_eps   = np.array(run["avg_eps"][:100])
part      = np.array(run["participation_rates"][:100])
rounds    = np.arange(1, 101)

mean_part = float(np.mean(part))           # ≈ 0.38 at γ=3
eps_star  = float(np.mean(avg_eps))        # ≈ 2.83

# Basic composition
cum_basic_full = np.cumsum(avg_eps)                    # full participation
cum_basic_sel  = np.cumsum(avg_eps * part)             # selective (round-by-round)

# Reduction at round 100
reduction_pct  = (1 - cum_basic_sel[-1] / cum_basic_full[-1]) * 100

# ε=10 reference line (common LDP threshold in literature)
EPS_REF = 10.0
ref_round = int(EPS_REF / eps_star)   # approx round where full-participation hits 10

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.5, 2.5))

C_FULL = "#d62728"    # deep red — full participation
C_SEL  = "#fb6a4a"    # lighter red — selective

ax.plot(rounds, cum_basic_full, "-",  color=C_FULL, lw=1.6,
        marker="o", markevery=(0, 15), markersize=3.5,
        markeredgecolor="white", markeredgewidth=0.4,
        label="Basic (full participation)", zorder=5)

ax.plot(rounds, cum_basic_sel,  "--", color=C_SEL,  lw=1.6,
        marker="s", markevery=(7, 15), markersize=3.5,
        markeredgecolor="white", markeredgewidth=0.4,
        label=f"Basic (selective, ≈{mean_part*100:.0f}%)", zorder=5)

# ε=10 reference line
ax.axhline(y=EPS_REF, color="#525252", ls=":", lw=0.9, zorder=3)
ax.text(70, EPS_REF + 1.5, r"$\varepsilon_\mathrm{ref}=10$ (LDP threshold)",
        fontsize=6, color="#525252", va="bottom",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
        zorder=8)

# Shaded region between full and selective
ax.fill_between(rounds, cum_basic_sel, cum_basic_full,
                alpha=0.12, color=C_FULL, zorder=2)

ax.set_xlabel("Communication Round")
ax.set_ylabel(r"Cumulative $\varepsilon_\mathrm{total}$")
ax.set_xlim(1, 100)
ax.set_ylim(0, 300)
ax.set_yticks([0, 50, 100, 150, 200, 250, 300])
ax.legend(loc="upper left", borderpad=0.3, fontsize=6.5, labelspacing=0.3)

fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=300)
print(f"Saved PDF : {OUT_PDF}")
print(f"Saved PNG : {OUT_PNG}")
print(f"\nKey numbers:")
print(f"  mean participation (γ=3): {mean_part*100:.1f}%")
print(f"  avg ε* per round: {eps_star:.3f}")
print(f"  cumulative ε at T=100 (full): {cum_basic_full[-1]:.1f}")
print(f"  cumulative ε at T=100 (selective): {cum_basic_sel[-1]:.1f}")
print(f"  reduction: {reduction_pct:.1f}%")

CAPTION = (
    r"\caption{Cumulative privacy budget under basic composition "
    r"($\gamma=3$, seed=42). "
    r"PAID-FD's Stackelberg game implicitly reduces privacy exposure "
    r"by allowing privacy-sensitive devices to opt out (selective "
    r"participation, $\approx$38\% on average at $\gamma=3$), "
    r"achieving $\approx$" + f"{reduction_pct:.0f}" + r"\% reduction "
    r"in cumulative $\varepsilon$ compared to mandatory full "
    r"participation over 100 rounds. "
    r"The dashed reference line marks $\varepsilon=10$, a commonly "
    r"used practical LDP threshold. "
    r"Advanced composition (Theorem~\ref{thm:advanced}) yields "
    r"tighter asymptotic bounds but is not visualized here as it does "
    r"not change the qualitative picture in our operational regime; "
    r"see Section~IV-C Remark on Practical Privacy Regime.}"
)
print("\n" + "─"*65)
print("LaTeX caption draft — Fig 10 v2:")
print("─"*65)
print(CAPTION)
