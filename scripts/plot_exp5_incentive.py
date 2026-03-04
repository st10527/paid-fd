#!/usr/bin/env python3
"""
Route B – Experiment 5: Incentive Mechanism Visualization
=========================================================
Pure computation – no GPU needed.

Generates:
  Fig 7a: Server utility U_ES(p) vs price p  (inverted-U curves, one per γ)
  Fig 7b: Participation rate vs price p       (sigmoid-like curves)
  Fig 7c: Equilibrium summary bar chart       (p*, participation, avg ε, avg s)

Uses the real Stackelberg solver + 50 heterogeneous devices (seed 42).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.game.stackelberg import ServerPricing, StackelbergSolver, DeviceBestResponse
from src.devices.heterogeneity import HeterogeneityGenerator

# ── Configuration ──
GAMMAS = [3, 5, 7, 10, 15]
N_DEVICES = 50
SEED = 42
N_PRICE_POINTS = 200
OUT_DIR = "results/figures"

# Color palette (colorblind-safe)
COLORS = {3: "#E64B35", 5: "#4DBBD5", 7: "#00A087", 10: "#3C5488", 15: "#F39B7F"}
MARKERS = {3: "o", 5: "s", 7: "^", 10: "D", 15: "v"}

os.makedirs(OUT_DIR, exist_ok=True)

# ── Generate devices (same seed as experiments) ──
gen = HeterogeneityGenerator(n_devices=N_DEVICES, seed=SEED)
devices = gen.generate()
print(f"Generated {len(devices)} devices (seed={SEED})")
print(f"  c_total range: [{min(d.c_total for d in devices):.3f}, {max(d.c_total for d in devices):.3f}]")
print(f"  lambda_i range: [{min(d.lambda_i for d in devices):.3f}, {max(d.lambda_i for d in devices):.3f}]")

# ── Compute curves for each γ ──
results = {}  # gamma -> { prices, utilities, participation_rates, qualities, p_star, ... }

for gamma in GAMMAS:
    pricing = ServerPricing(gamma=gamma, delta=0.01)
    curve = pricing.analyze_price_curve(devices, n_points=N_PRICE_POINTS)
    
    # Also solve for equilibrium
    solver = StackelbergSolver(gamma=gamma, delta=0.01)
    eq = solver.solve(devices)
    
    results[gamma] = {
        "prices": np.array(curve["prices"]),
        "utilities": np.array(curve["utilities"]),
        "participation_rates": np.array(curve["participation_rates"]),
        "qualities": np.array(curve["qualities"]),
        "p_star": eq["price"],
        "U_star": eq["server_utility"],
        "Q_star": eq["total_quality"],
        "part_rate": eq["participation_rate"],
        "n_part": eq["n_participants"],
        "avg_eps": eq["avg_eps"],
        "avg_s": eq["avg_s"],
        "total_payment": eq["total_payment"],
    }
    print(f"γ={gamma:2d}: p*={eq['price']:.3f}, part={eq['participation_rate']:.0%}, "
          f"avg_ε={eq['avg_eps']:.3f}, avg_s={eq['avg_s']:.1f}, U_ES={eq['server_utility']:.1f}")

# ══════════════════════════════════════════════════════════════
#  Figure 7a: Server Utility vs Price (Inverted-U Curves)
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

for gamma in GAMMAS:
    r = results[gamma]
    ax.plot(r["prices"], r["utilities"], color=COLORS[gamma], linewidth=2,
            label=f"γ = {gamma}")
    # Mark equilibrium
    ax.plot(r["p_star"], r["U_star"], marker=MARKERS[gamma], color=COLORS[gamma],
            markersize=10, markeredgecolor="black", markeredgewidth=1.2, zorder=5)

ax.set_xlabel("Price $p$", fontsize=13)
ax.set_ylabel("Server Utility $U_{ES}(p)$", fontsize=13)
ax.set_title("(a) Server Utility vs. Price — Inverted-U Curves", fontsize=14, pad=10)
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig7a_server_utility_vs_price.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/fig7a_server_utility_vs_price.png", dpi=200, bbox_inches="tight")
print(f"\n✓ Saved fig7a_server_utility_vs_price.pdf/png")
plt.close(fig)

# ══════════════════════════════════════════════════════════════
#  Figure 7b: Participation Rate vs Price
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

for gamma in GAMMAS:
    r = results[gamma]
    ax.plot(r["prices"], r["participation_rates"], color=COLORS[gamma], linewidth=2,
            label=f"γ = {gamma}")
    # Mark equilibrium
    ax.plot(r["p_star"], r["part_rate"], marker=MARKERS[gamma], color=COLORS[gamma],
            markersize=10, markeredgecolor="black", markeredgewidth=1.2, zorder=5)

ax.set_xlabel("Price $p$", fontsize=13)
ax.set_ylabel("Participation Rate", fontsize=13)
ax.set_title("(b) Participation Rate vs. Price", fontsize=14, pad=10)
ax.legend(fontsize=11, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)
ax.set_ylim(-0.02, 1.05)
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig7b_participation_vs_price.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/fig7b_participation_vs_price.png", dpi=200, bbox_inches="tight")
print(f"✓ Saved fig7b_participation_vs_price.pdf/png")
plt.close(fig)

# ══════════════════════════════════════════════════════════════
#  Figure 7c: Equilibrium Comparison (grouped bar chart)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
gamma_labels = [str(g) for g in GAMMAS]
x = np.arange(len(GAMMAS))
width = 0.6

# 7c-1: Optimal Price p*
vals = [results[g]["p_star"] for g in GAMMAS]
bars = axes[0].bar(x, vals, width, color=[COLORS[g] for g in GAMMAS], edgecolor="black", linewidth=0.8)
axes[0].set_ylabel("Price $p^*$", fontsize=12)
axes[0].set_title("Optimal Price", fontsize=12)
axes[0].set_xticks(x)
axes[0].set_xticklabels(gamma_labels)
axes[0].set_xlabel("γ", fontsize=12)
for bar, val in zip(bars, vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)

# 7c-2: Participation Rate
vals = [results[g]["part_rate"] for g in GAMMAS]
bars = axes[1].bar(x, vals, width, color=[COLORS[g] for g in GAMMAS], edgecolor="black", linewidth=0.8)
axes[1].set_ylabel("Participation Rate", fontsize=12)
axes[1].set_title("Participation", fontsize=12)
axes[1].set_xticks(x)
axes[1].set_xticklabels(gamma_labels)
axes[1].set_xlabel("γ", fontsize=12)
axes[1].set_ylim(0, 1.15)
for bar, val in zip(bars, vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.0%}", ha="center", va="bottom", fontsize=9)

# 7c-3: Average ε (privacy budget)
vals = [results[g]["avg_eps"] for g in GAMMAS]
bars = axes[2].bar(x, vals, width, color=[COLORS[g] for g in GAMMAS], edgecolor="black", linewidth=0.8)
axes[2].set_ylabel("Avg. Privacy Budget ε", fontsize=12)
axes[2].set_title("Privacy Budget", fontsize=12)
axes[2].set_xticks(x)
axes[2].set_xticklabels(gamma_labels)
axes[2].set_xlabel("γ", fontsize=12)
for bar, val in zip(bars, vals):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

# 7c-4: Total Payment
vals = [results[g]["total_payment"] for g in GAMMAS]
bars = axes[3].bar(x, vals, width, color=[COLORS[g] for g in GAMMAS], edgecolor="black", linewidth=0.8)
axes[3].set_ylabel("Total Payment", fontsize=12)
axes[3].set_title("Server Payment", fontsize=12)
axes[3].set_xticks(x)
axes[3].set_xticklabels(gamma_labels)
axes[3].set_xlabel("γ", fontsize=12)
for bar, val in zip(bars, vals):
    axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=9)

fig.suptitle("(c) Stackelberg Equilibrium Comparison Across γ", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig7c_equilibrium_comparison.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/fig7c_equilibrium_comparison.png", dpi=200, bbox_inches="tight")
print(f"✓ Saved fig7c_equilibrium_comparison.pdf/png")
plt.close(fig)

# ══════════════════════════════════════════════════════════════
#  Figure 7d: Per-Device Decision Distribution (scatter)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for gamma in [3, 7, 15]:
    solver = StackelbergSolver(gamma=gamma, delta=0.01)
    eq = solver.solve(devices)
    
    parts = [d for d in eq["decisions"] if d.participates]
    non_parts = [d for d in eq["decisions"] if not d.participates]
    
    ax_idx = {3: 0, 7: 1, 15: 2}[gamma]
    ax = axes[ax_idx]
    
    if parts:
        eps_vals = [d.eps_star for d in parts]
        s_vals = [d.s_star for d in parts]
        q_vals = [d.quality for d in parts]
        scatter = ax.scatter(eps_vals, s_vals, c=q_vals, cmap="YlOrRd", s=60,
                            edgecolors="black", linewidth=0.5, zorder=3)
        plt.colorbar(scatter, ax=ax, label="Quality $q_i$", shrink=0.8)
    
    if non_parts:
        ax.scatter([0]*len(non_parts), [0]*len(non_parts), c="gray", s=30,
                  marker="x", alpha=0.5, label="Non-participant")
    
    ax.set_xlabel("Privacy Budget $\\varepsilon_i^*$", fontsize=11)
    ax.set_ylabel("Upload Volume $s_i^*$", fontsize=11)
    ax.set_title(f"γ = {gamma} (n={len(parts)} participants)", fontsize=12)
    ax.grid(True, alpha=0.3)

fig.suptitle("(d) Per-Device Optimal Decisions at Equilibrium", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig7d_device_decisions.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/fig7d_device_decisions.png", dpi=200, bbox_inches="tight")
print(f"✓ Saved fig7d_device_decisions.pdf/png")
plt.close(fig)

# ══════════════════════════════════════════════════════════════
#  Print Equilibrium Table (for paper)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*85)
print("Table: Stackelberg Equilibrium Summary (50 devices, seed=42)")
print("="*85)
print(f"{'γ':>4} {'p*':>8} {'Part.':>8} {'n_part':>7} {'avg_ε':>8} {'avg_s':>8} {'U_ES':>10} {'Payment':>10}")
print("-"*85)
for gamma in GAMMAS:
    r = results[gamma]
    print(f"{gamma:4d} {r['p_star']:8.3f} {r['part_rate']:8.0%} {r['n_part']:7d} "
          f"{r['avg_eps']:8.4f} {r['avg_s']:8.1f} {r['U_star']:10.1f} {r['total_payment']:10.1f}")
print("="*85)

# Efficiency comparison
r3, r10 = results[3], results[10]
print(f"\n── γ=3 vs γ=10 Efficiency ──")
print(f"  Participation: {r3['part_rate']:.0%} vs {r10['part_rate']:.0%} ({r3['n_part']} vs {r10['n_part']} devices)")
print(f"  Avg ε:         {r3['avg_eps']:.4f} vs {r10['avg_eps']:.4f} ({r3['avg_eps']/r10['avg_eps']:.1f}× stronger privacy for γ=10)")
print(f"  Payment:       {r3['total_payment']:.0f} vs {r10['total_payment']:.0f} ({r10['total_payment']/r3['total_payment']:.1f}× more expensive)")
print(f"  Communication: {r3['n_part']} × {r3['avg_s']:.0f} vs {r10['n_part']} × {r10['avg_s']:.0f} uploads")

print(f"\n✅ All Exp 5 figures saved to {OUT_DIR}/")
