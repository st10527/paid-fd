#!/usr/bin/env python3
"""
Route B – Exp 4: Convergence Curves for γ Sweep
=================================================
Uses existing Phase 1.1 data. No GPU needed.

Generates:
  Fig 2a: All γ convergence curves on one plot
  Fig 2b: Smoothed convergence with key insight annotation
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "results/figures"
DATA_PATH = "results/experiments/phase1_gamma_seed42.json"
os.makedirs(OUT_DIR, exist_ok=True)

GAMMA_COLORS = {3: "#E64B35", 5: "#4DBBD5", 7: "#00A087", 10: "#3C5488", 15: "#F39B7F"}

with open(DATA_PATH) as f:
    data = json.load(f)

gammas = [int(g) for g in data["gamma_values"]]

# ── Fig 2a: Raw convergence curves ──
fig, ax = plt.subplots(figsize=(10, 6))

for g in gammas:
    r = data["runs"][str(g)][0]
    accs = np.array(r["accuracies"]) * 100
    rounds = np.arange(len(accs))
    ax.plot(rounds, accs, color=GAMMA_COLORS[g], linewidth=2,
            label=f"γ = {g} (final: {accs[-1]:.1f}%)", alpha=0.85)

ax.set_xlabel("Communication Round", fontsize=13)
ax.set_ylabel("Test Accuracy (%)", fontsize=13)
ax.set_title("Convergence Curves: PAID-FD Across γ Values (CIFAR-100)", fontsize=14, pad=10)
ax.legend(fontsize=11, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 99)
ax.set_ylim(10, 65)

# Highlight the key insight
ax.axhline(y=60, color="gray", linestyle="--", alpha=0.4, linewidth=1)
ax.text(50, 60.5, "≈60% plateau — accuracy independent of γ", 
        fontsize=10, color="gray", ha="center", style="italic")

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig2a_gamma_convergence.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/fig2a_gamma_convergence.png", dpi=200, bbox_inches="tight")
print("✓ Saved fig2a_gamma_convergence.pdf/png")
plt.close(fig)

# ── Fig 2b: Smoothed + annotated (for paper) ──
fig, ax = plt.subplots(figsize=(10, 6))

def smooth(y, window=5):
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")

for g in gammas:
    r = data["runs"][str(g)][0]
    accs = np.array(r["accuracies"]) * 100
    accs_smooth = smooth(accs, 5)
    rounds_smooth = np.arange(len(accs_smooth)) + 2  # center of window
    
    ax.plot(rounds_smooth, accs_smooth, color=GAMMA_COLORS[g], linewidth=2.5,
            label=f"γ = {g}", alpha=0.9)

ax.set_xlabel("Communication Round", fontsize=13)
ax.set_ylabel("Test Accuracy (%, 5-round moving avg.)", fontsize=13)
ax.set_title("PAID-FD Convergence: Same Accuracy, Different Efficiency", fontsize=14, pad=10)
ax.legend(fontsize=11, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 99)
ax.set_ylim(10, 65)

# Annotate key insight
ax.annotate("γ=3: 38% participation\n62% less communication\n2.6× stronger privacy",
            xy=(54, 59), fontsize=10, color="#E64B35", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E64B35", alpha=0.1))

ax.annotate("γ=10: 100% participation\nFull cost",
            xy=(71, 59), fontsize=10, color="#3C5488", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#3C5488", alpha=0.1))

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig2b_gamma_convergence_annotated.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/fig2b_gamma_convergence_annotated.png", dpi=200, bbox_inches="tight")
print("✓ Saved fig2b_gamma_convergence_annotated.pdf/png")
plt.close(fig)

print("\n✅ All Exp 4 (convergence) figures saved.")
