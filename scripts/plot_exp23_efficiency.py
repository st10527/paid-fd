#!/usr/bin/env python3
"""
Route B – Experiments 2 & 3: Efficiency & Privacy-Cost Tradeoff
===============================================================
Uses existing Phase 1.1 (γ sweep) and Phase 1.2 (λ sweep) data.
No GPU needed – pure plotting.

Generates:
  Fig 4a: Accuracy + Communication Cost vs γ  (dual-axis)
  Fig 4b: Pareto Front: Communication Cost vs Accuracy
  Fig 4c: Privacy–Cost Tradeoff (avg ε vs total payment)
  Fig 4d: Convergence Speed (rounds to reach 55% & 60%)
  Fig 5a: λ sensitivity — accuracy + participation vs λ
  Fig 5b: λ sensitivity — price + privacy vs λ
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load data ──
DATA_DIR = "results/experiments"
OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

with open(f"{DATA_DIR}/phase1_gamma_seed42.json") as f:
    gamma_data = json.load(f)

with open(f"{DATA_DIR}/phase1_lambda_seed42.json") as f:
    lambda_data = json.load(f)

# Color palette
GAMMA_COLORS = {3: "#E64B35", 5: "#4DBBD5", 7: "#00A087", 10: "#3C5488", 15: "#F39B7F"}
LAMBDA_COLORS = {0.3: "#E64B35", 0.5: "#4DBBD5", 1.0: "#00A087", 2.0: "#3C5488", 3.0: "#F39B7F"}

# ── Parse gamma sweep data ──
gammas = [int(g) for g in gamma_data["gamma_values"]]
gamma_metrics = {}

for g in gammas:
    r = gamma_data["runs"][str(g)][0]
    accs = np.array(r["accuracies"])
    energy = r["energy_history"]
    extras = r["extras"]
    
    # Communication cost: sum over 100 rounds (in Joules, proxy for KB)
    total_comm = sum(e["communication"] for e in energy)
    total_train = sum(e["training"] for e in energy)
    avg_price = np.mean(r["prices"])
    avg_eps = np.mean(r["avg_eps"])
    avg_s = np.mean(r["avg_s"])
    avg_part = np.mean(r["participation_rates"])
    total_payment = sum(e["price"] * e["total_quality"] for e in extras)
    
    # Rounds to reach accuracy thresholds
    r_55 = next((i for i, a in enumerate(accs) if a >= 0.55), 100)
    r_58 = next((i for i, a in enumerate(accs) if a >= 0.58), 100)
    r_60 = next((i for i, a in enumerate(accs) if a >= 0.60), 100)
    
    gamma_metrics[g] = {
        "final_acc": r["final_accuracy"],
        "best_acc": r["best_accuracy"],
        "accs": accs,
        "total_comm": total_comm,
        "total_train": total_train,
        "avg_price": avg_price,
        "avg_eps": avg_eps,
        "avg_s": avg_s,
        "avg_part": avg_part,
        "total_payment": total_payment,
        "r_55": r_55,
        "r_58": r_58,
        "r_60": r_60,
        "n_part": int(avg_part * 50),
    }

# ── Parse lambda sweep data ──
lambdas = [float(l) for l in lambda_data["lambda_values"]]
lambda_metrics = {}

for lam in lambdas:
    r = lambda_data["runs"][str(lam)][0]
    accs = np.array(r["accuracies"])
    extras = r["extras"]
    
    total_comm = sum(e["communication"] for e in r["energy_history"])
    total_payment = sum(e["price"] * e["total_quality"] for e in extras)
    
    lambda_metrics[lam] = {
        "final_acc": r["final_accuracy"],
        "best_acc": r["best_accuracy"],
        "accs": accs,
        "avg_price": np.mean(r["prices"]),
        "avg_eps": np.mean(r["avg_eps"]),
        "avg_part": np.mean(r["participation_rates"]),
        "total_comm": total_comm,
        "total_payment": total_payment,
    }

# ══════════════════════════════════════════════════════════════
#  Fig 4a: Accuracy + Communication Cost vs γ (dual-axis)
# ══════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(8, 5))

x = np.arange(len(gammas))
width = 0.35

# Left axis: accuracy bars
acc_vals = [gamma_metrics[g]["final_acc"] * 100 for g in gammas]
bars = ax1.bar(x - width/2, acc_vals, width, color=[GAMMA_COLORS[g] for g in gammas],
               edgecolor="black", linewidth=0.8, alpha=0.85, label="Final Accuracy (%)")
ax1.set_ylabel("Final Accuracy (%)", fontsize=13, color="#3C5488")
ax1.set_ylim(55, 65)
ax1.tick_params(axis="y", labelcolor="#3C5488")

# Annotate accuracy
for bar, val in zip(bars, acc_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
             f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Right axis: communication cost
ax2 = ax1.twinx()
comm_vals = [gamma_metrics[g]["total_comm"] / 1e6 for g in gammas]  # Convert to millions
ax2.bar(x + width/2, comm_vals, width, color="gray", alpha=0.5, edgecolor="black",
        linewidth=0.8, label="Total Comm. Cost (×10⁶)")
ax2.set_ylabel("Total Communication Cost (×10⁶ J)", fontsize=13, color="gray")
ax2.tick_params(axis="y", labelcolor="gray")

ax1.set_xlabel("γ (Server Valuation)", fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels([str(g) for g in gammas])
ax1.set_title("(a) Same Accuracy, Vastly Different Cost", fontsize=14, pad=10)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig4a_accuracy_vs_cost.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/fig4a_accuracy_vs_cost.png", dpi=200, bbox_inches="tight")
print(f"✓ Saved fig4a_accuracy_vs_cost.pdf/png")
plt.close(fig)

# ══════════════════════════════════════════════════════════════
#  Fig 4b: Pareto Front — Communication Efficiency
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))

for g in gammas:
    m = gamma_metrics[g]
    ax.scatter(m["total_comm"] / 1e6, m["final_acc"] * 100,
               color=GAMMA_COLORS[g], s=150, marker="o", edgecolors="black",
               linewidth=1.5, zorder=5)
    ax.annotate(f"γ={g}", (m["total_comm"]/1e6, m["final_acc"]*100),
                textcoords="offset points", xytext=(10, -5), fontsize=11)

# Connect with dashed line to show Pareto front
comm_sorted = sorted(gammas, key=lambda g: gamma_metrics[g]["total_comm"])
ax.plot([gamma_metrics[g]["total_comm"]/1e6 for g in comm_sorted],
        [gamma_metrics[g]["final_acc"]*100 for g in comm_sorted],
        "--", color="gray", alpha=0.5, linewidth=1.5)

ax.set_xlabel("Total Communication Cost (×10⁶ J)", fontsize=13)
ax.set_ylabel("Final Accuracy (%)", fontsize=13)
ax.set_title("(b) Communication Efficiency Pareto Front", fontsize=14, pad=10)
ax.grid(True, alpha=0.3)

# Add efficiency annotation
r3, r10 = gamma_metrics[3], gamma_metrics[10]
reduction = (1 - r3["total_comm"] / r10["total_comm"]) * 100
ax.annotate(f"{reduction:.0f}% less\ncommunication",
            xy=(r3["total_comm"]/1e6, r3["final_acc"]*100),
            xytext=(r10["total_comm"]/1e6 * 0.6, r3["final_acc"]*100 - 1.5),
            fontsize=10, color="#E64B35", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#E64B35", lw=1.5))

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig4b_pareto_front.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/fig4b_pareto_front.png", dpi=200, bbox_inches="tight")
print(f"✓ Saved fig4b_pareto_front.pdf/png")
plt.close(fig)

# ══════════════════════════════════════════════════════════════
#  Fig 4c: Privacy–Cost Tradeoff
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))

for g in gammas:
    m = gamma_metrics[g]
    ax.scatter(m["avg_eps"], m["total_payment"] / 1e3,
               color=GAMMA_COLORS[g], s=150, marker="o", edgecolors="black",
               linewidth=1.5, zorder=5)
    offset = (10, 5) if g != 15 else (10, -12)
    ax.annotate(f"γ={g}", (m["avg_eps"], m["total_payment"]/1e3),
                textcoords="offset points", xytext=offset, fontsize=11)

ax.set_xlabel("Average Privacy Budget ε (lower = stronger privacy)", fontsize=12)
ax.set_ylabel("Total Payment (×10³)", fontsize=13)
ax.set_title("(c) Privacy–Cost Tradeoff", fontsize=14, pad=10)
ax.grid(True, alpha=0.3)
ax.invert_xaxis()  # Lower ε (stronger privacy) on left

# Add annotation
ax.annotate("γ=3: strongest privacy\nlowest cost\nsame accuracy",
            xy=(r3["avg_eps"], r3["total_payment"]/1e3),
            xytext=(0.6, r3["total_payment"]/1e3 + 8),
            fontsize=10, color="#E64B35", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#E64B35", lw=1.5))

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig4c_privacy_cost_tradeoff.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/fig4c_privacy_cost_tradeoff.png", dpi=200, bbox_inches="tight")
print(f"✓ Saved fig4c_privacy_cost_tradeoff.pdf/png")
plt.close(fig)

# ══════════════════════════════════════════════════════════════
#  Fig 4d: Convergence Speed — Rounds to Accuracy Thresholds
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(gammas))
width = 0.25

for i, (threshold, label, alpha) in enumerate([
    ("r_55", "55%", 0.5), ("r_58", "58%", 0.7), ("r_60", "60%", 0.9)
]):
    vals = [gamma_metrics[g][threshold] for g in gammas]
    bars = ax.bar(x + (i-1)*width, vals, width, alpha=alpha,
                  color=[GAMMA_COLORS[g] for g in gammas], edgecolor="black", linewidth=0.6)
    # Label
    for bar, val in zip(bars, vals):
        display = f"R{val}" if val < 100 else ">100"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                display, ha="center", va="bottom", fontsize=7, rotation=0)

ax.set_ylabel("Rounds to Reach Accuracy", fontsize=13)
ax.set_xlabel("γ (Server Valuation)", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels([str(g) for g in gammas])
ax.set_title("(d) Convergence Speed: Fewer Participants, Faster Convergence", fontsize=13, pad=10)
ax.legend(["55%", "58%", "60%"], title="Threshold", fontsize=10, title_fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig4d_convergence_speed.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/fig4d_convergence_speed.png", dpi=200, bbox_inches="tight")
print(f"✓ Saved fig4d_convergence_speed.pdf/png")
plt.close(fig)

# ══════════════════════════════════════════════════════════════
#  Fig 5a: λ Sensitivity — Accuracy + Participation vs λ
# ══════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(8, 5))

lam_sorted = sorted(lambdas)
acc_vals = [lambda_metrics[l]["final_acc"] * 100 for l in lam_sorted]
part_vals = [lambda_metrics[l]["avg_part"] * 100 for l in lam_sorted]

ax1.plot(lam_sorted, acc_vals, "o-", color="#3C5488", linewidth=2.5, markersize=10,
         markeredgecolor="black", markeredgewidth=1, label="Accuracy (%)")
ax1.set_ylabel("Final Accuracy (%)", fontsize=13, color="#3C5488")
ax1.set_ylim(55, 65)
ax1.tick_params(axis="y", labelcolor="#3C5488")

ax2 = ax1.twinx()
ax2.plot(lam_sorted, part_vals, "s--", color="#E64B35", linewidth=2, markersize=9,
         markeredgecolor="black", markeredgewidth=1, label="Participation (%)")
ax2.set_ylabel("Participation Rate (%)", fontsize=13, color="#E64B35")
ax2.tick_params(axis="y", labelcolor="#E64B35")

ax1.set_xlabel("λ (Privacy Sensitivity Multiplier)", fontsize=13)
ax1.set_title("(a) λ Sensitivity: Accuracy & Participation", fontsize=14, pad=10)
ax1.grid(True, alpha=0.3)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=10)

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig5a_lambda_sensitivity.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/fig5a_lambda_sensitivity.png", dpi=200, bbox_inches="tight")
print(f"✓ Saved fig5a_lambda_sensitivity.pdf/png")
plt.close(fig)

# ══════════════════════════════════════════════════════════════
#  Fig 5b: λ Sensitivity — Price + Privacy vs λ
# ══════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(8, 5))

price_vals = [lambda_metrics[l]["avg_price"] for l in lam_sorted]
eps_vals = [lambda_metrics[l]["avg_eps"] for l in lam_sorted]

ax1.plot(lam_sorted, price_vals, "o-", color="#00A087", linewidth=2.5, markersize=10,
         markeredgecolor="black", markeredgewidth=1, label="Avg Price $p$")
ax1.set_ylabel("Average Price $p$", fontsize=13, color="#00A087")
ax1.tick_params(axis="y", labelcolor="#00A087")

ax2 = ax1.twinx()
ax2.plot(lam_sorted, eps_vals, "D--", color="#F39B7F", linewidth=2, markersize=9,
         markeredgecolor="black", markeredgewidth=1, label="Avg ε")
ax2.set_ylabel("Average Privacy Budget ε", fontsize=13, color="#F39B7F")
ax2.tick_params(axis="y", labelcolor="#F39B7F")

ax1.set_xlabel("λ (Privacy Sensitivity Multiplier)", fontsize=13)
ax1.set_title("(b) λ Sensitivity: Price & Privacy Equilibrium", fontsize=14, pad=10)
ax1.grid(True, alpha=0.3)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig5b_lambda_price_privacy.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/fig5b_lambda_price_privacy.png", dpi=200, bbox_inches="tight")
print(f"✓ Saved fig5b_lambda_price_privacy.pdf/png")
plt.close(fig)

# ══════════════════════════════════════════════════════════════
#  Summary Table
# ══════════════════════════════════════════════════════════════
print("\n" + "="*90)
print("Table: γ Sweep Summary (CIFAR-100, 50 devices, 100 rounds)")
print("="*90)
print(f"{'γ':>4} {'Acc(%)':>8} {'Part(%)':>8} {'Avg ε':>8} {'Comm(M)':>10} {'Payment(K)':>11} {'R@55%':>6} {'R@60%':>6}")
print("-"*90)
for g in gammas:
    m = gamma_metrics[g]
    r60_str = f"R{m['r_60']}" if m['r_60'] < 100 else ">100"
    print(f"{g:4d} {m['final_acc']*100:8.1f} {m['avg_part']*100:8.0f} {m['avg_eps']:8.4f} "
          f"{m['total_comm']/1e6:10.2f} {m['total_payment']/1e3:11.1f} "
          f"{'R'+str(m['r_55']):>6s} {r60_str:>6s}")
print("="*90)

print(f"\n✅ All Exp 2/3 figures saved to {OUT_DIR}/")
