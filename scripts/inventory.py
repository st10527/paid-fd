#!/usr/bin/env python3
"""Quick inventory of all experiment data."""
import json, numpy as np, glob, os

os.chdir("/Users/nclab/Desktop/tku_research paper/2026_TMC/paid_fd")

print("=== EXPERIMENT DATA INVENTORY ===\n")

for fp in sorted(glob.glob("results/experiments/*.json")):
    name = os.path.basename(fp)
    size_kb = os.path.getsize(fp) / 1024
    with open(fp) as f:
        d = json.load(f)

    seeds = d.get("seeds", ["?"])
    runs = d.get("runs", {})

    print(f"--- {name} ({size_kb:.0f} KB) ---")
    print(f"  Seeds: {seeds}")

    for method, run_list in runs.items():
        n_runs = len(run_list)
        n_rounds = len(run_list[0]["accuracies"]) if run_list else 0
        finals = [r["accuracies"][-1] for r in run_list]
        mean_f = np.mean(finals)
        std_f = np.std(finals)
        print(f"  {method}: {n_runs} runs x {n_rounds} rounds, "
              f"final={mean_f:.4f} +/- {std_f:.4f}")
    print()

print("=== FIGURES INVENTORY ===\n")
pdfs = sorted(glob.glob("results/figures/*.pdf"))
print(f"Total: {len(pdfs)} PDF figures")
for p in pdfs:
    print(f"  {os.path.basename(p)}")

print("\n=== SOURCE CODE METHODS ===\n")
for fp in sorted(glob.glob("src/methods/*.py")):
    name = os.path.basename(fp)
    if name == "__init__.py":
        continue
    lines = open(fp).readlines()
    print(f"  {name}: {len(lines)} lines")

print("\n=== MODELS ===\n")
for fp in sorted(glob.glob("src/models/*.py")):
    name = os.path.basename(fp)
    if name == "__init__.py":
        continue
    lines = open(fp).readlines()
    print(f"  {name}: {len(lines)} lines")
