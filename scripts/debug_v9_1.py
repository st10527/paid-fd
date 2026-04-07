#!/usr/bin/env python3
"""Deep-dive into v9.1 run details."""
import json

with open('results/experiments/v9_1_distill_fix.json') as f:
    data = json.load(f)

runs = data['runs']
name = 'D2_C5_T1_g5'
run = runs[name]

print(f"=== {name} details ===")
print(f"  method: {run.get('method')}")
print(f"  n_rounds: {run.get('n_rounds')}")
print(f"  elapsed_sec: {run.get('elapsed_sec')}")
print(f"  accuracies[:10]: {run['accuracies'][:10]}")
print(f"  losses[:10]: {run['losses'][:10]}")
print(f"  participation_rates[:5]: {run['participation_rates'][:5]}")
print(f"  prices[:5]: {run['prices'][:5]}")
print(f"  avg_eps[:5]: {run['avg_eps'][:5]}")
print(f"  avg_s[:5]: {run['avg_s'][:5]}")

extras = run.get('extras', {})
print(f"  extras keys: {list(extras.keys()) if extras else 'empty'}")
if extras:
    for k, v in extras.items():
        if isinstance(v, list):
            print(f"  extras[{k}][:3]: {v[:3]}")
        elif isinstance(v, dict):
            print(f"  extras[{k}]: dict with keys {list(v.keys())[:5]}")
        else:
            print(f"  extras[{k}]: {v}")

cfg = run.get('config', {})
print(f"  config: {cfg}")

# Also check D4 with CE anchor
print()
name2 = 'D4_C5_T1_CE_g5'
run2 = runs[name2]
print(f"=== {name2} details ===")
print(f"  accuracies[:10]: {run2['accuracies'][:10]}")
print(f"  losses[:10]: {run2['losses'][:10]}")
print(f"  elapsed_sec: {run2.get('elapsed_sec')}")
