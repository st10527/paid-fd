#!/usr/bin/env python3
"""Compute Route B efficiency metrics from existing Phase 1 data."""
import json
import numpy as np

with open('results/experiments/phase1_gamma_seed42.json') as f:
    d = json.load(f)

print('=== Route B Efficiency Metrics (Phase 1.1 gamma) ===')
print()

rows = []
for g in ['3', '5', '7', '10', '15']:
    run = d['runs'][g][0]
    acc = run['accuracies']
    n_dev = run['config']['n_devices']
    n_part = int(run['participation_rates'][0] * n_dev)
    avg_eps = run['avg_eps'][0]
    price = run['prices'][0]
    best_acc = max(acc)
    
    comm_per_round_kb = n_part * 100 * 4 / 1024
    total_comm_kb = comm_per_round_kb * 100
    total_payment = price * n_part * 100
    total_privacy = avg_eps * 100
    
    rounds_to_58 = next((i for i, a in enumerate(acc) if a >= 0.58), -1)
    rounds_to_60 = next((i for i, a in enumerate(acc) if a >= 0.60), -1)
    
    e0 = run['energy_history'][0]
    energy_per_round = e0['training'] + e0['inference'] + e0['communication']
    total_energy = energy_per_round * 100
    
    rows.append({
        'g': g, 'best': best_acc, 'n_part': n_part, 'eps': avg_eps,
        'price': price, 'comm_kb': total_comm_kb, 'payment': total_payment,
        'privacy': total_privacy, 'energy': total_energy,
        'r58': rounds_to_58, 'r60': rounds_to_60,
    })

# Table 1: Summary
fmt = "{:>4} | {:>6} | {:>5} | {:>5} | {:>9} | {:>10} | {:>10} | {:>6} | {:>6}"
print(fmt.format('g', 'Acc%', 'N_p', 'eps', 'Comm(KB)', 'Payment', 'PrivBudget', 'R@58%', 'R@60%'))
print('-' * 85)
for r in rows:
    r58 = f"R{r['r58']}" if r['r58'] >= 0 else "N/A"
    r60 = f"R{r['r60']}" if r['r60'] >= 0 else "N/A"
    print(fmt.format(
        r['g'], f"{r['best']*100:.1f}", r['n_part'], f"{r['eps']:.3f}",
        f"{r['comm_kb']:.1f}", f"{r['payment']:.0f}", f"{r['privacy']:.1f}",
        r58, r60
    ))

# Comparison
print()
print('=== gamma=3 vs gamma=10 Direct Comparison ===')
r3 = rows[0]  # gamma=3
r10 = rows[3]  # gamma=10
print(f"  Participants:    {r3['n_part']} vs {r10['n_part']}  ({(1 - r3['n_part']/r10['n_part'])*100:.0f}% fewer)")
print(f"  Per-device eps:  {r3['eps']:.3f} vs {r10['eps']:.3f}  ({r3['eps']/r10['eps']:.1f}x stronger privacy)")
print(f"  Communication:   {r3['comm_kb']:.0f}KB vs {r10['comm_kb']:.0f}KB  ({(1 - r3['comm_kb']/r10['comm_kb'])*100:.0f}% less)")
print(f"  Total payment:   {r3['payment']:.0f} vs {r10['payment']:.0f}  ({(1 - r3['payment']/r10['payment'])*100:.0f}% cheaper)")
print(f"  Accuracy:        {r3['best']*100:.2f}% vs {r10['best']*100:.2f}%  (only {abs(r3['best'] - r10['best'])*100:.2f}% diff)")
print(f"  R@58%:           R{r3['r58']} vs R{r10['r58']}")

# Lambda analysis
print()
print('=== Route B Efficiency Metrics (Phase 1.2 lambda) ===')
print()

with open('results/experiments/phase1_lambda_seed42.json') as f:
    dl = json.load(f)

rows_l = []
for l in ['0.3', '0.5', '1.0', '2.0', '3.0']:
    run = dl['runs'][l][0]
    acc = run['accuracies']
    n_dev = run['config']['n_devices']
    n_part = int(run['participation_rates'][0] * n_dev)
    avg_eps = run['avg_eps'][0]
    price = run['prices'][0]
    best_acc = max(acc)
    
    comm_per_round_kb = n_part * 100 * 4 / 1024
    total_comm_kb = comm_per_round_kb * 100
    total_payment = price * n_part * 100
    
    rounds_to_58 = next((i for i, a in enumerate(acc) if a >= 0.58), -1)
    rounds_to_60 = next((i for i, a in enumerate(acc) if a >= 0.60), -1)
    
    rows_l.append({
        'l': l, 'best': best_acc, 'n_part': n_part, 'eps': avg_eps,
        'price': price, 'comm_kb': total_comm_kb, 'payment': total_payment,
        'r58': rounds_to_58, 'r60': rounds_to_60,
    })

fmt2 = "{:>5} | {:>6} | {:>5} | {:>5} | {:>9} | {:>10} | {:>6} | {:>6}"
print(fmt2.format('lam', 'Acc%', 'N_p', 'eps', 'Comm(KB)', 'Payment', 'R@58%', 'R@60%'))
print('-' * 70)
for r in rows_l:
    r58 = f"R{r['r58']}" if r['r58'] >= 0 else "N/A"
    r60 = f"R{r['r60']}" if r['r60'] >= 0 else "N/A"
    print(fmt2.format(
        r['l'], f"{r['best']*100:.1f}", r['n_part'], f"{r['eps']:.3f}",
        f"{r['comm_kb']:.1f}", f"{r['payment']:.0f}", r58, r60
    ))
