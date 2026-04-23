"""
Extract efficiency frontier data for all gamma x seed combinations.
Writes results/analysis/efficiency_frontier_data.csv and prints summary table.
"""
import json
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

COMB_PATH = Path('results/experiments/v10_1_combined_20260409_2304.json')
OUT_DIR = Path('results/analysis')
OUT_DIR.mkdir(parents=True, exist_ok=True)

comb = json.load(open(COMB_PATH))['summaries']

# Collect rows
rows = []
for key, s in comb.items():
    if not (key.startswith('g') and '_s' in key and not key.startswith('lm')):
        continue
    parts = key.split('_')
    gamma = int(parts[0][1:])
    seed = int(parts[1][1:])
    rows.append({
        'key': key,
        'gamma': gamma,
        'seed': seed,
        'participation_rate': s['avg_participation'],
        'max_cum_eps': s['max_privacy_spent'],
        'mean_cum_eps': s['avg_privacy_spent'],
        'total_payment': s['cumulative_payment'],
        'best_acc': s['best_acc'] * 100,
    })

rows.sort(key=lambda r: (r['gamma'], r['seed']))

# Write CSV
csv_path = OUT_DIR / 'efficiency_frontier_data.csv'
fieldnames = ['gamma', 'seed', 'participation_rate', 'max_cum_eps',
              'mean_cum_eps', 'total_payment', 'best_acc']
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)
print(f'CSV written: {csv_path}')

# Summary table
gammas = sorted(set(r['gamma'] for r in rows))
print()
print('| γ  | Seeds | Participation     | Max Cum ε         | Mean Cum ε        | Total Payment      | Accuracy (%)    |')
print('|----|-------|-------------------|-------------------|-------------------|--------------------|-----------------|')
for g in gammas:
    gr = [r for r in rows if r['gamma'] == g]
    n = len(gr)
    def ms(field):
        vals = [r[field] for r in gr]
        return '%.3f ± %.3f' % (np.mean(vals), np.std(vals))
    def ms2(field):
        vals = [r[field] for r in gr]
        return '%.1f ± %.1f' % (np.mean(vals), np.std(vals))
    print('| %2d | %5d | %17s | %17s | %17s | %18s | %15s |' % (
        g, n,
        ms('participation_rate'),
        ms2('max_cum_eps'),
        ms2('mean_cum_eps'),
        ms2('total_payment'),
        ms('best_acc'),
    ))

print()
print('Field definitions:')
print('  participation_rate : avg fraction of N devices participating per round')
print('  max_cum_eps        : max cumulative epsilon across all devices over 100 rounds')
print('  mean_cum_eps       : mean cumulative epsilon across all devices over 100 rounds')
print('  total_payment      : sum of payments to all devices over all rounds')
print('  best_acc           : max test accuracy (%) over all rounds')
print()
print('Note: max_cum_eps = eps_star * participation_count; mean_cum_eps ~= avg_eps_per_round * 100')
