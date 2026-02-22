#!/usr/bin/env python3
"""SNR analysis: single-round vs EMA buffer for Fix 17 design."""
import numpy as np

C = 2.0
game_params = {
    '3':  {'N': 19, 'avg_eps': 0.852},
    '5':  {'N': 35, 'avg_eps': 0.524},
    '7':  {'N': 43, 'avg_eps': 0.408},
    '10': {'N': 50, 'avg_eps': 0.322},
}

print('SNR Analysis: single-round vs EMA buffer')
print('=' * 70)

for gamma, p in game_params.items():
    N = p['N']
    avg_eps = p['avg_eps']

    # BLUE aggregation: sum_eps_sq ~ N * avg_eps^2
    sum_eps_sq = N * avg_eps ** 2
    # BLUE variance per element: (2C)^2 * 2 / sum_eps_sq (Laplace var = 2*scale^2)
    blue_var = (2 * C) ** 2 * 2 / sum_eps_sq
    blue_std = np.sqrt(blue_var)

    signal_range = 2 * C
    snr_single = signal_range / blue_std

    # EMA buffer: effective independent samples = 1/(1-alpha^2)
    for alpha, label in [(0.9, '~5r'), (0.95, '~10r')]:
        eff = 1 / (1 - alpha ** 2)
        snr_ema = snr_single * np.sqrt(eff)
        print(f'  gamma={gamma} EMA({alpha}, {label}): SNR={snr_ema:.2f} (single={snr_single:.2f})')

    print()

# Key ratio
p3 = game_params['3']
p10 = game_params['10']
r = np.sqrt((p3['N'] * p3['avg_eps'] ** 2) / (p10['N'] * p10['avg_eps'] ** 2))
print(f'SNR ratio g3/g10 = {r:.2f}x (constant regardless of EMA window)')
print()

# What this means for teaching quality
print('Teaching quality prediction:')
print('=' * 70)
for alpha in [0.9, 0.95]:
    eff = 1 / (1 - alpha ** 2)
    print(f'\nEMA momentum={alpha} (effective window ~{eff:.0f} rounds):')
    for gamma, p in game_params.items():
        N = p['N']
        avg_eps = p['avg_eps']
        sum_eps_sq = N * avg_eps ** 2
        blue_var = (2 * C) ** 2 * 2 / sum_eps_sq
        blue_std = np.sqrt(blue_var)
        snr = (2 * C / blue_std) * np.sqrt(eff)

        if snr > 3:
            quality = 'GOOD (top-1 prob ~70-80%)'
        elif snr > 1.5:
            quality = 'MODERATE (top-1 prob ~40-60%)'
        else:
            quality = 'POOR (nearly uniform)'
        print(f'  gamma={gamma}: SNR={snr:.1f} -> {quality}')

# Distill contribution estimate
print()
print('Expected distillation contribution (beyond local training):')
print('=' * 70)
print('With distill_lr=0.001, alpha=0.5, EMA(0.9):')
for gamma in ['3', '5', '7', '10']:
    p = game_params[gamma]
    N = p['N']
    avg_eps = p['avg_eps']
    sum_eps_sq = N * avg_eps ** 2
    blue_var = (2 * C) ** 2 * 2 / sum_eps_sq
    blue_std = np.sqrt(blue_var)
    eff = 1 / (1 - 0.9 ** 2)
    snr = (2 * C / blue_std) * np.sqrt(eff)

    # Rough model: distillation contribution proportional to log(SNR) when SNR > 1
    contribution = max(0, np.log(snr)) * 1.5  # ~1.5% per unit of log(SNR)
    print(f'  gamma={gamma}: SNR={snr:.1f}, estimated KL boost ~{contribution:.1f}%')
