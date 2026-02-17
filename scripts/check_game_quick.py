#!/usr/bin/env python3
"""Quick game analysis for actual experiment gamma values."""
import numpy as np
import sys; sys.path.insert(0, '.')
from src.devices.heterogeneity import HeterogeneityGenerator
from src.game.stackelberg import StackelbergSolver

gen = HeterogeneityGenerator(n_devices=50, config_path='config/devices/heterogeneity.yaml', seed=42)
devices = gen.generate()

print('=== Game results for actual experiment gamma values ===')
print(f'{"gamma":>5s}  {"part":>4s}  {"avg_eps":>7s}  {"noise_scale":>11s}  {"SNR":>5s}')
print('-' * 40)

for gamma in [3, 5, 7, 10, 15, 20]:
    solver = StackelbergSolver(gamma=gamma)
    result = solver.solve(devices)
    participants = [d for d in result['decisions'] if d.participates]
    if not participants:
        print(f'{gamma:5d}  NONE')
        continue
    eps_vals = [d.eps_star for d in participants]
    n_part = len(participants)
    avg_eps = np.mean(eps_vals)
    
    C = 5.0
    sensitivity = 2.0 * C / n_part
    noise_scale = sensitivity / avg_eps
    snr = 3.0 / noise_scale  # approximate
    
    print(f'{gamma:5d}  {n_part:4d}  {avg_eps:7.3f}  {noise_scale:11.4f}  {snr:5.2f}')

# Also compute: how many public samples per class?
print(f'\n=== Public data coverage ===')
print(f'2000 public samples / 100 classes = 20 samples/class on average')
print(f'Each distill epoch: 2000/256 = ~8 batches')
print(f'5 distill epochs/round = ~40 gradient steps from 2000 samples')
print(f'100 rounds * 40 = 4000 total gradient steps on just 2000 samples')
print(f'\nThis is SEVERE overfitting to 2000 images!')
print(f'The server model memorizes the 2000 public images instead of generalizing.')
