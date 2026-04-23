import json
import numpy as np
from pathlib import Path

TMC = Path('results/experiments/tmc')
v101 = json.load(open('results/experiments/v10_1_3seeds_20260409_0922.json'))
key = list(v101['runs'].keys())[0]
paid_acc = np.array(v101['runs'][key]['accuracies'][:100]) * 100
print("PAID-FD (%s) r0=%.1f r10=%.1f r50=%.1f r99=%.1f" % (
    key, paid_acc[0], paid_acc[9], paid_acc[49], paid_acc[99]))

for tag in ['expF_faireps1_s42', 'expA_fixedeps3_s42', 'expA_csra_s42',
            'expAp_fedgmkd_s42', 'expAp_fedavg_s42']:
    p = TMC / (tag + '.json')
    if p.exists():
        a = np.array(json.load(open(p))['accuracies'][:100]) * 100
        print("  %-28s r0=%.1f r10=%.1f r50=%.1f r99=%.1f" % (
            tag, a[0], a[9], a[49], a[99]))

run = v101['runs'][key]
eps_vals = np.array(run['avg_eps'][:100])
part_vals = np.array(run['participation_rates'][:100])
eps_star = float(np.mean(eps_vals))
mp = float(np.mean(part_vals))
R = np.arange(1, 101, dtype=float)
bf = np.cumsum(eps_vals)
bs = bf * mp
log_inv_delta = float(np.log(1.0 / 1e-5))
exp_term = float(np.exp(eps_star) - 1)
af = eps_star * np.sqrt(2 * R * log_inv_delta) + R * eps_star * exp_term
asel = eps_star * np.sqrt(2 * R * mp * log_inv_delta) + R * mp * eps_star * exp_term
print("Fig10 R=100: bf=%.1f bs=%.1f af=%.1f asel=%.1f" % (
    bf[99], bs[99], af[99], asel[99]))
print("eps_star=%.3f  mean_part=%.2f  exp_term=%.3f" % (eps_star, mp, exp_term))
