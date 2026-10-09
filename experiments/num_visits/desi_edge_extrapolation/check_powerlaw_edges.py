import json
from pathlib import Path
import numpy as np
import pandas as pd
from bedcosmo.num_visits.empirical.desi.training_matrix import extrapolate_spectrum_edges

out = Path('/home/ashandonay/.codex/visualizations/2026/10/07/01a118ba-2514-7622-b640-68ff1ec126ec')
d = np.load('/home/ashandonay/scratch/bedcosmo/num_visits/desi_training_data/desi_rest_frame_training_matrix.npz')
w = d['wave_rest_aa']
rows = []
failures = []
for row in np.random.default_rng(42).choice(len(d['redshift']), 100, replace=False):
    flux = d['flux'][row]
    ivar = d['relative_ivar'][row]
    observed = ivar > 0
    indices = np.flatnonzero(observed)
    z = float(d['redshift'][row])
    for side, width in [('blue', 300.0), ('red', 800.0)]:
        held = observed & ((w <= w[indices[0]] + width / (1 + z)) if side == 'blue' else (w >= w[indices[-1]] - width / (1 + z)))
        training_weight = ivar.copy()
        training_weight[held] = 0
        actual_mean = np.average(flux[held], weights=ivar[held])
        mean_sigma = np.sqrt(1 / np.sum(ivar[held]))
        for method in ('constant', 'powerlaw'):
            try:
                pred, _ = extrapolate_spectrum_edges(w, flux, training_weight, method=method)
            except ValueError as error:
                failures.append({'row': int(row), 'side': side, 'method': method, 'error': str(error)})
                continue
            pred_mean = np.average(pred[held], weights=ivar[held])
            rows.append({'row': int(row), 'targetid': int(d['targetid'][row]), 'side': side, 'method': method, 'z': z, 'true_mean': float(actual_mean), 'mean_sigma': float(mean_sigma), 'pred_mean': float(pred_mean), 'fractional_error': float(abs(pred_mean - actual_mean) / abs(actual_mean)) if actual_mean > 3 * mean_sigma else np.nan})
table = pd.DataFrame(rows)
table.to_csv(out / 'powerlaw-heldout-edge-check.csv', index=False)
summary = {'n_galaxies': 100, 'seed': 42, 'heldout_observed_width_aa': {'blue':300, 'red':800}, 'failures': failures, 'metric': 'absolute fractional error of inverse-variance-weighted mean flux; true mean must exceed 3 sigma', 'sides':{}}
for side in ('blue', 'red'):
    paired = table[table.side == side].pivot(index='row', columns='method', values='fractional_error').dropna()
    summary['sides'][side] = {'n_pairs':len(paired), 'constant_median_fractional_error':float(paired.constant.median()), 'powerlaw_median_fractional_error':float(paired.powerlaw.median()), 'powerlaw_win_fraction':float((paired.powerlaw < paired.constant).mean())}
(out / 'powerlaw-heldout-edge-check.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
