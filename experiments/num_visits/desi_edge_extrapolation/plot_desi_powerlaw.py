from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from speclite import filters
from bedcosmo.num_visits.empirical.desi.training_matrix import extrapolate_spectrum_edges, _extrapolate_powerlaw

matrix = np.load('/home/ashandonay/scratch/bedcosmo/num_visits/desi_training_data/desi_rest_frame_training_matrix.npz')
wave = matrix['wave_rest_aa']
redshifts = matrix['redshift']
flux_matrix = matrix['flux']
weights = matrix['relative_ivar']
lsst_filters = filters.load_filters(*[f'lsst2023-{band}' for band in 'ugrizy'])
lsst_blue = min(f.wavelength.min() for f in lsst_filters)
lsst_red = max(f.wavelength.max() for f in lsst_filters)

valid = weights > 0
first = np.argmax(valid, axis=1)
last = len(wave) - 1 - np.argmax(valid[:, ::-1], axis=1)
score = np.maximum(wave[first] - lsst_blue / (1 + redshifts), 0) + np.maximum(
    lsst_red / (1 + redshifts) - wave[last], 0
)
examples = []
median_weight = np.nanmedian(np.where(valid, weights, np.nan), axis=1)
for zlo, zhi in ((0.0, 0.1), (0.4, 0.6), (0.9, 1.1)):
    eligible = (redshifts >= zlo) & (redshifts < zhi) & (score > 0)
    examples.append(int(np.argmax(np.where(eligible, median_weight, -1))))

colors = ['#665191', '#2f7f83', '#c66b24', '#4c78a8', '#8a6d3b', '#a05195']
fig, axes = plt.subplots(4, 1, figsize=(13.5, 11), sharex=True, constrained_layout=True,
                         gridspec_kw={'height_ratios': [0.45, 1, 1, 1]})
for j, band in enumerate(lsst_filters):
    axes[0].plot(band.wavelength, band.response, color=colors[j], lw=1.5, label=band.name[-1])
axes[0].set_ylabel('Throughput')
axes[0].set_ylim(0, 1.05)
axes[0].legend(ncol=6, frameon=False, loc='upper right', title='LSST bands')
axes[0].grid(axis='y', alpha=0.2)

for ax, row in zip(axes[1:], examples):
    z = float(redshifts[row])
    target_lo, target_hi = lsst_blue / (1 + z), lsst_red / (1 + z)
    step = float(np.median(np.diff(wave)))
    grid_lo = np.floor(min(target_lo, wave[0]) / step) * step
    grid_hi = np.ceil(max(target_hi, wave[-1]) / step) * step
    extended_wave = np.arange(grid_lo, grid_hi + step / 2, step)
    flux = np.zeros(len(extended_wave), dtype=float)
    weight = np.zeros(len(extended_wave), dtype=float)
    offset = int(round((wave[0] - grid_lo) / step))
    flux[offset:offset + len(wave)] = flux_matrix[row]
    weight[offset:offset + len(wave)] = weights[row]
    original_indices = np.flatnonzero(weight > 0)
    measured_lo, measured_hi = extended_wave[original_indices[[0, -1]]] * (1 + z)
    x_obs = extended_wave * (1 + z)

    constant, _ = extrapolate_spectrum_edges(extended_wave, flux, weight, method='constant')
    powerlaw, _ = extrapolate_spectrum_edges(extended_wave, flux, weight, method='powerlaw')
    within_lsst = (x_obs >= lsst_blue) & (x_obs <= lsst_red)
    left_tail = within_lsst & (x_obs < measured_lo)
    right_tail = within_lsst & (x_obs > measured_hi)
    measured = (weight > 0) & within_lsst

    ax.plot(x_obs[measured], flux[measured], color='#20242b', lw=1.0, label='measured DESI')
    ax.plot(x_obs[left_tail], constant[left_tail], color='#2878b5', lw=2.0, ls='--', label='constant edge')
    ax.plot(x_obs[right_tail], constant[right_tail], color='#2878b5', lw=2.0, ls='--')
    ax.plot(x_obs[left_tail], powerlaw[left_tail], color='#e1812c', lw=2.0, ls=':', label='power-law fit / extension')
    ax.plot(x_obs[right_tail], powerlaw[right_tail], color='#e1812c', lw=2.0, ls=':')
    for endpoint, direction in ((original_indices[0], 1), (original_indices[-1], -1)):
        edge_select = (weight > 0) & (direction * (extended_wave - extended_wave[endpoint]) >= 0) & (direction * (extended_wave - extended_wave[endpoint]) <= 500)
        fit_wave = extended_wave[edge_select]
        fit_flux = _extrapolate_powerlaw(fit_wave, flux[edge_select], weight[edge_select], extended_wave[endpoint], fit_wave)
        ax.plot(fit_wave * (1 + z), fit_flux, color='#e1812c', lw=1.5)
        ax.axvspan(fit_wave[0] * (1 + z), fit_wave[-1] * (1 + z), color='gray', alpha=0.10)
    ax.axvline(measured_lo, color='#20242b', lw=0.8, alpha=0.55)
    ax.axvline(measured_hi, color='#20242b', lw=0.8, alpha=0.55)
    plotted = np.concatenate((flux[measured], constant[within_lsst], powerlaw[within_lsst]))
    ymin, ymax = np.nanmin(plotted), np.nanmax(plotted)
    padding = 0.08 * (ymax - ymin)
    ax.set_ylim(ymin - padding, ymax + padding)
    ax.set_ylabel('Flux / robust scale')
    ax.set_title(f'z = {z:.2f}  ·  target {int(matrix["targetid"][row])}', loc='left', fontsize=11)
    ax.grid(axis='y', alpha=0.2)

axes[-1].set_xlim(lsst_blue, lsst_red)
axes[-1].set_xlabel('Observed wavelength [Å]')
axes[1].legend(loc='upper right', ncol=3, frameon=False, fontsize=9)
fig.suptitle('Positive power-law continuation across the full LSST range', fontsize=15)
fig.text(0.5, -0.005, 'Solid black: measured DESI bins shifted to observed wavelength. Dashed/dotted: constant and positive power-law extrapolations; vertical lines mark measured endpoints. Full flux ranges shown; shaded regions are the 500 Å rest-frame fit windows.', ha='center', fontsize=9)
out = Path('/home/ashandonay/.codex/visualizations/2026/10/07/01a118ba-2514-7622-b640-68ff1ec126ec/desi-lsst-powerlaw-examples.png')
fig.savefig(out, dpi=180, bbox_inches='tight')
print(out)
print([(int(matrix['targetid'][i]), round(float(redshifts[i]), 3)) for i in examples])
