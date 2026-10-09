# DESI spectrum edge extrapolation study

This directory preserves the exploratory scripts, sample IDs, measurements, and
figures used to compare constant, power-law, and regularized-slope continuation
of DESI galaxy spectra toward the LSST wavelength limits.

The [physical endpoint quality-cut comparison](quality-cut-native-endpoint-comparison.png)
and [validation-error comparison](quality-cut-validation-comparison.png) show
the threshold/run-count study (the outlined 10% cell is the historical trial).
Regenerate them with `python experiments/num_visits/desi_edge_extrapolation/plot_quality_cut_comparison.py`.

Start with [the final tuning report](edge-tuning-report.md) and
[the separate-test redshift comparison](edge-tuning-test-redshift.png).
Constant has the lowest aggregate median test broadband error on both sides;
there is no established blue power-law advantage on the separate sample.
The artificial cutoffs cannot determine the optimal ivar threshold at the
physical DESI endpoints. Full missing u/y flux, downstream NMF performance,
and the visit prior have not been validated.

## Implementation versus experiments

The package implementation in `training_matrix.py` offers constant (default),
robust linear, and robust power-law edge extrapolation through
`desi.build_matrix --edge-extrapolation`. Measured bins and internal gaps are
preserved, and inferred tails receive downweighted fitting weights.

The active comparison uses constant and power-law continuation. Regularized
slope has been dropped from consideration; its scripts and results remain here
as historical research records. The regularized tangent and adaptive-window
studies were exploratory only. Matrix building now uses a fixed permissive 5% relative-ivar /
three-consecutive-bin endpoint cut, with a 50–300 observed-Angstrom reference
window. The historical 10% trial and tuning results below remain archived as
original measurements; they do not establish an optimal threshold.
The unregularized tangent appears in historical comparisons but is omitted
from the historical three-method comparison.

## Reproducing the study

These are snapshots of local exploratory scripts, not installed CLI commands.
They use `Path(__file__).parent` for outputs and retain the original absolute
input paths under `/home/ashandonay/scratch/bedcosmo/`. The DESI training NPZ and
native coadds are external inputs and are not committed. Adjust those input
paths to use the same datasets on another machine. Running the scripts writes
outputs in this directory; use a writable copy to preserve the archived results.

With the package installed in the `bedcosmo` conda environment and the original
input datasets available, the final study is reproduced with:

```bash
conda activate bedcosmo
python experiments/num_visits/desi_edge_extrapolation/tune_edge_quality.py
python experiments/num_visits/desi_edge_extrapolation/summarize_edge_tuning.py
```

`tune_edge_quality.py` reads the earlier `try_regularized_slope.py` function
bodies without executing its driver. Earlier stratified/regularized CSVs record
which galaxies must be excluded from the independent sample. These dependencies
are included here. The protocol JSON records the input rows, TARGETIDs, seeds,
holdout widths, screening criteria, scoring, and limitations. Diagnostic tail
fits and synthetic checks use the package's power-law helper.

The native edge audit (`audit_desi_blue_edges.py`) reads original coadds and
checks the saved matrix against the existing rest-frame binning. The ivar
study (`check_edge_ivar.py`) produces a full-population per-object CSV
`edge-ivar-threshold-diagnostic.csv`, omitted here because it is 17 MB; its
aggregate summary and protocol are included, and the script regenerates it.
The remaining scripts preserve the preceding comparisons and example plots.

## Current constant versus power-law comparison

Constant remains the pipeline default. The latest diagnostic holds out the
outer 300 observed-frame Angstroms on each side of the same 400 galaxies and
compares constant (100 rest-frame Angstrom window) with power law (500).
The [three-slice comparison](binned-300-edge-error-distributions.png) scores
three 100-observed-Angstrom means and their RMS, preventing cancellation
between slices. Neither method has a clear median advantage. Reference noise
remains, particularly at blue wavelengths; this reused exploratory sample does
not validate full LSST extrapolation or downstream NMF performance.

[Five actual extensions](five-desi-lsst-extrapolation-methods.png),
[whole-region errors](last-300-edge-error-distributions.png), and
[500 versus 1000 Angstrom power-law windows](powerlaw-1000-window-error-distributions.png)
are saved with their CSV measurements and JSON protocols.

To reproduce the current diagnostics in a writable copy:

```bash
python evaluate_last_300_edges.py
python plot_last_300_error_distributions.py
python evaluate_binned_300_edges.py
python plot_five_desi_lsst_extensions.py
```

New production matrices are built separately in `desi_training_data_extrapolated`;
full `desi.build_prior` builds now default to that matrix. The original
`desi_training_data` matrix is preserved. See the package DESI README for commands.
