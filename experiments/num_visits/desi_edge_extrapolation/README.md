# DESI spectrum edge extrapolation study

This directory preserves the exploratory scripts, sample IDs, measurements, and
figures used to compare constant, power-law, and regularized-slope continuation
of DESI galaxy spectra toward the LSST wavelength limits.

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

The regularized tangent, adaptive-window studies, and ivar endpoint cut in
this directory are exploratory only. They are not wired into matrix building.
The unregularized tangent appears in historical comparisons but is omitted
from the final three-method comparison.

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
