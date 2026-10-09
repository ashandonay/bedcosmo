# DESI edge tuning diagnostic

Read-only exploratory analysis; no production package, saved matrix, NMF basis, or visit prior was changed.

## Protocol

- 400 validation galaxies and 400 separate test galaxies, each with 100 per redshift bin [0,0.6), [0.6,0.9), [0.9,1.2), [1.2,2).
- Previously plotted/compared galaxies excluded by TARGETID from both splits. Split IDs and seed are in edge-tuning-protocol.json.
- Quality grid: no cut, or thresholds 0.05/0.10/0.20 times median positive ivar 50–300 observed Angstrom inward, with 1/3/5 consecutive supported bins.
- Methods: constant over 100 rest Angstrom (median-filtered flux); robust power law over 500 rest Angstrom; earlier regularized quadratic endpoint tangent with internal window/shrinkage selection.
- Hide outer 600 observed Angstrom on blue side, 800 on red side. Retain the noisy measured holdout as reference; candidate quality cuts can discard additional training bins, whose prediction errors are also included.
- Reconstruct DESI-covered LSST g (blue test) or z (red test) photon-weighted band flux. Error is absolute change in integrated band flux divided by original measured band flux. Coefficients are wavelength times throughput times wavelength-bin width; ivar is used for measurement uncertainty, not photometric integration.
- Require measured response coverage >=99.5% and band flux S/N >3 on both sides. This is a screened galaxy sample, not the whole DESI population.
- Select quality settings per method and side using validation median broadband error. Freeze selections before scoring test galaxies. Select a winning method per side using validation only.
- Bootstrap 10,000 galaxy resamples for median confidence intervals. These describe sample variation, not total extrapolation/model uncertainty.

## Separate test results

| Method | Median blue g-flux error | Median red z-flux error |
|---|---:|---:|
| Constant | 1.78% | 1.60% |
| Power law | 1.84% | 1.69% |
| Regularized slope | 2.01% | 2.01% |

Validation preferred power law blue and constant red. The blue test median difference (power law minus constant) is +0.061 percentage points, with paired bootstrap 95% interval [-0.228, +0.248] percentage points. The blue validation preference did not establish an advantage on the separate test sample.

## What cannot be tuned with this test

The artificial cutoff is inside well-measured DESI data. All nine quality-cut candidates leave every blue validation endpoint unchanged; the most active red candidates move only 2/400 validation endpoints. The trial 0.10/3-bin cut changes no artificial endpoint in the test sample. Its identical errors to no-cut constant are a consequence of identical retained data, not evidence that real physical endpoints need no filtering.

On physical endpoints of those same 400 test galaxies, the trial 0.10/3-bin cut moves 54.5% of blue endpoints (median loss 11.67 observed Angstrom; p90 23.26), and 3.5% of red endpoints. This establishes how much coverage the rule discards, not the accuracy of the resulting extrapolation. Keep that threshold provisional.

Full u/y extrapolation, latent noiseless continuum, NMF quality, and the downstream visit prior have not been validated. The existing regularized estimator's inner holdouts do not apply the candidate ivar cut; only its outer fit endpoint is cut in this experiment. Constant remains a reasonable simple default given the current results, with no statistically established blue-edge power-law benefit.

Five synthetic checks and result integrity checks passed; no fit failures occurred among evaluated configurations. No GPU/SLURM jobs were submitted.

## Files

- tune_edge_quality.py: sampling, grid scoring, frozen validation selection, test scoring.
- summarize_edge_tuning.py: bootstrap comparison, redshift figure, physical-endpoint trimming diagnostics.
- edge-tuning-protocol.json and edge-tuning-selected.json: reproducible sample/selection records.
- edge-tuning-validation.csv and edge-tuning-test.csv: per-galaxy measurements.
- edge-tuning-test-summary.csv and edge-tuning-test-inference.json: aggregates and paired comparison.
- edge-tuning-native-trimming-summary.csv: physical-endpoint coverage loss.
