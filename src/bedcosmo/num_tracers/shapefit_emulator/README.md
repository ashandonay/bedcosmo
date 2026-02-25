# ShapeFit Emulator Workflow

This folder contains a lightweight pipeline to:

1. generate a training dataset from `desilike`'s `ShapeFitPowerSpectrumExtractor`,
2. train a PyTorch MLP to emulate ShapeFit outputs,
3. plot basic dataset diagnostics.

Targets predicted by the model:

- `qiso`
- `qap`
- `f_sigmar`
- `m`

## Scripts

- `prep_data.py`: build `shapefit_train.npz` and `shapefit_test.npz`
- `train_nn.py`: train MLP and save checkpoint (`.pt`)
- `plot_data.py`: visualize train/test distributions
- `predict.py`: Python helper functions for inference from a saved checkpoint

## Quick Start

Run from repo root:

```bash
# 1) Generate data
python src/bedcosmo/num_tracers/shapefit/prep_data.py --n-samples 10000 --sigma-clip 4

# 2) Plot dataset diagnostics
python src/bedcosmo/num_tracers/shapefit/plot_data.py \
  --data-path "$SCRATCH/bedcosmo/num_tracers/shapefit"

# 3) Train emulator
python src/bedcosmo/num_tracers/shapefit/train_nn.py \
  --data-path "$SCRATCH/bedcosmo/num_tracers/shapefit" \
  --model-path "$SCRATCH/bedcosmo/num_tracers/shapefit/shapefit_mlp.pt"
```

## `prep_data.py` Arguments

- `--n-samples` (int, default `10000`): number of accepted samples to keep.
- `--batch-size` (int, default `256`): LHS samples drawn per loop.
- `--seed` (int, default `0`): random seed for sampling.
- `--test-size` (float, default `0.2`): test fraction for train/test split.
- `--save-path` (str, default `$SCRATCH/bedcosmo/num_tracers/shapefit`): output directory.
- `--sigma-clip` (float, default `4.0`): truncation for normal priors (`mu +/- sigma_clip*sigma`).
- `--verbose-every` (int, default `200`): progress print cadence (in attempts modulo behavior).
- `--omega-m-min` (float, default `0.05`): minimum allowed derived `Omega_m`.
- `--omega-m-max` (float, default `1.0`): maximum allowed derived `Omega_m`.
- `--priors-json` (str, default empty): override default priors with a JSON dictionary.

Default priors:

- `omega_cdm ~ U(0.01, 0.99)`
- `omega_b ~ N(0.02218, 0.00055^2)` (truncated at `sigma-clip`)
- `h ~ U(0.2, 1.0)`
- `ln10A_s ~ U(1.61, 3.91)`
- `n_s ~ N(0.9649, 0.042^2)` (truncated at `sigma-clip`)

Internally, extractor inputs are:

- `Omega_m = (omega_cdm + omega_b) / h^2`
- `n_s`

Outputs saved:

- `<save-path>/shapefit_train.npz`
- `<save-path>/shapefit_test.npz`

Each NPZ contains:

- `x`: input parameters
- `y`: ShapeFit targets
- `param_names`
- `target_names`

## `plot_data.py` Arguments

- `--data-path` (str): folder containing NPZ files.
- `--output-dir` (str, optional): where to write plots (defaults to `data-path`).
- `--bins` (int, default `40`): histogram bins.

Generated plots:

- `inputs_train_test_hist.png`
- `targets_train_test_hist.png`
- `omega_m_vs_qiso.png`

## `train_nn.py` Arguments

- `--data-path` (str, default `.`): location of NPZ files.
- `--model-path` (str, default `./shapefit_mlp.pt`): output checkpoint path.
- `--epochs` (int, default `1000`): training epochs.
- `--batch-size` (int, default `256`): mini-batch size.
- `--lr` (float, default `2e-3`): AdamW learning rate.
- `--weight-decay` (float, default `1e-6`): AdamW weight decay.
- `--hidden-dim` (int, default `256`): hidden width of each MLP block.
- `--n-hidden` (int, default `3`): number of hidden blocks.
- `--seed` (int, default `0`): numpy/torch seed.

Training prints:

- periodic train/test normalized MSE
- per-target MAE and RMSE in physical target space

## Prediction (Python API)

`predict.py` provides:

- `load_emulator(model_path)`
- `predict_shapefit(model_path, params)`

Example:

```python
from bedcosmo.num_tracers.shapefit.predict import predict_shapefit

params = {
    "omega_cdm": 0.12,
    "omega_b": 0.0222,
    "h": 0.68,
    "ln10A_s": 3.04,
    "n_s": 0.965,
}
pred = predict_shapefit("/path/to/shapefit_mlp.pt", params)
print(pred)
```

## Notes / Troubleshooting

- JAX/CUDA warnings from `desilike` are often non-fatal for this workflow.
- If acceptance is very low, tighten priors and/or `omega_m` bounds.
- If `SCRATCH` is not set, pass `--save-path` explicitly.
