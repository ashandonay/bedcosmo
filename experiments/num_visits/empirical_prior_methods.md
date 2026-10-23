# Empirical SED priors: ILR and joint activation-pattern methods

Both methods start from the **same fitted DESI population and NMF template bank**.
Their purpose is to generate coefficient/scale/redshift samples that reproduce
that fitted population. Neither method repairs a poor spectral basis or
independently establishes that the DESI sample represents the LSST population.

## Shared inputs and physical meaning

For a galaxy, the NMF fit gives nonnegative coefficients `c_k`, redshift `z`,
and an overall scale. Write `s = sum(c_k)` and `a_k = c_k/s`, so `sum(a_k)=1`.
The physical prior row is `(a_1,...,a_K,log(s),z)`.

The DESI builder learns templates on measured wavelengths, then extrapolates
finished template edges for LSST coverage. Extrapolation does not add measured
pixels to the NMF objective. In this experiment both prior methods use those
same finished spectra and the same quality-passing coefficient rows. Their
main difference is how they model the distribution of those rows.

## Existing ILR pipeline

1. Replace exact-zero fractions with small positive contributions using the
   configured epsilon (`1e-5`), and normalize. This moves every row into the
   interior of the simplex. Exact activation patterns are no longer retained.
2. Take centered log-ratios, then express them in a fixed orthonormal basis of
   the sum-zero hyperplane. Eight fractions become seven ILR coordinates.
   The ILR change of coordinates is invertible for positive fractions.
3. Append `log(s)` and `z`, standardize the nine coordinates globally, and fit
   a Gaussian KDE. The saved production DESI8 KDE uses bandwidth `0.3`.
   Smoothing in log-ratio coordinates changes physical contributions
   nonlinearly. Our coefficient-first tests show that this bandwidth spreads
   probability away from the fitted coefficient population more than smaller
   bandwidths do.
4. Draw KDE samples with the existing training-bound rejection. Decode ILR
   samples with the inverse map when physical template fractions are needed.
   Decoding introduces no additional fitted approximation.
5. Fit a prior normalizing flow to the KDE samples. The existing pipeline has
   a native ILR-space NF and a second gaussianized-space NF for its transformed
   posterior/EIG path. The production configuration samples its prior pool
   from the native flow and uses the appropriate flow density when scoring.

The representation is continuous and has one common dimension. This is
convenient for the existing posterior model, but exact zeros have already
been replaced before the KDE or NF is trained.

## New joint-KDE pipeline

1. Keep the fitted physical fractions, including exact zeros. Define the
   activation pattern `M = (a_k > 0)`, a binary mask over the template bank.
2. Count observed patterns: `P(M) = number of galaxies with M / N`.
   Per-template absence probabilities are implied by these joint pattern
   probabilities; they are not independently sampled Bernoulli variables.
3. Conditional on a pattern, choose a complete fitted galaxy uniformly. Its
   active fractions, log scale, and redshift are the **shared center** of one
   product Gaussian kernel. Sharing the center preserves observed joint
   structure; adding independent noise within a kernel does not make the
   whole mixture independent across features.
4. Each kernel width is `h * within-pattern feature standard deviation`.
   Reflect fractions into `[0,1]`, reflect scale/redshift into global training
   bounds, retain inactive fractions at exactly zero, and normalize the
   active fractions. This boundary/closure operation changes the raw Gaussian
   kernel density. Bandwidth `0.1` in this experiment is in within-pattern
   physical-feature units, not the same coordinate units as ILR bandwidth.
5. Sample as many joint physical prior rows as needed. No CLR/ILR basis or
   empirical gaussianizer is used in this KDE construction.

The earlier independent-coefficient prototype is only a diagnostic baseline.
It sampled zero indicators and positive values independently and paired them
with independent scale/redshift rows. Its loss of correlations was a result
of that independence assumption, not an inevitable feature of zero-mass KDEs.

## NF prior for the new method

A continuous flow cannot assign positive probability mass to an exact-zero
face of a larger continuous space. We therefore retain the discrete pattern
model and fit continuous flows **within** its faces.

For a pattern with `m` active templates, there are only `m-1` independent
fractions. We use sequential stick fractions in ascending template-bank order:

```
v_j = a_j / (a_j + ... + a_m),     j = 1,...,m-1
u_j = log(v_j / (1-v_j))
```

For active fractions `(0.2,0.5,0.3)`, the sticks are `(0.2,0.625)`.
The flow's two shape coordinates are their logits. Given the mask and sticks,
the original fractions are recovered by successively allocating the remaining
mass. Inactive templates are never part of this transformation and remain zero.
This is an explicit constraint transform for the **NF stage**, not a claim
that the complete pipeline is transform-free. It avoids CLR/ILR rotations.

Log scale and redshift are also mapped with logits between their global saved
training bounds. The resulting `m+1` continuous coordinates are standardized
using only the NF training draws. There is no empirical CDF gaussianization.

Rather than training a network separately for every pattern, one conditional
zuko neural-spline flow is shared by patterns having the same active count `m`.
Its context is the full binary template mask. Context tells the shared flow
which bank components the ordered active coordinates refer to. It does not
choose the pattern or change its empirical probability.

Patterns represented by exactly one fitted galaxy have zero kernel width in
all coordinates. They are kept as explicit point masses at that galaxy's row;
training a continuous NF on such a point would be mathematically inappropriate.
Non-singleton patterns with zero-width continuous coordinates fail with a clear
error rather than receiving hidden jitter or a different fallback model.
The current full-redshift build has 156 patterns, 28 singleton atoms (0.226%
of its population), and seven continuous NF groups (`m=2,...,8`).

Training draws are generated from the joint KDE, with independent KDE draws
used to choose the best checkpoint by validation negative log-likelihood.
Fresh seeds are used again for the final NF-fidelity check. These are tests
of the NF approximation to its KDE target; they are separate from holding out
DESI galaxies to assess how the KDE itself generalizes.

### Sampling and scoring

To sample the NF prior:

1. Draw the pattern from its empirical categorical distribution.
2. If it is a singleton, return its stored row.
3. Otherwise select the flow for its active count, condition on the mask,
   draw continuous coordinates, and invert the standardization and logits.

For a continuous face, scoring returns

```
log p(M) + log q_flow(standardized logits | M)
         - sum(log training_std) + log |d logits / d physical_face_row|.
```

The stick-logit Jacobian in the first `m-1` active-fraction coordinates is
`-sum(log(a_active))`. For an auxiliary coordinate `x` bounded by `(L,U)`,
its logit Jacobian is `log(U-L)-log(x-L)-log(U-x)`.
Singletons return their categorical log probability mass at the exact point;
other values on that atomic pattern have zero probability. Unknown patterns
also have zero probability. There is no finite-density surrogate for atoms.

This NF has a fast direct score in its own mixed reference measure. It does
not need the radial quadrature required to evaluate the exact reflected and
normalized **KDE** density. Its density is a learned approximation to that KDE,
so fresh-sample validation remains necessary.

## Running the new NF stage

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
python scripts/joint_empirical_prior_flow.py \
  --prior-dir "$SCRATCH/bedcosmo/num_visits/empirical_prior/desi8_full_redshift" \
  --output-dir experiments/num_visits/plots/joint_prior_nf/desi8_full_redshift_v1 \
  --bandwidth 0.1 --n-samples 120000 --epochs 150 --patience 25 \
  --hidden 64 --transforms 3 --bins 8 --batch-size 1024 --threads 2 --seed 23

python scripts/validate_joint_empirical_prior_flow.py \
  --flow-path experiments/num_visits/plots/joint_prior_nf/desi8_full_redshift_v1/joint_prior_flow.pt \
  --output-dir experiments/num_visits/plots/joint_prior_nf/desi8_full_redshift_v1/validation \
  --n-samples 20000 --seed 991
```

The training CLI exposes precisely these configuration fields. Defaults match
this example. `--prior-dir` is read-only; `--output-dir` must be a new directory
outside the source build. Training is CPU-only and caps PyTorch threads with
`--threads`. Validation uses two PyTorch threads. Run it where CPU time is
available; these commands do not use `submit.sh` or create MLflow runs.

The training output contains `request.json`, `joint_kde_fit.npz`, per-group
partial checkpoints, `training.json`, and the completed `joint_prior_flow.pt`.
Partial checkpoints are clearly named `completed_through_active*.pt`; they
are not complete prior artifacts. Load the final artifact using
`JointPriorFlow.load(path)` from the script, then call `sample(n, seed=...)`
or `log_prob(physical_rows)`. It includes pattern masses, singleton rows,
constraint bounds, NF weights, normalization constants, and source fingerprints.
For interactive use from the repository root, put the scripts directory on the
Python path explicitly:

```bash
PYTHONPATH=scripts python - <<'PYTHON'
from joint_empirical_prior_flow import JointPriorFlow
flow = JointPriorFlow.load(
    "experiments/num_visits/plots/joint_prior_nf/desi8_full_redshift_v1/joint_prior_flow.pt"
)
rows = flow.sample(1000, seed=42)  # a1..a8, log_c_scale, z
log_prob = flow.log_prob(rows)
pattern_ids, gaussian_groups, log_abs_det = flow.gaussianize(rows)
restored = flow.degaussianize(pattern_ids, gaussian_groups)
PYTHON
```

Validation writes `fresh_samples.npz`, `validation.json`, and PNG/PDF marginal
and convergence plots. It checks valid simplex rows, exact support masks and
finite scores on fresh target draws. Reported distribution discrepancies and
runtime measurements are descriptive; they are not formal significance tests
or guarantees of performance in rare patterns.

### Gaussian coordinates from the trained NF

The saved flow also provides an invertible Gaussianization without retraining:
physical row → stick/aux logits → saved standardization → conditional spline
transform → standard Gaussian base coordinates. The inverse follows the same
steps in reverse. This uses the learned joint transform, rather than fitting
empirical marginal CDF tables as in the existing pipeline.

`gaussianize(rows)` returns `(pattern_ids, gaussian_groups, log_abs_det)`.
The IDs retain input row order. `gaussian_groups[m]` has shape `(n_m, m+1)`,
in the input order of non-atomic rows with `m` active templates. The returned
log Jacobian includes the physical chart, standardization and spline transform.
`degaussianize(pattern_ids, gaussian_groups)` restores physical rows, including
exact inactive zeros and singleton points. IDs are local to the saved artifact.
Singleton atoms have no Gaussian coordinates and have zero log Jacobian.

Conditioned on any continuous pattern, NF-generated rows have a standard
Gaussian base law by construction. Fresh KDE draws become only approximately
Gaussian, to the accuracy of the NF fit. Their physical log density is
`log P(pattern) + log Normal(base; 0,I) + log_abs_det`; singleton rows instead
retain their pattern probability masses. The discrete pattern cannot itself
be mapped invertibly to a continuous Gaussian. This representation consequently
does not yet supply the fixed-dimensional posterior required by the current
BED runtime.

The validation command above also writes `gaussianization.json`,
`gaussian_coordinates.npz`, and `gaussianization_qq.png/.pdf`. It checks inverse
accuracy, exact masks, and the density/Jacobian identity for independent KDE
and NF draws. It reports per-coordinate means, standard deviations and KS
distances to a standard Gaussian, plus covariance departures from identity,
pooled by active count. These pooled checks do not establish Gaussianity for
every individual pattern or joint independence beyond second moments.

On 20,000 fresh draws from each distribution, maximum physical round-trip
error was below `8e-6` for every feature, with exact mask preservation; maximum
log-density identity error was below `8e-6`. For KDE draws in the common
`m=4,...,8` groups, largest marginal KS distances ranged from 0.038 to 0.052.
The `m=2` KDE group had only 18 draws, insufficient for a useful Gaussianity
assessment. Q–Q curves near the diagonal indicate approximately Gaussian
marginals; residual departures measure the learned NF's approximation to its
KDE target. These checks did not retrain or change the prior artifact.

## What changes and what stays shared

| Aspect | Existing method | New method |
|---|---|---|
| Template spectra and DESI coefficient fits | Shared inputs | Shared inputs |
| Exact zeros | Epsilon-smoothed away | Explicit activation patterns |
| KDE coordinates | Global standardized ILR + scale/z | Physical coordinates within a pattern |
| Kernel center | Complete ILR/scale/z row | Complete active-fraction/scale/z row |
| Bound handling | Reject outside training feature box | Reflect before simplex closure |
| NF shape coordinates | ILR basis | Sequential stick logits on active face |
| NF dimensionality | Fixed `K+1` | `m+1`, sharing networks by active count |
| Discrete mass | None after smoothing | Empirical patterns and singleton atoms |
| Density evaluation | Continuous flow score | Pattern mass + face flow + Jacobian, or atom mass |
| Existing BED posterior compatibility | Already integrated | Still requires a compatible posterior representation |

The new sampler's observed coefficient fidelity is comparable to suitably
narrow-bandwidth ILR in our tests; it also preserves sparsity explicitly.
Neither method has an automatic accuracy advantage. Bandwidth choice,
finite-sample coverage and the quality of the underlying template fits matter.
Observed patterns absent from the training population cannot be generated by
the empirical categorical model.

The new NF is an offline learned prior artifact, not yet a drop-in replacement
for `NumVisits`' current `density_type: flow` path. Prior and posterior scores
must use compatible pattern/face/atom measures before estimating EIG.
Training a prior NF does not by itself complete that posterior/EIG integration.

## First trained artifact: full-redshift DESI8 build

The first run trained seven conditional NSF groups on 120,000 joint-KDE draws
at physical bandwidth 0.1, with 24,000 independent target draws for checkpoint
selection. Each group completed 150 epochs on two CPU threads, using three
spline transforms, eight bins and two 64-unit hidden layers. Training the
groups took approximately 307 seconds in total, excluding input preparation.

A further 20,000 fresh target draws and 20,000 fresh NF draws gave:

| Comparison | Joint coefficient MMD² | Largest one-coordinate KS |
|---|---:|---:|
| NF vs joint-KDE target | 0.001088 | 0.0202 |
| Joint KDE vs fitted DESI population | 0.001373 | 0.0268 |
| NF vs fitted DESI population | 0.001600 | 0.0255 |

MMD uses the same training-scaled physical coordinates and three 700-row metric
subsamples as the coefficient comparison. These are descriptive fidelity
checks, not significance tests; the DESI rows built the target KDE and are not
a new galaxy holdout. All fresh KDE rows received finite mixed-model scores,
and the generated rows retained valid simplex sums and known exact support
patterns. Pattern frequencies fluctuate with independent Monte Carlo draws;
the model itself retains the exact empirical pattern probabilities.

On this CPU run, generating 20,000 NF samples took about 0.96 seconds and
scoring 20,000 fresh target rows took 0.14 seconds. The joint KDE sampler was
faster to sample (0.014 seconds); an NF is not required to speed up sampling.
Its direct density score avoids the exact KDE's radial integration. This is
not a benchmark against the old prior NF or a production EIG throughput claim.

The model and validation outputs are generated locally by the commands above,
under the ignored plots directory. Weights and input datasets are not committed
to the repository. Production prior selection is unchanged.
