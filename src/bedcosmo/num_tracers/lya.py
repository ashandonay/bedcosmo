"""How the Lya QSO bin enters the num_tracers design.

Tracer QSOs (the QSO bin, 0.8 < z < 2.1) and Lya QSOs come from one QSO target
selection: which side of z = 2.1 a target lands on is known only after its spectrum.
A design can therefore move the Lya sample only together with the QSO class.

The ``Lya QSO`` row of desi_data.csv / desi_tracers.csv carries DESI's Lya x QSO
*tracer* count: quasar positions at z > 1.77 (DR1 709,565, DESI 2024 IV Table 1;
DR2 1,289,874, DESI DR2 Lya BAO paper Sec 2). That sample overlaps the QSO bin over
1.77 < z < 2.1, so its ``observed`` column double counts those fibers. Only quasars
with a Lya forest (z > ~2.1) are fibers the QSO bin does not already hold:
DR1 531,000 (DESI 2024 IV Table 1, Lya region A), DR2 824,989 (z > 2.09, DR2 Lya BAO
paper Sec 2). ``LYA_FOREST_FRACTION`` is that ratio; it is survey design (a catalog
count), not a fitted number.

Accountings (``NumTracers(lya_accounting=...)``):

- ``"outside"``: Lya fibers are outside the design budget; Lya keeps the dataset's
  count and nominal error in every design.
- ``"in_budget_fixed"``: Lya's full ``observed`` count sits in the QSO class and
  moves with it, but its error is held nominal (the accounting of runs logged before
  the option existed).
- ``"qso_coupled"``: Lya's forest-only fibers sit in the QSO class; Lya's tracer
  count follows the QSO class, ``N_Lya = N_Lya,nominal * f_QSO / f_QSO,nominal``, and
  its nominal DESI covariance block is rescaled by the law ``lya_sigma_scaling`` names.
"""

LYA_ACCOUNTINGS = ("outside", "in_budget_fixed", "qso_coupled")

# Fraction of the Lya QSO row's tracer count (z > 1.77) that has a Lya forest
# (z > ~2.1): the part not already counted in the QSO bin.
LYA_FOREST_FRACTION = {
    "dr1": 531_000 / 709_565,
    "dr2": 824_989 / 1_289_874,
}


# Effective signal-to-noise per mode of the Lya forest at BAO scales, x = P_F / P_N
# (an effective n̄P): ~0.22 at z ~ 2.5, k = 0.14 h/Mpc, mu = 0.6 for DESI (Font-Ribera,
# McDonald, Mostek et al. 2014, JCAP 05, 023, arXiv:1308.4164, Sec. 4, "scaling from the
# Lya forest BAO errors gives ~0.22 for the Lya forest"). It is a forecast for the full
# survey, not a fit to a DESI measurement.
LYA_NOISE_POWER_X = 0.22


def _inverse_sqrt_n(n_ratio):
    """sigma ∝ N^-1/2: the local slope of the noise-power law where n̄P = 1."""
    return n_ratio ** -0.5


def _inverse_n(n_ratio):
    """sigma ∝ N^-1: the noise-dominated (n̄P << 1) limit of the noise-power law."""
    return n_ratio ** -1.0


def _noise_power(n_ratio, x=LYA_NOISE_POWER_X):
    """sigma ∝ 1 + P_N/P_F with P_N ∝ 1/N, normalized to 1 at nominal.

    The per-mode BAO Fisher information goes as (n̄P / (1 + n̄P))^2, so sigma ∝
    (1 + 1/(r x)) with r = N / N_nominal and x = n̄P at nominal. Its local slope
    d ln sigma / d ln N = -1 / (1 + r x) runs from -1 (sparse) toward 0 (cosmic-variance
    limited); 1/sqrt(N) is the slope at r x = 1 only.
    """
    return (1.0 + 1.0 / (n_ratio * x)) / (1.0 + 1.0 / x)


# sigma(N) / sigma(N_nominal) as a function of n_ratio = N / N_nominal.
LYA_SIGMA_SCALINGS = {
    "noise_power": _noise_power,
    "inverse_sqrt_n": _inverse_sqrt_n,
    "inverse_n": _inverse_n,
}


def resolve_lya_accounting(lya_accounting, lya_in_budget):
    """The accounting from ``lya_accounting`` or the legacy ``lya_in_budget`` flag.

    ``lya_in_budget`` is the older spelling: False -> "outside", True ->
    "in_budget_fixed". Given together, they must agree on whether Lya is in the budget.
    """
    if lya_accounting is None:
        return "in_budget_fixed" if lya_in_budget else "outside"
    if lya_accounting not in LYA_ACCOUNTINGS:
        raise ValueError(
            f"lya_accounting must be one of {LYA_ACCOUNTINGS}; got {lya_accounting!r}.")
    if lya_in_budget is not None and bool(lya_in_budget) != (lya_accounting != "outside"):
        raise ValueError(
            f"lya_in_budget={lya_in_budget!r} contradicts lya_accounting={lya_accounting!r}; "
            "set only lya_accounting.")
    return lya_accounting


def budget_tracers(desi_tracers, dataset, lya_accounting):
    """desi_tracers rows (fractions) whose fibers the design divides.

    "outside" drops Lya QSO; "in_budget_fixed" keeps its full row; "qso_coupled" keeps
    only its forest fibers (``observed``, ``targets`` and ``passed`` scaled by
    ``LYA_FOREST_FRACTION``).
    """
    if lya_accounting == "outside":
        return desi_tracers.loc[desi_tracers["tracer"] != "Lya QSO"]
    if lya_accounting == "in_budget_fixed":
        return desi_tracers
    if dataset not in LYA_FOREST_FRACTION:
        raise ValueError(
            f"lya_accounting='qso_coupled' needs the Lya forest fraction for {dataset!r}; "
            f"known: {sorted(LYA_FOREST_FRACTION)}.")
    table = desi_tracers.copy()
    lya = table["tracer"] == "Lya QSO"
    for column in ("observed", "targets", "passed"):
        table.loc[lya, column] = table.loc[lya, column] * LYA_FOREST_FRACTION[dataset]
    return table
