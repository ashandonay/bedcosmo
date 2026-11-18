"""How the Lya QSO bin enters the num_tracers design.

Tracer QSOs (the QSO bin, 0.8 < z < 2.1) and Lya QSOs come from one QSO target
selection: which side of z = 2.1 a target lands on is known only after its spectrum.
A design can therefore move the Lya sample only together with the QSO class.

The ``Lya QSO`` row of desi_data.csv / desi_tracers.csv carries DESI's Lya x QSO
*tracer* count: quasar positions at z > 1.77 (DR1 709,565, DESI 2024 IV Table 1;
DR2 1,289,874, DESI DR2 Lya BAO paper Sec 2). That sample overlaps the QSO bin over
1.77 < z < 2.1, so its ``observed`` column double counts those fibers. Only the
quasars beyond the QSO bin (z > 2.1) are fibers it does not already hold:
DR1 448,995 (the DR1 Lya quasar catalog, VAC zlya v1.0, whose z > 1.77 count
reproduces Table 1's 709,565), DR2 824,989 (z > 2.09, DR2 Lya BAO paper Sec 2).
DESI 2024 IV Table 1's 531,000 "Lya region A" is NOT this count: it is the z > 1.988
quasars (every quasar whose region A reaches the 3600 A limit), which includes the
1.99 < z < 2.1 quasars already in the QSO bin. ``LYA_FOREST_FRACTION`` is that ratio;
it is survey design (a catalog count), not a fitted number.

Accountings (``NumTracers(lya_accounting=...)``):

- ``"outside"``: Lya fibers are outside the design budget; Lya keeps the dataset's
  count and nominal error in every design.
- ``"in_budget_fixed"``: Lya's full ``observed`` count sits in the QSO class and
  moves with it, but its error is held nominal (the accounting of runs logged before
  the option existed).
- ``"qso_coupled"``: Lya's forest-only fibers sit in the QSO class; Lya's tracer
  count follows the QSO class, ``N_Lya = N_Lya,nominal * f_QSO / f_QSO,nominal``. In
  emulator mode its block comes from the Lya table (tracer-forecast-emulator's
  first-principles Lya forecast); in scaling mode its nominal DESI block is rescaled
  as sigma ∝ 1/sqrt(N), like every other row.
"""

LYA_ACCOUNTINGS = ("outside", "in_budget_fixed", "qso_coupled")

# Fraction of the Lya QSO row's tracer count (z > 1.77) beyond the QSO bin
# (z > 2.1): the part not already counted in it.
LYA_FOREST_FRACTION = {
    "dr1": 448_995 / 709_565,
    "dr2": 824_989 / 1_289_874,
}


def resolve_lya_accounting(lya_accounting, lya_in_budget):
    """The accounting from ``lya_accounting`` or the legacy ``lya_in_budget`` flag.

    ``lya_in_budget`` is the older spelling: False -> "outside", True ->
    "in_budget_fixed". Given together, they must agree on whether Lya is in the budget.
    With neither, the accounting is "qso_coupled".
    """
    if lya_accounting is None:
        if lya_in_budget is None:
            return "qso_coupled"
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
