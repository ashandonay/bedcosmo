"""NumTracers._report_emulator_coverage: trained N box against the design pool.

Runs on a stub instance with a hand-built passed map; no DESI data, prior or
trained checkpoint needed.
"""
import warnings

import pandas as pd
import pytest
import torch

from bedcosmo.num_tracers.experiment import NumTracers

BUDGET = 1_000_000


@pytest.fixture
def stub():
    exp = NumTracers.__new__(NumTracers)
    exp.device = torch.device("cpu")
    exp.global_rank = 0
    exp.design_labels = ["BGS", "LRG", "ELG", "QSO"]
    exp.nominal_total_obs = BUDGET
    exp.desi_data = pd.DataFrame({"tracer": ["BGS", "LRG3+ELG1", "LRG3+ELG1", "QSO"]})
    # BGS fed by BGS alone; the mixed bin by LRG and ELG.
    exp._passed_weights = torch.tensor([
        [0.9, 0.0, 0.0, 0.0],
        [0.0, 0.3, 0.2, 0.0],
        [0.0, 0.3, 0.2, 0.0],
        [0.0, 0.0, 0.0, 0.7],
    ], dtype=torch.float64)
    exp._passed_offset = torch.zeros(4, dtype=torch.float64)
    exp.designs = torch.tensor([
        [0.1, 0.3, 0.3, 0.3],
        [0.9, 0.0, 0.05, 0.05],
    ], dtype=torch.float64)
    return exp


def test_reports_budget_fraction_and_warns_above_the_box(stub, capsys):
    # BGS trained to N = 0.45 * BUDGET, i.e. BGS f <= 0.5; the second design asks 0.81e6.
    stub._emulators = {"BGS": {"n_train_hi": 0.45 * BUDGET},
                       "LRG3_ELG1": {"n_train_hi": 0.5 * BUDGET}}
    with pytest.warns(RuntimeWarning, match=r"BGS \(1/2 designs\)"):
        stub._report_emulator_coverage()
    out = capsys.readouterr().out
    assert "BGS f <= 0.5 of the budget" in out
    assert "designs reach 8.1e+05 (f = 0.9)" in out
    # Mixed bin: counts only, and inside its box.
    assert "LRG3_ELG1  trained N <= 5e+05; designs reach 1.5e+05" in out
    assert "BGS        trained" in out


def test_silent_when_the_box_covers_the_pool(stub, capsys):
    stub._emulators = {"BGS": {"n_train_hi": 0.9 * BUDGET}, "QSO": {"n_train_hi": BUDGET}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stub._report_emulator_coverage()
    assert "designs above" not in capsys.readouterr().out


def test_shapefit_networks_are_checked(stub, capsys):
    stub._emulators = {}
    stub.shapefit_bins = ["BGS", "QSO"]
    stub._shapefit_n_weights = torch.tensor(
        [[0.9, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.7]], dtype=torch.float64) * BUDGET
    stub._shapefit_n_offset = torch.zeros(2, dtype=torch.float64)
    box = {"n_train_hi": 0.45 * BUDGET}
    stub._shapefit_emulators = {"mean": {"BGS": box, "QSO": box},
                                "covar": {"BGS": box, "QSO": box}}
    with pytest.warns(RuntimeWarning, match=r"BGS mean \(1/2 designs\), BGS covar"):
        stub._report_emulator_coverage()
    out = capsys.readouterr().out
    assert "QSO covar  trained N <= 4.5e+05 (QSO f <= 0.643 of the budget)" in out


def test_no_emulators_is_a_no_op(stub, capsys):
    stub._emulators = {}
    stub._report_emulator_coverage()
    assert capsys.readouterr().out == ""


def _pool(tmp_path, accounting):
    import json
    import numpy as np
    np.save(tmp_path / "designs.npy", np.array([[0.25, 0.25, 0.25, 0.25]]))
    prov = {"args": {} if accounting is None else {"lya_accounting": accounting}}
    (tmp_path / "provenance.json").write_text(json.dumps(prov))
    return str(tmp_path / "designs.npy")


def test_pool_built_for_another_accounting_is_refused(stub, tmp_path):
    stub.lya_accounting = "qso_coupled"
    with pytest.raises(ValueError, match="built for lya_accounting='outside'"):
        stub._check_pool_accounting(_pool(tmp_path, "outside"))


@pytest.mark.parametrize("accounting", ["qso_coupled", None])
def test_matching_or_unrecorded_pool_is_accepted(stub, tmp_path, accounting):
    stub.lya_accounting = "qso_coupled"
    stub._check_pool_accounting(_pool(tmp_path, accounting))
