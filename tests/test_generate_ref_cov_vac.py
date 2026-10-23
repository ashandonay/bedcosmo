"""generate_ref_cov_vac places the VAC BAO dilation covariances on the desi_data rows."""

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "generate_ref_cov_vac",
    Path(__file__).resolve().parent.parent / "scripts" / "num_tracers" / "generate_ref_cov_vac.py",
)
gen = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gen)


def _write_vac(path, names, cov):
    with h5py.File(path, "w") as h:
        h["covariance/value"] = np.asarray(cov, dtype=np.float64)
        h["covariance/observable/baorecon/labels_values"] = np.array([n.encode() for n in names])


@pytest.fixture
def layout(tmp_path):
    # desi_data rows: BGS DV, LRG1 DM, LRG1 DH, Lya DM, Lya DH (D_M before D_H, as in DESI).
    data = pd.DataFrame({
        "tracer": ["BGS", "LRG1", "LRG1", "Lya QSO", "Lya QSO"],
        "quantity": ["DV_over_rs", "DM_over_rs", "DH_over_rs", "DM_over_rs", "DH_over_rs"],
        "value_at_z": [8.0, 13.0, 21.0, 39.0, 8.5],
    })
    published = np.diag([0.15, 0.25, 0.6, 0.94, 0.17]) ** 2
    published[3, 4] = published[4, 3] = -0.05
    _write_vac(tmp_path / f"likelihood_bao-recon_syst_{gen.VAC_TRACER['BGS']}.h5", ["qiso"], [[0.02 ** 2]])
    # The VAC orders the anisotropic block (qpar, qper) = (D_H, D_M).
    _write_vac(tmp_path / f"likelihood_bao-recon_syst_{gen.VAC_TRACER['LRG1']}.h5", ["qpar", "qper"],
               [[0.03 ** 2, -0.0002], [-0.0002, 0.018 ** 2]])
    return data, published, tmp_path


def test_fractional_covariance_on_desi_rows(layout):
    data, published, vac_dir = layout
    cov = gen.build_ref_cov(data, published, str(vac_dir), "syst")
    sig = np.sqrt(np.diag(cov))
    np.testing.assert_allclose(sig[:3], [0.02 * 8.0, 0.018 * 13.0, 0.03 * 21.0])  # qper->DM, qpar->DH
    np.testing.assert_allclose(cov[1, 2], -0.0002 * 13.0 * 21.0)
    np.testing.assert_array_equal(cov[3:, 3:], published[3:, 3:])  # Lya keeps the published block
    assert cov[0, 1] == cov[0, 3] == cov[1, 3] == 0  # no cross-tracer terms


def test_missing_parameter_raises(layout, tmp_path):
    data, published, vac_dir = layout
    _write_vac(vac_dir / f"likelihood_bao-recon_syst_{gen.VAC_TRACER['LRG1']}.h5", ["qiso"], [[1e-4]])
    with pytest.raises(ValueError, match="do not match rows"):
        gen.build_ref_cov(data, published, str(vac_dir), "syst")
