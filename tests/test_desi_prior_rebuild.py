"""Prior-only rebuilds reuse and validate saved factors before writing."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from bedcosmo.num_visits.empirical.desi import build_prior


@pytest.fixture
def saved_build(tmp_path):
    wave = np.array([1390.0, 4000.0, 9120.0])
    basis = np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 3.0]])
    coefficients = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    targetids = np.array([39627568982265273, 39627568982265274, 39627568982265275])
    prior = tmp_path / "desi2"
    prior.mkdir()
    matrix = tmp_path / "matrix.npz"
    np.savez(
        matrix,
        wave_rest_aa=wave,
        targetid=targetids,
        healpix=[1, 1, 1],
        redshift=[0.2, 0.6, 1.29],
        flux=coefficients @ basis,
        relative_ivar=np.ones((3, 3)),
        normalization_scale=np.ones(3),
    )
    np.savez(
        prior / "desi_basis.npz",
        wave_rest_aa=wave,
        basis=basis,
        coefficients=coefficients,
        targetid=targetids,
        support_mask=np.ones(3, bool),
    )
    build_prior.write_template_bank(prior / "templates", Path("desi2.param"), wave, basis)
    (prior / "factorization_request.json").write_text('{"unchanged": true}')
    (prior / "build_provenance.json").write_text(
        json.dumps(
            {
                "template": {
                    "template_param": "desi2.param",
                    "rank": 2,
                    "normalization": {"wave_min_aa": 3600.0, "wave_max_aa": 4200.0},
                },
                "factorization": {"training_matrix": str(matrix), "method": "Nearly-NMF",
                                  "training_matrix_sha256": build_prior.training_matrix_sha256(matrix)},
                "selection": {"prior_z_min": 0.21, "prior_z_max": 1.28},
                "arguments": {"rank": 2},
            }
        )
    )
    return prior, matrix


def test_prior_only_cli_rebuilds_selection_and_kde_without_changing_factors(
    saved_build, monkeypatch
):
    prior, _ = saved_build
    protected = [
        prior / "desi_basis.npz",
        prior / "factorization_request.json",
        *sorted((prior / "templates").iterdir()),
    ]
    original = {path: path.read_bytes() for path in protected}
    calls = []
    monkeypatch.setattr("sys.argv", ["build_prior", "--prior-only", "--output-dir", str(prior)])
    with monkeypatch.context() as patch:
        patch.setattr(build_prior.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd))
        build_prior.main()
    table = pd.read_csv(prior / "desi_eazy_empirical_weights.csv")
    np.testing.assert_allclose(table.z, [0.6, 1.29])
    assert table.quality_pass.all()
    assert table.chi2_dof.max() == pytest.approx(0.0)
    assert {path: path.read_bytes() for path in protected} == original
    meta = json.loads((prior / "build_provenance.json").read_text())
    assert meta["selection"]["prior_z_max"] == pytest.approx(3199 / 1390 - 1)
    assert meta["factorization"]["method"] == "Nearly-NMF"
    assert len(calls) == 1
    assert build_prior.KDE_MODULE in calls[0]
    assert str(prior / "sed_prior_kde_native.joblib") in calls[0]
    assert (prior / "prior_args.yaml").exists()
    assert (prior / "template_redshifts.png").stat().st_size > 0


def test_prior_only_plot_uses_saved_template_param_in_custom_directory(saved_build, monkeypatch):
    prior, _ = saved_build
    destination = prior.with_name("custom-build")
    prior.rename(destination)
    monkeypatch.setattr("sys.argv", [
        "build_prior", "--prior-only", "--output-dir", str(destination), "--skip-kde"
    ])
    build_prior.main()
    assert (destination / "template_redshifts.png").stat().st_size > 0


@pytest.mark.parametrize("field", ["targetid", "flux", "relative_ivar", "redshift", "normalization_scale"])
def test_prior_only_mismatched_matrix_fails_before_writing(saved_build, monkeypatch, field):
    prior, matrix = saved_build
    with np.load(matrix) as data:
        arrays = dict(data)
    arrays[field] = arrays[field] * 2
    np.savez(matrix, **arrays)
    provenance = (prior / "build_provenance.json").read_bytes()
    monkeypatch.setattr("sys.argv", ["build_prior", "--prior-only", "--output-dir", str(prior)])
    with pytest.raises(ValueError, match="fingerprint does not match"):
        build_prior.main()
    assert (prior / "build_provenance.json").read_bytes() == provenance


def test_checkpoint_reuse_rejects_changed_matrix_fingerprint(tmp_path):
    args = build_prior.argparse.Namespace(training_matrix_sha256="original")
    wave = np.array([1390., 4000., 9120.])
    build_prior.require_compatible_checkpoints(tmp_path, args, wave, 100)
    args.training_matrix_sha256 = "changed"
    with pytest.raises(ValueError, match="request"):
        build_prior.require_compatible_checkpoints(tmp_path, args, wave, 100)


def test_prior_only_resolves_saved_settings_without_changing_request(saved_build, monkeypatch):
    prior, _ = saved_build
    monkeypatch.setattr("sys.argv", ["build_prior", "--prior-only", "--output-dir", str(prior),
                                    "--skip-kde", "--prior-z-min", "0.5"])
    args = build_prior.parse_args()
    requested = vars(args).copy()
    build_prior.rebuild_prior(args, prior)
    config = yaml.safe_load((prior / "prior_args.yaml").read_text())
    assert config["template_param"] == "desi2.param"
    assert config["template_norm_min"] == 3600.
    assert config["template_norm_max"] == 4200.
    assert set(config["parameters"]) == {"f1", "log_c_scale", "z"}
    selection = json.loads((prior / "build_provenance.json").read_text())["selection"]
    assert selection["prior_z_min"] == .5
    assert selection["prior_z_max"] == pytest.approx(3199 / 1390 - 1)
    assert vars(args) == requested


def test_prior_only_inactive_component_still_builds_kde(saved_build, monkeypatch):
    prior, _ = saved_build
    with np.load(prior / "desi_basis.npz") as data:
        arrays = dict(data)
    arrays["coefficients"][:, 1] = 0
    arrays["coefficients"][:, 0] = 1
    np.savez(prior / "desi_basis.npz", **arrays)
    calls = []
    monkeypatch.setattr("sys.argv", ["build_prior", "--prior-only", "--output-dir", str(prior),
                                    "--max-chi2-dof", "100"])
    with monkeypatch.context() as patch:
        patch.setattr(build_prior.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd))
        build_prior.main()
    assert (prior / "template_redshifts.png").stat().st_size > 0
    assert len(calls) == 1
    assert build_prior.KDE_MODULE in calls[0]


def test_short_build_name_resolves_under_empirical_prior(saved_build, monkeypatch, tmp_path):
    prior, _ = saved_build
    destination = tmp_path / "bedcosmo/num_visits/empirical_prior/desi2"
    destination.parent.mkdir(parents=True)
    prior.rename(destination)
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["build_prior", "--prior-only", "--build-name", "desi2", "--skip-kde"])
    build_prior.main()
    table = pd.read_csv(destination / "desi_eazy_empirical_weights.csv")
    np.testing.assert_allclose(table.z, [.6, 1.29])
    assert (destination / "prior_args.yaml").exists()
    assert (destination / "template_redshifts.png").stat().st_size > 0


@pytest.mark.parametrize("name", ["", ".", "..", "/tmp/build", "nested/build"])
def test_build_name_requires_a_single_directory_name(monkeypatch, name):
    monkeypatch.setattr("sys.argv", ["build_prior", "--build-name", name])
    with pytest.raises(ValueError, match="single directory name"):
        build_prior.main()
