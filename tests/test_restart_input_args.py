"""Restart copies the source run's config, frozen artifacts and model args into the new run."""

from __future__ import annotations

from types import SimpleNamespace

import mlflow
import pytest

from bedcosmo.train import Trainer


@pytest.fixture
def restart_trainer(tmp_path):
    storage = tmp_path / "num_tracers"
    mlflow.set_tracking_uri(f"file:{storage}/mlruns")
    exp_id = mlflow.set_experiment("base").experiment_id
    with mlflow.start_run() as src:
        src_id = src.info.run_id
    src_artifacts = storage / "mlruns" / exp_id / src_id / "artifacts"
    (src_artifacts / "emulators" / "mean").mkdir(parents=True)
    (src_artifacts / "empirical").mkdir()
    (src_artifacts / "prior_args.yaml").write_text("parameters: {}\n")
    (src_artifacts / "design_args.yaml").write_text("labels: [a]\n")
    (src_artifacts / "emulators" / "ELG2.pt").write_bytes(b"elg2")
    (src_artifacts / "emulators" / "mean" / "LRG1.pt").write_bytes(b"lrg1")
    (src_artifacts / "empirical" / "sed_prior_kde_native.joblib").write_bytes(b"kde")

    trainer = Trainer.__new__(Trainer)
    trainer.storage_path = str(storage)
    trainer.restart_exp_id = exp_id
    trainer.restart_run_id = src_id
    trainer.run_args = {}
    trainer.verbose = False
    with mlflow.start_run() as new:
        yield trainer, storage / "mlruns" / exp_id / new.info.run_id / "artifacts"


def test_restart_copies_source_artifacts(restart_trainer):
    # Copying into the new run's artifacts and then logging that dir used to copy each
    # emulator onto itself and raise shutil.SameFileError.
    trainer, new_artifacts = restart_trainer
    trainer._save_input_args(restart_run=True)
    assert (new_artifacts / "prior_args.yaml").read_text() == "parameters: {}\n"
    assert (new_artifacts / "design_args.yaml").read_text() == "labels: [a]\n"
    assert (new_artifacts / "emulators" / "ELG2.pt").read_bytes() == b"elg2"
    assert (new_artifacts / "emulators" / "mean" / "LRG1.pt").read_bytes() == b"lrg1"
    assert (new_artifacts / "empirical" / "sed_prior_kde_native.joblib").read_bytes() == b"kde"


def test_fix_model_args_copies_architecture_from_source_run():
    # condition_design is no longer a run arg (variable_redshift never logged it), so
    # restart must not require it from the source run.
    params = {
        "cosmo_model": "base",
        "flow_type": "MAF",
        "n_transforms": "4",
        "activation": "elu",
        "cond_hidden_size": "256",
        "cond_n_layers": "4",
        "nf_transform": "affine",
    }
    trainer = Trainer.__new__(Trainer)
    trainer.run_args = {"n_transforms": 6, "total_steps": 100}
    trainer.global_rank = 0
    trainer._fix_model_args(SimpleNamespace(data=SimpleNamespace(params=params)))
    assert trainer.run_args == {**params, "total_steps": 100}
