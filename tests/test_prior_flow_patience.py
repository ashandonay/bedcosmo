"""Early stopping in ``train_prior_flow``: ``patience`` ends training once the eval NLL
stops improving, and the flow it returns is still the best-eval checkpoint."""

import numpy as np
import pytest

from bedcosmo.prior_flow import SPACE_NATIVE, train_prior_flow


def _fit(**kwargs):
    # Few samples and a wide net, so the eval NLL turns over within a few dozen epochs.
    x = np.random.default_rng(0).normal(size=(200, 2))
    return train_prior_flow(
        x,
        space=SPACE_NATIVE,
        hidden_features=(64, 64),
        transforms=2,
        bins=6,
        batch_size=50,
        lr=1e-2,
        seed=0,
        verbose=False,
        **kwargs,
    )


def test_patience_stops_after_best_epoch():
    pf = _fit(epochs=300, patience=5)
    train = pf.meta["train"]
    history = train["eval_nll_history"]
    assert train["stopped_epoch"] is not None
    assert len(history) == train["stopped_epoch"] + 1 < 300
    assert train["stopped_epoch"] - train["best_epoch"] == 5
    assert train["best_eval_nll"] == min(history) == history[train["best_epoch"]]


def test_no_patience_runs_every_epoch():
    train = _fit(epochs=12).meta["train"]
    assert train["stopped_epoch"] is None
    assert len(train["eval_nll_history"]) == 12
    assert train["best_eval_nll"] == min(train["eval_nll_history"])


def test_patience_must_be_positive():
    with pytest.raises(ValueError, match="patience"):
        _fit(epochs=5, patience=0)
