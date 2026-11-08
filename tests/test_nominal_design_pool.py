"""The nominal design as an ordinary member of the evaluation design pool.

Evaluation folds the nominal design into ``experiment.designs`` and records its row
as the top-level ``nominal_index`` of eig_data; readers take its EIG from the
per-design arrays. Files written before that kept a scalar in a sibling
``nominal`` block, which ``nominal_value`` still reads.
"""

import matplotlib

matplotlib.use("Agg")

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from matplotlib.collections import PolyCollection

from bedcosmo.artifacts import nominal_grid_eig, nominal_value
from bedcosmo.base import BaseExperiment, find_nominal_index
from bedcosmo.grid_calc import GridCalculation
from bedcosmo.plotting import BasePlotter
from bedcosmo.util import _merge_sibling_eig_data


class _PoolExperiment(BaseExperiment):
    """BaseExperiment with only the design-pool machinery wired up."""

    def __init__(self, nominal_design=None, design_axes=None, constraint=None, design_pts=None):
        self.device = "cpu"
        if nominal_design is not None:
            self.nominal_design = torch.tensor(nominal_design, dtype=torch.float64)
        if design_pts is not None:
            self.designs_grid = None
            self.designs = torch.tensor(design_pts, dtype=torch.float64)
        else:
            self.designs_grid = self._build_design_grid(
                design_axes=design_axes, labels=list(design_axes), constraint=constraint
            )
            self.designs = self._designs_from_grid(self.designs_grid)

    def init_designs(self, **kwargs):
        raise NotImplementedError

    def init_prior(self, parameters, **kwargs):
        raise NotImplementedError

    def pyro_model(self, design):
        raise NotImplementedError

    def sample_parameters(self, sample_shape, prior=None, **kwargs):
        raise NotImplementedError


def _budget_axes():
    # num_visits-style: three bands on 10..50 step 20, total visits fixed at 90.
    axes = {name: np.arange(10.0, 51.0, 20.0) for name in ("u", "g", "r")}

    def budget(**kwargs):
        return (np.abs(sum(kwargs.values()) - 90.0) < 1e-9).astype(int)

    return axes, budget


def test_off_tick_nominal_is_appended_and_grid_grows_by_exactly_one():
    # 37 and 16 are not ticks. Inserting them into the axes would let the budget
    # admit other new combinations (e.g. 37+16+37); the pool must grow by one.
    axes, budget = _budget_axes()
    exp = _PoolExperiment([37.0, 16.0, 37.0], design_axes=axes, constraint=budget)
    before = exp.designs.clone()

    index = exp.add_nominal_to_designs()

    assert index == len(before)
    torch.testing.assert_close(exp.designs[:index], before)
    torch.testing.assert_close(exp.designs[index], exp.nominal_design)
    grid_designs = exp._designs_from_grid(exp.designs_grid)
    assert {tuple(d) for d in grid_designs.tolist()} == {tuple(d) for d in exp.designs.tolist()}


def test_unsorted_pool_keeps_its_order():
    pts = [[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]]
    exp = _PoolExperiment([1.5, 2.5], design_pts=pts)
    exp.designs_grid = exp._build_design_grid(design_pts=exp.designs, labels=["a", "b"])

    assert exp.add_nominal_to_designs() == 3
    torch.testing.assert_close(exp.designs[:3], torch.tensor(pts, dtype=torch.float64))


def test_grid_eig_is_mapped_onto_pool_rows():
    """Grid EIG comes out in the grid's sorted cell order; eig_by_design reorders it."""
    pts = [[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]]
    exp = _PoolExperiment(None, design_pts=pts)
    grid = exp._build_design_grid(design_pts=exp.designs, labels=["a", "b"])
    calc = SimpleNamespace(experiment=exp, design_grid=grid)
    # Grid cells are sorted: (1,3), (2,2), (3,1). EIG per cell = first coordinate.
    eig_in_grid_order = exp._designs_from_grid(grid)[:, 0].numpy()

    eig = GridCalculation.eig_by_design(calc, eig_in_grid_order)

    np.testing.assert_array_equal(eig, [3.0, 1.0, 2.0])


def test_nominal_already_in_pool_is_left_alone():
    axes, budget = _budget_axes()
    exp = _PoolExperiment([10.0, 30.0, 50.0], design_axes=axes, constraint=budget)
    before = exp.designs.clone()

    index = exp.add_nominal_to_designs()

    torch.testing.assert_close(exp.designs, before)
    torch.testing.assert_close(exp.designs[index], exp.nominal_design)


def test_experiment_without_nominal_keeps_its_pool():
    axes, budget = _budget_axes()
    exp = _PoolExperiment(None, design_axes=axes, constraint=budget)
    before = exp.designs.clone()

    assert exp.add_nominal_to_designs() is None
    torch.testing.assert_close(exp.designs, before)


def test_explicit_points_without_grid_append_nominal_last():
    exp = _PoolExperiment([5.0, 6.0], design_pts=[[1.0, 2.0], [3.0, 4.0]])

    assert exp.add_nominal_to_designs() == 2
    assert exp.designs_grid is None
    torch.testing.assert_close(exp.designs[2], exp.nominal_design)


def test_find_nominal_index_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="dimensions"):
        find_nominal_index(np.zeros((3, 2)), np.zeros(3))


def test_nominal_value_reads_pool_row():
    eig_data = {"nominal_index": 1}
    block = {"eigs_avg": [1.0, 2.5, 3.0], "eigs_std": [0.1, 0.2, 0.3]}

    assert nominal_value(eig_data, block, None) == 2.5
    assert nominal_value(eig_data, block, None, "eigs_std") == 0.2
    assert nominal_value(eig_data, block, None, "prior_entropy_avg") is None


def test_nominal_value_reads_legacy_block():
    block = {"eigs_avg": [1.0, 2.5, 3.0]}

    assert nominal_value({}, block, {"eigs_avg": 4.0}) == 4.0
    assert nominal_value({}, block, {"eigs_avg": [4.0]}) == 4.0
    assert nominal_value({"nominal_index": None}, block, None) is None


def test_nominal_grid_eig_reads_grid_block_by_index():
    eig_data = {
        "nominal_index": 0,
        "step_10": {"variable": {"eigs_avg": [1.0, 2.0], "grid": {"eigs_avg": [7.0, 8.0]}}},
    }
    legacy = {"step_10": {"nominal": {"grid": {"eigs_avg": 9.0}}}}

    assert nominal_grid_eig(eig_data, "step_10") == 7.0
    assert nominal_grid_eig(legacy, "step_10") == 9.0


def test_sibling_merge_rejects_mismatched_pools():
    nf = {"input_designs": [[1.0], [2.0], [3.0]], "nominal_index": 2, "step_1": {}}
    grid = {"input_designs": [[1.0], [2.0]], "nominal_index": None, "step_1": {}}

    with pytest.raises(ValueError, match="different design pools"):
        _merge_sibling_eig_data(nf, grid, "step_1")

    merged = _merge_sibling_eig_data(nf, dict(nf), "step_1")
    assert merged["nominal_index"] == 2


@pytest.mark.parametrize("nominal_design", [[2.0, 2.0], None])
def test_eig_designs_marks_nominal_by_index(tmp_path, nominal_design):
    plotter = BasePlotter(cosmo_exp="num_visits")
    designs = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])

    fig, (ax0, _) = plotter.eig_designs(
        eig_values=np.array([1.0, 2.0, 3.0]),
        input_designs=designs,
        design_labels=["a", "b"],
        nominal_design=nominal_design,
        nominal_eig=2.0,
        nominal_index=1,
        save_dir=str(tmp_path),
        filename="eig_designs",
    )

    labels = ax0.get_legend_handles_labels()[1]
    assert "Nominal Design" in labels


@pytest.mark.parametrize("scatter", [True, False])
def test_eig_designs_scatter_draws_error_bars_instead_of_band(tmp_path, scatter):
    plotter = BasePlotter(cosmo_exp="num_visits")

    fig, (ax0, _) = plotter.eig_designs(
        eig_values=np.array([1.0, 2.0, 3.0]),
        eig_std_values=np.array([0.1, 0.2, 0.3]),
        input_designs=np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]),
        design_labels=["a", "b"],
        nominal_design=[2.0, 2.0],
        nominal_index=1,
        scatter=scatter,
        save_dir=str(tmp_path),
        filename="eig_designs",
    )

    # errorbar() adds an ErrorbarContainer; the curve style fills a std band.
    bands = [c for c in ax0.collections if isinstance(c, PolyCollection)]
    assert (len(ax0.containers) == 1) is scatter
    assert (len(bands) == 0) is scatter
