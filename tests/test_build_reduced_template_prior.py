from __future__ import annotations

import numpy as np
import pandas as pd

from bedcosmo.num_visits.empirical.build_reduced_template_prior import (
    parse_template_subset,
    reduced_weights_table,
    write_reduced_template_param,
)


def test_parse_template_subset():
    assert parse_template_subset("T7+T1") == (1, 7)


def test_reduced_weights_table_maps_membership_schema():
    memberships = pd.DataFrame(
        {
            "targetid": [11],
            "healpix": [23040],
            "z": [0.7],
            "chi2_dof": [1.01],
            "dof": [100],
            "templates": ["T1+T7"],
            "reduced_chi2_dof": [1.02],
            "reduced_log_c_scale": [6.2],
            "delta_chi2_dof": [0.01],
            "lsst_color_rms": [0.005],
            "template_1": [1],
            "c_1": [2.0],
            "a_1": [0.25],
            "template_2": [7],
            "c_2": [6.0],
            "a_2": [0.75],
        }
    )

    output = reduced_weights_table(
        memberships,
        (1, 7),
        n_full_templates=12,
        max_chi2_dof=1.2,
    )

    assert output.loc[0, "dof"] == 110
    assert output.loc[0, "quality_pass"]
    np.testing.assert_allclose(output.loc[0, ["c1", "c2"]].to_numpy(float), [2.0, 6.0])
    np.testing.assert_allclose(output.loc[0, ["a1", "a2"]].to_numpy(float), [0.25, 0.75])
    assert output.loc[0, "log_c_scale"] == 6.2


def test_write_reduced_template_param(tmp_path):
    path = write_reduced_template_param(
        tmp_path / "reduced.param",
        ["templates/t1.dat", "templates/t2.dat", "templates/t3.dat"],
        (1, 3),
    )

    assert path.read_text().splitlines()[1:] == [
        "1 templates/t1.dat 1.0",
        "2 templates/t3.dat 1.0",
    ]
