import numpy as np
import pytest
import xarray as xr

from osdagbridge.core.bridge_types.plate_girder.plot_generator import _transverse_member_values
from osdagbridge.core.bridge_types.plate_girder.results_data import _classify_member_tags


def _dataset():
    force_components = [
        "Vx_i", "Vx_j", "Vy_i", "Vy_j", "Vz_i", "Vz_j",
        "Mx_i", "Mx_j", "My_i", "My_j", "Mz_i", "Mz_j",
    ]
    displacement_components = ["x", "y", "z"]
    components = force_components + displacement_components
    forces = np.zeros((1, len(components)))
    displacements = np.zeros((2, len(components)))
    for index in range(0, len(force_components), 2):
        forces[0, index:index + 2] = [1000 * (index + 1), 1000 * (index + 2)]
    for component_index in range(len(force_components), len(components)):
        displacements[:, component_index] = [0.001, 0.002]
    return xr.Dataset(
        {
            "forces": (("Element", "Component"), forces),
            "displacements": (("Node", "Component"), displacements),
        },
        coords={
            "Element": [101],
            "Node": [1, 2],
            "Component": components,
        },
    )


def test_classifies_force_result_tags_by_lateral_connectivity():
    nodes = {
        1: [0.0, 0.0, 0.0],
        2: [10.0, 0.25, 0.0],
        3: [0.0, 0.0, 5.0],
        4: [0.0, 0.0, 10.0],
    }
    members = {
        10: [1, 2],
        20: [3, 4],
        30: [1, 3],
    }

    longitudinal, transverse = _classify_member_tags(
        nodes,
        members,
        element_tags=[10, 20, 99],
    )

    assert longitudinal == [10]
    assert transverse == [20]


@pytest.mark.parametrize(
    ("component_key", "expected_min", "expected_max"),
    [
        ("Fx", 1.0, 2.0),
        ("Fy", 3.0, 4.0),
        ("Fz", 5.0, 6.0),
        ("Mx", 7.0, 8.0),
        ("My", 9.0, 10.0),
        ("Mz", 11.0, 12.0),
        ("Dx", 1.0, 2.0),
        ("Dy", -2.0, -1.0),
        ("Dz", -2.0, -1.0),
    ],
)
def test_transverse_plot_extracts_selected_component(
    component_key,
    expected_min,
    expected_max,
):
    nodes = {
        1: [0.0, 0.0, 0.0],
        2: [0.0, 0.0, 5.0],
    }
    members = {101: [1, 2]}

    values_by_member = _transverse_member_values(
        _dataset(),
        component_key,
        nodes,
        members,
        [101],
    )
    values = [value for _, _, _, member_values in values_by_member for value in member_values]

    assert min(values) == pytest.approx(expected_min)
    assert max(values) == pytest.approx(expected_max)
