import numpy as np
import pandas as pd
from ase import Atoms
from matplotlib import pyplot as plt

from mapsy import Maps, compare_stationary_points
from mapsy.data import Grid, System


def _build_maps(frame: pd.DataFrame) -> Maps:
    cell = np.diag([10.0, 10.0, 10.0])
    grid = Grid(scalars=[2, 2, 2], cell=cell)
    atoms = Atoms("H", positions=[[5.0, 5.0, 5.0]], cell=cell, pbc=True)
    maps = Maps(System(grid=grid, atoms=atoms), [])
    maps.data = frame.copy()
    maps.features = [column for column in frame.columns if column not in {"x", "y", "z"}]
    return maps


def test_compare_stationary_points_weights_low_energy_reference_points() -> None:
    reference = pd.DataFrame(
        {
            "type": ["minimum", "saddle_1", "minimum", "maximum"],
            "x": [0.0, 1.0, 4.0, 2.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "value": [0.0, 0.05, 0.8, 1.0],
        }
    )
    predicted = pd.DataFrame(
        {
            "type": ["minimum", "saddle_1", "maximum"],
            "x": [0.1, 1.2, 2.0],
            "y": [0.0, 0.0, 0.0],
            "value": [0.02, 0.0, 1.1],
        }
    )

    summary, matches = compare_stationary_points(
        reference,
        predicted,
        xy_scale=0.5,
        energy_scale=0.05,
        energy_weight_scale=0.10,
        max_xy_error=0.4,
        max_energy_error=0.10,
    )

    assert summary["n_reference"] == 3
    assert summary["n_predicted"] == 2
    assert summary["n_matched"] == 2
    assert summary["n_missed"] == 1
    assert summary["weighted_recall"] > 0.99
    assert matches.loc[matches["status"] == "missed", "type"].tolist() == ["minimum"]


def test_compare_stationary_points_uses_periodic_coordinate_distance() -> None:
    reference = pd.DataFrame(
        {
            "type": ["minimum"],
            "x": [0.1],
            "y": [2.0],
            "value": [0.0],
        }
    )
    predicted = pd.DataFrame(
        {
            "type": ["minimum"],
            "x": [9.9],
            "y": [2.0],
            "value": [0.01],
        }
    )

    summary, matches = compare_stationary_points(
        reference,
        predicted,
        periods=(10.0, None),
        xy_scale=0.5,
        energy_scale=0.05,
        max_xy_error=0.5,
        max_energy_error=0.1,
    )

    np.testing.assert_allclose(summary["weighted_recall"], 1.0)
    assert bool(matches.iloc[0]["matched"])
    np.testing.assert_allclose(matches.iloc[0]["dxy"], 0.2)
    np.testing.assert_allclose(matches.iloc[0]["delta_x"], 0.2)


def test_compare_stationary_points_accepts_single_type_string() -> None:
    reference = pd.DataFrame(
        {
            "type": ["minimum", "saddle_1"],
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "value": [0.0, 0.2],
        }
    )
    predicted = pd.DataFrame(
        {
            "type": ["minimum", "saddle_1"],
            "x": [0.1, 1.0],
            "y": [0.0, 0.0],
            "value": [0.0, 0.2],
        }
    )

    summary, matches = compare_stationary_points(reference, predicted, types="minimum")

    assert summary["types"] == ("minimum",)
    assert summary["n_reference"] == 1
    assert matches["type"].tolist() == ["minimum"]


def test_maps_compare_stationary_points_2d_compares_two_features() -> None:
    nx = 8
    ny = 8
    x_axis = np.arange(nx, dtype=np.float64)
    y_axis = np.arange(ny, dtype=np.float64)
    X, Y = np.meshgrid(x_axis, y_axis)
    values = 2.0 - np.cos(2.0 * np.pi * X / nx) - np.cos(2.0 * np.pi * Y / ny)
    frame = pd.DataFrame(
        {
            "x": X.reshape(-1),
            "y": Y.reshape(-1),
            "z": np.zeros(nx * ny, dtype=np.float64),
            "reference_energy": values.reshape(-1),
            "gp_energy": values.reshape(-1),
        }
    )
    maps = _build_maps(frame)

    summary, matches, reference_stationary, predicted_stationary = (
        maps.compare_stationary_points_2d(
            reference_feature="reference_energy",
            predicted_feature="gp_energy",
            region=None,
            grad_tol=1.0e-10,
            curvature_tol=1.0e-6,
            return_stationary=True,
        )
    )

    assert summary["reference_feature"] == "reference_energy"
    assert summary["predicted_feature"] == "gp_energy"
    assert summary["n_reference"] == 3
    assert summary["n_matched"] == 3
    np.testing.assert_allclose(summary["weighted_recall"], 1.0)
    assert np.allclose(matches.loc[matches["matched"], "dxy"], 0.0)
    assert set(reference_stationary["type"]) == set(predicted_stationary["type"])


def test_maps_plot_stationary_point_comparison_2d_labels_matches() -> None:
    nx = 8
    ny = 8
    x_axis = np.arange(nx, dtype=np.float64)
    y_axis = np.arange(ny, dtype=np.float64)
    X, Y = np.meshgrid(x_axis, y_axis)
    values = 2.0 - np.cos(2.0 * np.pi * X / nx) - np.cos(2.0 * np.pi * Y / ny)
    frame = pd.DataFrame(
        {
            "x": X.reshape(-1),
            "y": Y.reshape(-1),
            "z": np.zeros(nx * ny, dtype=np.float64),
            "reference_energy": values.reshape(-1),
            "gp_energy": values.reshape(-1),
        }
    )
    maps = _build_maps(frame)

    fig, axs, summary, matches = maps.plot_stationary_point_comparison_2d(
        reference_feature="reference_energy",
        predicted_feature="gp_energy",
        region=None,
        grad_tol=1.0e-10,
        curvature_tol=1.0e-6,
        colorbar=False,
        legend=False,
    )

    assert len(axs) == 2
    assert axs[0].get_title() == "Reference: reference_energy"
    assert axs[1].get_title() == "Predicted: gp_energy"
    assert summary["n_matched"] == 3
    assert matches["plot_label"].tolist() == ["1", "2", "3"]
    assert [text.get_text() for text in axs[0].texts] == ["1", "2", "3"]
    assert [text.get_text() for text in axs[1].texts] == ["1", "2", "3"]
    plt.close(fig)


def test_maps_plot_stationary_point_comparison_2d_closes_periodic_cell() -> None:
    nx = 8
    ny = 8
    x_axis = np.arange(nx, dtype=np.float64)
    y_axis = np.arange(ny, dtype=np.float64)
    X, Y = np.meshgrid(x_axis, y_axis)
    values = 2.0 - np.cos(2.0 * np.pi * (X - 7.8) / nx) - np.cos(2.0 * np.pi * (Y - 7.7) / ny)
    frame = pd.DataFrame(
        {
            "x": X.reshape(-1),
            "y": Y.reshape(-1),
            "z": np.zeros(nx * ny, dtype=np.float64),
            "reference_energy": values.reshape(-1),
            "gp_energy": values.reshape(-1),
        }
    )
    maps = _build_maps(frame)

    fig, axs, summary, matches = maps.plot_stationary_point_comparison_2d(
        reference_feature="reference_energy",
        predicted_feature="gp_energy",
        region=None,
        types=("minimum",),
        colorbar=False,
        legend=False,
    )

    assert summary["n_matched"] == 1
    assert matches["ref_x"].iloc[0] > x_axis[-1]
    assert matches["ref_y"].iloc[0] > y_axis[-1]
    np.testing.assert_allclose(axs[0].get_xlim(), (0.0, float(nx)))
    np.testing.assert_allclose(axs[0].get_ylim(), (0.0, float(ny)))
    np.testing.assert_allclose(axs[1].get_xlim(), (0.0, float(nx)))
    np.testing.assert_allclose(axs[1].get_ylim(), (0.0, float(ny)))
    plt.close(fig)
