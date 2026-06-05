import numpy as np
import pandas as pd
from ase import Atoms
from matplotlib import pyplot as plt

from mapsy import Maps
from mapsy.data import Grid, System


def _build_maps(frame: pd.DataFrame) -> Maps:
    cell = np.diag([10.0, 10.0, 10.0])
    grid = Grid(scalars=[2, 2, 2], cell=cell)
    atoms = Atoms("H", positions=[[5.0, 5.0, 5.0]], cell=cell, pbc=True)
    maps = Maps(System(grid=grid, atoms=atoms), [])
    maps.data = frame.copy()
    maps.features = [column for column in frame.columns if column not in {"x", "y", "z"}]
    return maps


def test_select_points_prefers_feature_diversity_from_centroids() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 0.0, 0.0, 0.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 1.0, 2.0, 3.0],
            "f1": [0.0, 10.0, 9.0, 0.0],
            "f2": [0.0, 0.0, 0.0, 10.0],
            "predicted_label": [0.0, 0.0, 0.0, 0.0],
            "uncertainty": [0.0, 0.0, 0.0, 0.0],
        }
    )
    maps = _build_maps(frame)
    maps.add_special_points([0], kind="centroid", iteration=0, label_status="unlabeled")

    selected = maps.select_points(
        npoints=2,
        feature_columns=["f1", "f2"],
        real_space_weight=0.0,
        feature_space_weight=1.0,
        energy_column="predicted_label",
        uncertainty_column="uncertainty",
        energy_weight=0.0,
        uncertainty_weight=0.0,
    )

    assert set(selected.index.tolist()) == {1, 3}
    assert selected.iloc[1].name != 2
    assert "selection_score" in selected.columns
    assert "feature_space_score" in selected.columns


def test_centroids_property_is_backed_by_special_points_registry() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
        }
    )
    maps = _build_maps(frame)

    maps.centroids = np.array([1], dtype=np.int64)

    np.testing.assert_array_equal(maps.centroids, np.array([1], dtype=np.int64))
    special = maps.get_special_points(kind="centroid")
    assert special["point_index"].tolist() == [1]
    assert special["label_status"].tolist() == ["unlabeled"]


def test_add_special_points_preserves_metadata_and_joins_map_data() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "predicted_label": [-0.2, -0.1],
            "uncertainty": [0.7, 0.2],
        }
    )
    maps = _build_maps(frame)

    maps.add_special_points(
        [0],
        kind="adaptive",
        iteration=2,
        label_status="completed",
        observed_label=[-0.2],
        selection_score=[0.85],
    )

    special = maps.get_special_points(kind="adaptive")
    assert special["point_index"].tolist() == [0]
    assert special["iteration"].tolist() == [2]
    assert special["label_status"].tolist() == ["completed"]
    assert special["observed_label"].tolist() == [-0.2]
    assert special["selection_score"].tolist() == [0.85]
    assert special["predicted_label"].tolist() == [-0.2]


def test_select_points_balances_energy_and_uncertainty() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
            "predicted_label": [0.0, -0.4, -0.1],
            "uncertainty": [0.1, 0.2, 0.9],
        }
    )
    maps = _build_maps(frame)

    selected = maps.select_points(
        npoints=1,
        feature_columns=[],
        energy_column="predicted_label",
        uncertainty_column="uncertainty",
        real_space_weight=0.0,
        feature_space_weight=0.0,
        energy_weight=2.0,
        uncertainty_weight=1.0,
    )

    assert selected.index.tolist() == [1]
    assert float(selected.iloc[0]["energy_score"]) > float(selected.iloc[0]["uncertainty_score"])


def test_select_special_points_registers_selection_for_workflow_use() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
            "predicted_label": [0.0, -0.4, -0.1],
            "uncertainty": [0.1, 0.2, 0.9],
        }
    )
    maps = _build_maps(frame)

    selected = maps.select_special_points(
        npoints=1,
        kind="adaptive",
        iteration=3,
        feature_columns=[],
        energy_column="predicted_label",
        uncertainty_column="uncertainty",
        real_space_weight=0.0,
        feature_space_weight=0.0,
        energy_weight=2.0,
        uncertainty_weight=1.0,
    )

    assert selected["point_index"].tolist() == [1]
    assert selected["kind"].tolist() == ["adaptive"]
    assert selected["iteration"].tolist() == [3]
    assert selected["label_status"].tolist() == ["unlabeled"]
    assert "selection_score" in selected.columns
    assert "energy_score" in selected.columns

    registry = maps.get_special_points(kind="adaptive")
    assert registry["point_index"].tolist() == [1]


def test_select_special_points_can_use_pivoted_qr_and_layer_filter() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0],
            "layer": [0, 1, 1, 1],
            "f1": [0.0, 1.0, 0.0, 0.9],
            "f2": [0.0, 0.0, 1.0, 0.1],
        }
    )
    maps = _build_maps(frame)

    selected = maps.select_special_points(
        npoints=2,
        kind="site",
        feature_columns=["f1", "f2"],
        method="pivoted_qr",
        layer=1,
        scale_features=False,
    )

    assert selected["point_index"].tolist() == [1, 2]
    assert selected["selection_method"].tolist() == ["pivoted_qr", "pivoted_qr"]
    assert "pivot_score" in selected.columns


def test_select_points_can_use_pivoted_cholesky_rbf() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
            "f1": [0.0, 1.0, 10.0],
        }
    )
    maps = _build_maps(frame)

    selected = maps.select_points(
        npoints=2,
        feature_columns=["f1"],
        method="pivoted_cholesky",
        gamma=0.1,
        scale_features=False,
    )

    assert selected.index.tolist() == [0, 2]
    assert selected["selection_method"].tolist() == [
        "pivoted_cholesky",
        "pivoted_cholesky",
    ]
    assert selected["kernel_gamma"].tolist() == [0.1, 0.1]


def test_select_points_can_target_local_minima_instead_of_global_minimum() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
            "predicted_label": [-1.0, -0.5, -0.2],
            "gx": [0.2, 0.0, 0.0],
            "gy": [0.0, 0.0, 0.0],
            "gz": [0.0, 0.0, 0.0],
            "k1": [1.0, 2.0, -1.0],
            "k2": [1.0, 3.0, 4.0],
        }
    )
    maps = _build_maps(frame)

    selected = maps.select_points(
        npoints=1,
        feature_columns=[],
        energy_column="predicted_label",
        real_space_weight=0.0,
        feature_space_weight=0.0,
        energy_weight=1.0,
        uncertainty_weight=0.0,
        energy_selection_mode="stationary",
        gradient_columns=["gx", "gy", "gz"],
        curvature_columns=["k1", "k2"],
        stationary_orders=0,
        gradient_tolerance=1.0e-8,
        curvature_tolerance=1.0e-8,
    )

    assert selected.index.tolist() == [1]


def test_select_points_can_target_minima_and_first_order_transition_states() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
            "predicted_label": [-1.0, -0.5, -0.4],
            "grad_norm": [0.2, 0.0, 0.0],
            "k1": [1.0, 2.0, -1.0],
            "k2": [1.0, 3.0, 4.0],
        }
    )
    maps = _build_maps(frame)

    selected = maps.select_points(
        npoints=2,
        feature_columns=[],
        energy_column="predicted_label",
        real_space_weight=0.0,
        feature_space_weight=0.0,
        energy_weight=1.0,
        uncertainty_weight=0.0,
        energy_selection_mode="stationary",
        gradient_norm_column="grad_norm",
        curvature_columns=["k1", "k2"],
        stationary_orders=[0, 1],
        gradient_tolerance=1.0e-8,
        curvature_tolerance=1.0e-8,
    )

    assert set(selected.index.tolist()) == {1, 2}


def test_maps_analyze_stationary_points_2d_refines_periodic_surface() -> None:
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
            "energy": values.reshape(-1),
        }
    )
    maps = _build_maps(frame)

    stationary, derivatives = maps.analyze_stationary_points_2d(
        feature="energy",
        region=None,
        grad_tol=1.0e-10,
        curvature_tol=1.0e-6,
    )

    assert derivatives["candidate_mask"].shape == (ny, nx)
    assert stationary["type"].value_counts().to_dict() == {
        "minimum": 1,
        "maximum": 1,
        "saddle_1": 2,
    }
    minimum = stationary.loc[stationary["type"] == "minimum"].iloc[0]
    maximum = stationary.loc[stationary["type"] == "maximum"].iloc[0]
    saddles = stationary.loc[stationary["type"] == "saddle_1"].sort_values(["x", "y"])

    np.testing.assert_allclose([minimum["x"], minimum["y"]], [0.0, 0.0], atol=1.0e-8)
    np.testing.assert_allclose([maximum["x"], maximum["y"]], [4.0, 4.0], atol=1.0e-8)
    np.testing.assert_allclose(
        saddles[["x", "y"]].to_numpy(dtype=np.float64),
        np.array([[0.0, 4.0], [4.0, 0.0]], dtype=np.float64),
        atol=1.0e-8,
    )


def test_maps_analyze_stationary_points_2d_uses_value_extrema_candidates() -> None:
    nx = 7
    ny = 7
    x_axis = np.arange(nx, dtype=np.float64)
    y_axis = np.arange(ny, dtype=np.float64)
    X, Y = np.meshgrid(x_axis, y_axis)
    values = -((X - 3.3) ** 2 + (Y - 2.7) ** 2)
    frame = pd.DataFrame(
        {
            "x": X.reshape(-1),
            "y": Y.reshape(-1),
            "z": np.zeros(nx * ny, dtype=np.float64),
            "energy": values.reshape(-1),
        }
    )
    maps = _build_maps(frame)

    gradient_only, _ = maps.analyze_stationary_points_2d(
        feature="energy",
        region=None,
        grad_tol=0.0,
        curvature_tol=1.0e-8,
        periodic_x=False,
        periodic_y=False,
        include_value_extrema=False,
    )
    stationary, derivatives = maps.analyze_stationary_points_2d(
        feature="energy",
        region=None,
        grad_tol=0.0,
        curvature_tol=1.0e-8,
        periodic_x=False,
        periodic_y=False,
    )

    assert gradient_only.empty
    assert np.count_nonzero(derivatives["value_max_candidate_mask"]) == 1
    maximum = stationary.loc[stationary["type"] == "maximum"].iloc[0]
    np.testing.assert_allclose([maximum["x"], maximum["y"]], [3.3, 2.7], atol=1.0e-8)


def test_maps_plot_stationary_points_2d_overlays_classified_points() -> None:
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
            "energy": values.reshape(-1),
        }
    )
    maps = _build_maps(frame)

    fig, ax, stationary = maps.plot_stationary_points_2d(
        feature="energy",
        region=None,
        grad_tol=1.0e-10,
        curvature_tol=1.0e-6,
        levels=8,
        colorbar=False,
    )

    assert ax.get_title() == "Stationary points of energy"
    assert ax.get_xlabel() == "x (Å)"
    assert ax.get_ylabel() == "y (Å)"
    assert stationary["type"].value_counts().to_dict() == {
        "minimum": 1,
        "maximum": 1,
        "saddle_1": 2,
    }
    legend = ax.get_legend()
    assert legend is not None
    assert {text.get_text() for text in legend.get_texts()} == {
        "minimum",
        "maximum",
        "saddle_1",
    }
    plt.close(fig)
