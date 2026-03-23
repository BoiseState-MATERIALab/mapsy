import numpy as np
import pandas as pd
from ase import Atoms

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
