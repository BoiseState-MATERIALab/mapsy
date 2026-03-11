import numpy as np
import pandas as pd
import pytest

from mapsy import MultiMaps


class FakeMaps:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        self.contactspace = object()
        self.data: pd.DataFrame | None = None
        self.features: list[str] = []

    def atcontactspace(self) -> pd.DataFrame:
        self.data = self._frame.copy()
        self.features = self.data.columns.drop(["x", "y", "z"]).tolist()
        return self.data

    def graph(self, clusters: np.ndarray) -> np.ndarray:
        nclusters = int(np.max(clusters)) + 1
        return np.zeros((nclusters, nclusters), dtype=np.int64)


def test_multimaps_combines_contact_space_data() -> None:
    map_a = FakeMaps(
        pd.DataFrame(
            {
                "x": [0.0, 1.0],
                "y": [0.0, 1.0],
                "z": [0.0, 0.0],
                "f1": [0.1, 0.2],
                "f2": [1.0, 1.1],
            }
        )
    )
    map_b = FakeMaps(
        pd.DataFrame(
            {
                "x": [2.0, 3.0],
                "y": [2.0, 3.0],
                "z": [0.0, 0.0],
                "f1": [0.3, 0.4],
                "f2": [1.2, 1.3],
            }
        )
    )

    multimaps = MultiMaps([map_a, map_b], names=["a", "b"])
    combined = multimaps.atcontactspace()

    assert multimaps.features == ["f1", "f2"]
    assert combined["system"].tolist() == ["a", "a", "b", "b"]
    assert combined["map_index"].tolist() == [0, 0, 1, 1]
    assert combined["point_index"].tolist() == [0, 1, 0, 1]


def test_multimaps_requires_shared_features() -> None:
    map_a = FakeMaps(
        pd.DataFrame(
            {
                "x": [0.0],
                "y": [0.0],
                "z": [0.0],
                "f1": [0.1],
            }
        )
    )
    map_b = FakeMaps(
        pd.DataFrame(
            {
                "x": [1.0],
                "y": [1.0],
                "z": [0.0],
                "f2": [0.2],
            }
        )
    )

    with pytest.raises(ValueError, match="same feature columns"):
        MultiMaps([map_a, map_b]).atcontactspace()


def test_multimaps_reduce_propagates_pca_columns() -> None:
    map_a = FakeMaps(
        pd.DataFrame(
            {
                "x": [0.0, 0.1, 0.2],
                "y": [0.0, 0.1, 0.2],
                "z": [0.0, 0.0, 0.0],
                "f1": [0.0, 0.1, 0.2],
                "f2": [1.0, 1.1, 1.2],
            }
        )
    )
    map_b = FakeMaps(
        pd.DataFrame(
            {
                "x": [1.0, 1.1, 1.2],
                "y": [1.0, 1.1, 1.2],
                "z": [0.0, 0.0, 0.0],
                "f1": [2.0, 2.1, 2.2],
                "f2": [3.0, 3.1, 3.2],
            }
        )
    )

    multimaps = MultiMaps([map_a, map_b])
    multimaps.atcontactspace()
    multimaps.reduce(npca=2)

    assert multimaps.data is not None
    assert {"pca0", "pca1"}.issubset(multimaps.data.columns)
    assert map_a.data is not None
    assert map_b.data is not None
    assert {"pca0", "pca1"}.issubset(map_a.data.columns)
    assert {"pca0", "pca1"}.issubset(map_b.data.columns)


def test_multimaps_cluster_propagates_labels() -> None:
    map_a = FakeMaps(
        pd.DataFrame(
            {
                "x": [0.0, 0.1, 0.2],
                "y": [0.0, 0.1, 0.2],
                "z": [0.0, 0.0, 0.0],
                "f1": [0.0, 0.1, 0.2],
                "f2": [0.0, 0.1, 0.2],
            }
        )
    )
    map_b = FakeMaps(
        pd.DataFrame(
            {
                "x": [2.0, 2.1, 2.2],
                "y": [2.0, 2.1, 2.2],
                "z": [0.0, 0.0, 0.0],
                "f1": [5.0, 5.1, 5.2],
                "f2": [5.0, 5.1, 5.2],
            }
        )
    )

    multimaps = MultiMaps([map_a, map_b])
    multimaps.atcontactspace()
    multimaps.cluster(nclusters=2, random_state=0)

    assert multimaps.data is not None
    assert "Cluster" in multimaps.data.columns
    assert multimaps.nclusters == 2
    assert multimaps.cluster_graph is not None
    assert multimaps.cluster_graph.shape == (2, 2)
    assert map_a.data is not None
    assert map_b.data is not None
    assert "Cluster" in map_a.data.columns
    assert "Cluster" in map_b.data.columns
