import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from scipy.sparse.csgraph import connected_components

from mapsy import GraphResult, Maps, MultiMaps
from mapsy.data import Grid, System


class StubContactSpace:
    def __init__(
        self,
        positions: np.ndarray,
        probabilities: np.ndarray,
        neighbors: list[np.ndarray],
    ) -> None:
        self.nm = len(positions)
        self.neighbors = [np.asarray(row, dtype=np.int64) for row in neighbors]
        self.data = pd.DataFrame(positions, columns=["x", "y", "z"])
        self.data.loc[:, "probability"] = probabilities
        self.data.loc[:, "region"] = np.zeros(self.nm, dtype=np.int64)
        self._feature_columns: list[str] = []

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)


class FakeMaps:
    def __init__(
        self,
        frame: pd.DataFrame,
        probabilities: np.ndarray,
        neighbors: list[np.ndarray],
    ) -> None:
        self._frame = frame
        self.contactspace = StubContactSpace(
            frame.loc[:, ["x", "y", "z"]].to_numpy(dtype=np.float64),
            probabilities=np.asarray(probabilities, dtype=np.float64),
            neighbors=neighbors,
        )
        self.data: pd.DataFrame | None = None
        self.features: list[str] = []

    def atcontactspace(self) -> pd.DataFrame:
        self.data = self._frame.copy()
        self.features = self.data.columns.drop(["x", "y", "z"]).tolist()
        return self.data


def _build_system(cell_length: float = 10.0) -> System:
    cell = np.diag([cell_length, cell_length, cell_length])
    grid = Grid(scalars=[8, 8, 8], cell=cell)
    atoms = Atoms(
        "H",
        positions=[[cell_length / 2, cell_length / 2, cell_length / 2]],
        cell=cell,
        pbc=True,
    )
    return System(grid=grid, atoms=atoms)


def test_maps_build_graph_realspace_uses_probability_node_weights() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    probabilities = np.array([2.0, 1.0, 2.0])
    neighbors = [
        np.array([1, -1, -1, -1, -1, -1]),
        np.array([0, 2, -1, -1, -1, -1]),
        np.array([1, -1, -1, -1, -1, -1]),
    ]

    system = _build_system()
    contactspace = StubContactSpace(positions, probabilities, neighbors)
    maps = Maps(system, [], contactspace)
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "pca0": [0.0, 0.1, 0.2],
        }
    )
    maps.features = ["pca0"]

    result = maps.build_graph(mode="realspace", realspace_weight=2.0)

    assert isinstance(result, GraphResult)
    np.testing.assert_allclose(result.node_weights, np.array([1.0, 0.5, 1.0]))
    assert result.matrix.shape == (3, 3)
    assert np.isclose(result.matrix[0, 1], np.sqrt(0.5) * 2.0)
    assert np.isclose(result.matrix[1, 2], np.sqrt(0.5) * 2.0)
    assert float(result.matrix[0, 2]) == 0.0


def test_maps_build_graph_feature_connects_close_points() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    probabilities = np.ones(3)
    neighbors = [np.full(6, -1, dtype=np.int64) for _ in range(3)]

    system = _build_system()
    contactspace = StubContactSpace(positions, probabilities, neighbors)
    maps = Maps(system, [], contactspace)
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "pca0": [0.0, 0.1, 10.0],
        }
    )
    maps.features = ["pca0"]

    result = maps.build_graph(mode="feature", feature_columns=["pca0"], feature_k=1)

    assert isinstance(result, GraphResult)
    edge_lookup = {
        (int(row.source), int(row.target)): float(row.weight)
        for row in result.edge_table.itertuples(index=False)
    }
    assert edge_lookup[(0, 1)] > 0.0
    assert edge_lookup[(0, 1)] > edge_lookup.get((1, 2), 0.0)


def test_maps_build_graph_feature_knn_mst_adds_connectivity_backbone() -> None:
    positions = np.column_stack([np.arange(6, dtype=np.float64), np.zeros(6), np.zeros(6)])
    probabilities = np.ones(6, dtype=np.float64)
    neighbors = [np.full(6, -1, dtype=np.int64) for _ in range(6)]

    system = _build_system()
    contactspace = StubContactSpace(positions, probabilities, neighbors)
    maps = Maps(system, [], contactspace)
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "pca0": [0.0, 0.1, 5.0, 5.1, 10.0, 10.1],
        }
    )
    maps.features = ["pca0"]

    knn_graph = maps.build_graph(
        mode="feature",
        feature_columns=["pca0"],
        feature_k=1,
        feature_connectivity="knn",
    )
    mst_graph = maps.build_graph(
        mode="feature",
        feature_columns=["pca0"],
        feature_k=1,
        feature_connectivity="knn_mst",
    )

    n_knn_components, _ = connected_components(knn_graph.matrix, directed=False)
    n_mst_components, _ = connected_components(mst_graph.matrix, directed=False)

    assert n_knn_components > 1
    assert n_mst_components == 1


def test_maps_build_graph_directional_weight_prefers_boundary_normal_direction() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    probabilities = np.ones(3)
    neighbors = [
        np.array([1, 2, -1, -1, -1, -1]),
        np.array([0, -1, -1, -1, -1, -1]),
        np.array([0, -1, -1, -1, -1, -1]),
    ]

    system = _build_system()
    contactspace = StubContactSpace(positions, probabilities, neighbors)
    contactspace.data.loc[:, "boundary_gradient_x"] = np.array([1.0, 1.0, 1.0])
    contactspace.data.loc[:, "boundary_gradient_y"] = np.array([0.0, 0.0, 0.0])
    contactspace.data.loc[:, "boundary_gradient_z"] = np.array([0.0, 0.0, 0.0])

    maps = Maps(system, [], contactspace)
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "pca0": [0.0, 0.1, 0.2],
        }
    )
    maps.features = ["pca0"]

    result = maps.build_graph(
        mode="realspace",
        realspace_weight=1.0,
        directional_weight=1.0,
    )

    edge_lookup = {
        (int(row.source), int(row.target)): float(row.weight)
        for row in result.edge_table.itertuples(index=False)
    }
    assert edge_lookup[(0, 1)] > edge_lookup[(0, 2)]


def test_multimaps_build_graph_can_toggle_cross_system_feature_edges() -> None:
    frame_a = pd.DataFrame(
        {
            "x": [0.0, 0.2],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "f1": [0.0, 1.0],
        }
    )
    frame_b = pd.DataFrame(
        {
            "x": [2.0, 2.2],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "f1": [0.05, 1.05],
        }
    )
    neighbors = [np.full(6, -1, dtype=np.int64) for _ in range(2)]

    map_a = FakeMaps(frame_a, probabilities=np.array([1.0, 1.0]), neighbors=neighbors)
    map_b = FakeMaps(frame_b, probabilities=np.array([1.0, 1.0]), neighbors=neighbors)
    multimaps = MultiMaps([map_a, map_b], names=["a", "b"])
    multimaps.atcontactspace()

    local_result = multimaps.build_graph(
        mode="feature",
        feature_columns=["f1"],
        feature_k=1,
        connect_systems=False,
    )
    cross_result = multimaps.build_graph(
        mode="feature",
        feature_columns=["f1"],
        feature_k=1,
        connect_systems=True,
    )

    def has_cross_system_edges(result: GraphResult) -> bool:
        node_table = result.node_table
        for row in result.edge_table.itertuples(index=False):
            if (
                node_table.iloc[int(row.source)]["map_index"]
                != node_table.iloc[int(row.target)]["map_index"]
            ):
                return True
        return False

    assert not has_cross_system_edges(local_result)
    assert has_cross_system_edges(cross_result)


def test_maps_cluster_can_use_graph_affinity_with_spectral() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.2, 0.0, 0.0],
        ]
    )
    probabilities = np.ones(4)
    neighbors = [
        np.array([1, -1, -1, -1, -1, -1]),
        np.array([0, -1, -1, -1, -1, -1]),
        np.array([3, -1, -1, -1, -1, -1]),
        np.array([2, -1, -1, -1, -1, -1]),
    ]

    system = _build_system()
    contactspace = StubContactSpace(positions, probabilities, neighbors)
    maps = Maps(system, [], contactspace)
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "pca0": np.zeros(4),
        }
    )
    maps.features = ["pca0"]

    graph = maps.build_graph(mode="realspace")
    result = maps.cluster(nclusters=2, method="spectral", graph=graph)

    assert result.labels[0] == result.labels[1]
    assert result.labels[2] == result.labels[3]
    assert result.labels[0] != result.labels[2]


def test_maps_cluster_rejects_graph_for_kmeans() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    probabilities = np.ones(2)
    neighbors = [np.array([1, -1, -1, -1, -1, -1]), np.array([0, -1, -1, -1, -1, -1])]

    system = _build_system()
    contactspace = StubContactSpace(positions, probabilities, neighbors)
    maps = Maps(system, [], contactspace)
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "pca0": [0.0, 1.0],
        }
    )
    maps.features = ["pca0"]

    graph = maps.build_graph(mode="realspace")
    with pytest.raises(ValueError, match="kmeans clustering does not accept a graph_matrix input"):
        maps.cluster(nclusters=2, method="kmeans", graph=graph)
