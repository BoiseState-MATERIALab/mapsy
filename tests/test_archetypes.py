import matplotlib
import numpy as np
import pandas as pd
from ase import Atoms

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from mapsy import Maps, MultiMaps
from mapsy.data import Grid, System


class StubContactSpace:
    def __init__(
        self,
        positions: np.ndarray,
        probabilities: np.ndarray,
        neighbors: list[np.ndarray],
        core_distance: np.ndarray | None = None,
        layer: np.ndarray | None = None,
    ) -> None:
        self.nm = len(positions)
        self.neighbors = [np.asarray(row, dtype=np.int64) for row in neighbors]
        self.data = pd.DataFrame(positions, columns=["x", "y", "z"])
        self.data.loc[:, "probability"] = probabilities
        self.data.loc[:, "region"] = np.zeros(self.nm, dtype=np.int64)
        self.data.loc[:, "core_distance"] = (
            np.asarray(core_distance, dtype=np.float64)
            if core_distance is not None
            else np.ones(self.nm, dtype=np.float64)
        )
        self.data.loc[:, "layer"] = (
            np.asarray(layer, dtype=np.int64)
            if layer is not None
            else np.zeros(self.nm, dtype=np.int64)
        )
        self._feature_columns: list[str] = []

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)


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


def _build_maps() -> Maps:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    probabilities = np.array([0.1, 1.0, 1.0, 1.0, 0.1])
    neighbors = [
        np.array([1, -1, -1, -1, -1, -1]),
        np.array([0, 2, -1, -1, -1, -1]),
        np.array([1, 3, -1, -1, -1, -1]),
        np.array([2, -1, -1, -1, -1, -1]),
        np.full(6, -1, dtype=np.int64),
    ]
    contactspace = StubContactSpace(positions, probabilities, neighbors)
    maps = Maps(_build_system(), [], contactspace)
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "f1": [0.0, 10.0, 20.0, 30.0, 40.0],
        }
    )
    maps.features = ["f1"]
    return maps


def test_maps_select_archetypes_prefers_high_probability_extremes() -> None:
    maps = _build_maps()

    result = maps.select_archetypes(
        2,
        feature_columns=["f1"],
        min_probability_quantile=0.5,
    )

    assert set(result.selected_indexes.tolist()) == {1, 3}
    assert result.archetype_table["selection_rank"].tolist() == [0, 1]

    special = maps.get_special_points(kind="archetype")
    assert set(special["point_index"].tolist()) == {1, 3}


def test_maps_select_archetypes_can_filter_candidates_by_layer() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    probabilities = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float64)
    neighbors = [
        np.array([1, -1, -1, -1, -1, -1]),
        np.array([0, 2, -1, -1, -1, -1]),
        np.array([1, 3, -1, -1, -1, -1]),
        np.array([2, -1, -1, -1, -1, -1]),
    ]
    layer = np.array([0, 0, 1, 1], dtype=np.int64)
    maps = Maps(
        _build_system(),
        [],
        StubContactSpace(
            positions,
            probabilities,
            neighbors,
            np.array([0.6, 0.4, 0.2, 0.1], dtype=np.float64),
            layer,
        ),
    )
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "f1": [0.0, 1.0, 10.0, 11.0],
            "layer": layer,
        }
    )
    maps.features = ["f1"]

    result = maps.select_archetypes(1, feature_columns=["f1"], layer=1)

    assert result.candidate_indexes.tolist() == [2, 3]
    assert result.selected_indexes.tolist() == [3]
    assert result.metadata is not None
    assert result.metadata["min_probability_quantile"] is None


def test_maps_reduce_can_fit_pca_on_layer() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    probabilities = np.ones(len(positions), dtype=np.float64)
    neighbors = [np.full(6, -1, dtype=np.int64) for _ in range(len(positions))]
    layer = np.array([0, 0, 1, 1], dtype=np.int64)
    maps = Maps(
        _build_system(),
        [],
        StubContactSpace(
            positions,
            probabilities,
            neighbors,
            np.array([0.9, 0.1, 0.2, 0.8], dtype=np.float64),
            layer,
        ),
    )
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "f1": [100.0, 0.0, 2.0, 200.0],
            "f2": [50.0, 1.0, 1.0, 60.0],
            "layer": layer,
        }
    )
    maps.features = ["f1", "f2"]

    result = maps.reduce(npca=1, layer=0)

    assert result.npca == 1
    assert maps.pca_analysis_result is not None
    np.testing.assert_allclose(
        maps.pca_analysis_result.estimator.mean_,
        np.array([50.0, 25.5], dtype=np.float64),
    )
    assert "pca0" in maps.data.columns


def test_maps_cluster_can_fit_on_layer() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    probabilities = np.ones(len(positions), dtype=np.float64)
    neighbors = [
        np.array([1, -1, -1, -1, -1, -1]),
        np.array([0, 2, -1, -1, -1, -1]),
        np.array([1, 3, -1, -1, -1, -1]),
        np.array([2, -1, -1, -1, -1, -1]),
    ]
    layer = np.array([0, 0, 1, 1], dtype=np.int64)
    maps = Maps(
        _build_system(),
        [],
        StubContactSpace(
            positions,
            probabilities,
            neighbors,
            np.array([0.8, 0.1, 0.2, 0.7], dtype=np.float64),
            layer,
        ),
    )
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "f1": [100.0, 0.0, 10.0, 200.0],
            "layer": layer,
        }
    )
    maps.features = ["f1"]

    result = maps.cluster(nclusters=2, method="kmeans", random_state=0, layer=1)

    assert result.metadata is not None
    assert result.metadata["layer"] == [1]
    np.testing.assert_array_equal(
        maps.data["Cluster"].to_numpy(dtype=np.int64), np.array([-1, -1, 0, 1])
    )


def test_maps_cluster_can_propagate_layer_labels() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    probabilities = np.ones(len(positions), dtype=np.float64)
    neighbors = [
        np.array([1, -1, -1, -1, -1, -1]),
        np.array([0, 2, -1, -1, -1, -1]),
        np.array([1, 3, -1, -1, -1, -1]),
        np.array([2, -1, -1, -1, -1, -1]),
    ]
    layer = np.array([0, 1, 1, 0], dtype=np.int64)
    maps = Maps(
        _build_system(),
        [],
        StubContactSpace(
            positions,
            probabilities,
            neighbors,
            np.array([0.8, 0.1, 0.2, 0.7], dtype=np.float64),
            layer,
        ),
    )
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "f1": [100.0, 0.0, 10.0, 200.0],
            "layer": layer,
        }
    )
    maps.features = ["f1"]
    maps.build_graph(mode="realspace", feature_columns=["f1"])

    result = maps.cluster(
        nclusters=2,
        method="kmeans",
        random_state=0,
        layer=1,
        propagate=True,
        propagation_mode="region_grow",
    )

    labels = maps.data["Cluster"].to_numpy(dtype=np.int64)
    assert np.all(labels >= 0)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[3]
    assert result.metadata is not None
    assert result.metadata["propagate"] is True
    assert "cluster_confidence" in maps.data.columns


def test_maps_sites_can_select_one_site_per_cluster_and_layer() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    probabilities = np.ones(len(positions), dtype=np.float64)
    neighbors = [np.full(6, -1, dtype=np.int64) for _ in range(len(positions))]
    layer = np.array([0, 1, 0, 1], dtype=np.int64)
    maps = Maps(
        _build_system(),
        [],
        StubContactSpace(
            positions,
            probabilities,
            neighbors,
            np.ones(len(positions), dtype=np.float64),
            layer,
        ),
    )
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "f1": [0.0, 0.5, 10.0, 9.5],
            "Cluster": [0, 0, 1, 1],
            "layer": layer,
        }
    )
    maps.features = ["f1"]
    maps.cluster_features = ["f1"]
    maps.cluster_centers = np.array([[0.0], [10.0]], dtype=np.float64)
    maps.cluster_graph = np.zeros((2, 2), dtype=np.int64)
    maps.cluster_edges = np.zeros((2, 2), dtype=np.int64)
    maps.cluster_sizes = np.array([2, 2], dtype=np.int64)

    maps.sites(region=0, per_layer=True)

    special = maps.get_special_points(kind="centroid")
    assert special["point_index"].tolist() == [0, 1, 2, 3]
    assert special["layer"].tolist() == [0, 1, 0, 1]
    assert special["cluster"].tolist() == [0, 0, 1, 1]


def test_maps_scatter_can_plot_contactspace_core_mask_categorically() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    probabilities = np.ones(len(positions), dtype=np.float64)
    neighbors = [
        np.array([1, -1, -1, -1, -1, -1]),
        np.array([0, 2, -1, -1, -1, -1]),
        np.array([1, 3, -1, -1, -1, -1]),
        np.array([2, -1, -1, -1, -1, -1]),
    ]
    maps = Maps(
        _build_system(),
        [],
        StubContactSpace(
            positions,
            probabilities,
            neighbors,
            np.array([0.6, 0.4, 0.2, 0.1], dtype=np.float64),
        ),
    )
    maps.contactspace.data.loc[:, "is_core"] = np.array([False, False, True, True], dtype=bool)
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "f1": [0.0, 1.0, 2.0, 3.0],
        }
    )
    maps.features = ["f1"]

    fig, ax = maps.scatter(feature="is_core", categorical=True, region=None)

    legend = ax.get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["is_core = 0", "is_core = 1"]
    plt.close(fig)


def test_maps_scatter_core_projection_prefers_smallest_core_distance() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    probabilities = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    neighbors = [np.full(6, -1, dtype=np.int64) for _ in range(len(positions))]
    maps = Maps(
        _build_system(),
        [],
        StubContactSpace(
            positions,
            probabilities,
            neighbors,
            np.array([0.4, 0.1, 0.2], dtype=np.float64),
        ),
    )
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "f1": [10.0, 20.0, 30.0],
        }
    )
    maps.features = ["f1"]

    fig, ax = maps.scatter_core_projection(feature="f1", plane=("x", "y"), region=None)

    offsets = ax.collections[0].get_offsets()
    values = np.asarray(ax.collections[0].get_array(), dtype=np.float64)
    assert offsets.shape[0] == 2
    np.testing.assert_allclose(values, np.array([20.0, 30.0], dtype=np.float64))
    plt.close(fig)


def test_maps_scatter_core_projection_can_use_distance_column() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    probabilities = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    neighbors = [np.full(6, -1, dtype=np.int64) for _ in range(len(positions))]
    maps = Maps(
        _build_system(),
        [],
        StubContactSpace(
            positions,
            probabilities,
            neighbors,
            np.array([0.1, 0.4, 0.2], dtype=np.float64),
        ),
    )
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "f1": [10.0, 20.0, 30.0],
            "interface_center_distance": [0.4, 0.1, 0.2],
        }
    )
    maps.features = ["f1"]

    fig, ax = maps.scatter_core_projection(
        feature="f1",
        plane=("x", "y"),
        selector="center",
        distance_column="interface_center_distance",
        region=None,
    )

    offsets = ax.collections[0].get_offsets()
    values = np.asarray(ax.collections[0].get_array(), dtype=np.float64)
    assert offsets.shape[0] == 2
    np.testing.assert_allclose(values, np.array([20.0, 30.0], dtype=np.float64))
    plt.close(fig)


def test_maps_scatter_core_projection_can_weighted_average_duplicates() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    probabilities = np.array([1.0, 3.0, 1.0], dtype=np.float64)
    neighbors = [np.full(6, -1, dtype=np.int64) for _ in range(len(positions))]
    maps = Maps(
        _build_system(),
        [],
        StubContactSpace(
            positions,
            probabilities,
            neighbors,
            np.array([0.4, 0.1, 0.2], dtype=np.float64),
        ),
    )
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "f1": [10.0, 20.0, 30.0],
        }
    )
    maps.features = ["f1"]

    fig, ax = maps.scatter_core_projection(
        feature="f1",
        plane=("x", "y"),
        selector="weighted_mean",
        region=None,
    )

    values = np.sort(np.asarray(ax.collections[0].get_array(), dtype=np.float64))
    np.testing.assert_allclose(values, np.array([17.5, 30.0], dtype=np.float64))
    plt.close(fig)


def test_maps_scatter_core_projection_can_filter_to_layer() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    probabilities = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    neighbors = [np.full(6, -1, dtype=np.int64) for _ in range(len(positions))]
    maps = Maps(
        _build_system(),
        [],
        StubContactSpace(
            positions,
            probabilities,
            neighbors,
            np.array([0.4, 0.1, 0.2], dtype=np.float64),
            np.array([1, 0, 1], dtype=np.int64),
        ),
    )
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "f1": [10.0, 20.0, 30.0],
        }
    )
    maps.features = ["f1"]

    fig, ax = maps.scatter_core_projection(feature="f1", plane=("x", "y"), region=None, layer=1)

    offsets = ax.collections[0].get_offsets()
    values = np.sort(np.asarray(ax.collections[0].get_array(), dtype=np.float64))
    assert offsets.shape[0] == 2
    np.testing.assert_allclose(values, np.array([10.0, 30.0], dtype=np.float64))
    plt.close(fig)


def test_maps_min_projection_selects_lowest_energy_along_normal() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    probabilities = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    neighbors = [np.full(6, -1, dtype=np.int64) for _ in range(len(positions))]
    maps = Maps(
        _build_system(),
        [],
        StubContactSpace(positions, probabilities, neighbors),
    )
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "energy": [-0.1, -0.3, -0.2],
        }
    )
    maps.features = ["energy"]

    projected = maps.min_projection(feature="energy", plane=("x", "y"), region=None)

    assert projected.shape[0] == 2
    selected = projected.sort_values(["x", "y"]).reset_index(drop=True)
    np.testing.assert_allclose(selected["energy"], np.array([-0.3, -0.2]))
    np.testing.assert_allclose(selected["z"], np.array([1.0, 0.0]))
    assert selected["multiplicity"].tolist() == [2, 1]


def test_maps_scatter_min_projection_returns_selected_energy_projection() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    probabilities = np.ones(len(positions), dtype=np.float64)
    neighbors = [np.full(6, -1, dtype=np.int64) for _ in range(len(positions))]
    maps = Maps(
        _build_system(),
        [],
        StubContactSpace(positions, probabilities, neighbors),
    )
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "energy": [0.4, 0.1, 0.2],
        }
    )
    maps.features = ["energy"]

    fig, ax, projected = maps.scatter_min_projection(
        feature="energy",
        plane=("x", "y"),
        region=None,
        return_projection=True,
    )

    offsets = ax.collections[0].get_offsets()
    values = np.sort(np.asarray(ax.collections[0].get_array(), dtype=np.float64))
    assert offsets.shape[0] == 2
    np.testing.assert_allclose(values, np.array([0.1, 0.2], dtype=np.float64))
    assert projected["multiplicity"].tolist() == [2, 1]
    plt.close(fig)


def test_maps_propagate_archetypes_marks_ambiguous_regions() -> None:
    maps = _build_maps()
    selection = maps.select_archetypes(
        2,
        feature_columns=["f1"],
        min_probability_quantile=0.5,
    )
    graph = maps.build_graph(mode="realspace")

    result = maps.propagate_archetypes(
        graph=graph,
        selected_indexes=selection.selected_indexes,
        confidence_threshold=0.51,
        margin_threshold=0.01,
    )

    assignment = result.assignment_table.set_index("point_index")
    assert int(assignment.loc[1, "assigned_archetype_index"]) == 1
    assert int(assignment.loc[3, "assigned_archetype_index"]) == 3
    assert bool(assignment.loc[2, "is_ambiguous"])
    assert bool(assignment.loc[4, "is_ambiguous"])
    assert int(assignment.loc[4, "assigned_archetype_index"]) == -1

    assert bool(maps.data.loc[2, "is_ambiguous"])
    assert int(maps.data.loc[1, "assigned_archetype_index"]) == 1


def test_maps_propagate_archetypes_shortest_path_gives_compact_groups() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    probabilities = np.ones(len(positions), dtype=np.float64)
    neighbors = [
        np.array([1, -1, -1, -1, -1, -1]),
        np.array([0, 2, -1, -1, -1, -1]),
        np.array([1, 3, -1, -1, -1, -1]),
        np.array([2, 4, -1, -1, -1, -1]),
        np.array([3, -1, -1, -1, -1, -1]),
    ]
    maps = Maps(_build_system(), [], StubContactSpace(positions, probabilities, neighbors))
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "f1": positions[:, 0],
        }
    )
    maps.features = ["f1"]
    graph = maps.build_graph(mode="realspace")

    result = maps.propagate_archetypes(
        graph=graph,
        selected_indexes=np.array([0, 4]),
        propagation_mode="shortest_path",
        confidence_threshold=0.0,
        margin_threshold=0.0,
    )

    assigned = result.assignment_table.set_index("point_index")["assigned_archetype_index"]
    assert assigned.loc[0] == 0
    assert assigned.loc[1] == 0
    assert assigned.loc[3] == 4
    assert assigned.loc[4] == 4


def test_maps_propagate_archetypes_region_grow_splits_by_topology() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    probabilities = np.ones(len(positions), dtype=np.float64)
    neighbors = [
        np.array([1, -1, -1, -1, -1, -1]),
        np.array([0, 2, -1, -1, -1, -1]),
        np.array([1, 3, -1, -1, -1, -1]),
        np.array([2, 4, -1, -1, -1, -1]),
        np.array([3, -1, -1, -1, -1, -1]),
    ]
    maps = Maps(_build_system(), [], StubContactSpace(positions, probabilities, neighbors))
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "f1": positions[:, 0],
        }
    )
    maps.features = ["f1"]
    graph = maps.build_graph(mode="realspace")

    result = maps.propagate_archetypes(
        graph=graph,
        selected_indexes=np.array([0, 4]),
        propagation_mode="region_grow",
        confidence_threshold=0.0,
        margin_threshold=0.0,
    )

    assigned = result.assignment_table.set_index("point_index")["assigned_archetype_index"]
    assert assigned.loc[0] == 0
    assert assigned.loc[1] == 0
    assert assigned.loc[3] == 4
    assert assigned.loc[4] == 4


def test_maps_propagate_archetypes_watershed_prefers_connected_regions() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    probabilities = np.array([1.0, 0.5, 1.0, 1.0], dtype=np.float64)
    neighbors = [
        np.array([1, -1, -1, -1, -1, -1]),
        np.array([0, 2, -1, -1, -1, -1]),
        np.array([1, 3, -1, -1, -1, -1]),
        np.array([2, -1, -1, -1, -1, -1]),
    ]
    maps = Maps(_build_system(), [], StubContactSpace(positions, probabilities, neighbors))
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "f1": positions[:, 0],
        }
    )
    maps.features = ["f1"]
    graph = maps.build_graph(mode="realspace")

    result = maps.propagate_archetypes(
        graph=graph,
        selected_indexes=np.array([0, 3]),
        propagation_mode="watershed",
        confidence_threshold=0.0,
        margin_threshold=0.0,
    )

    assigned = result.assignment_table.set_index("point_index")["assigned_archetype_index"]
    assert assigned.loc[0] == 0
    assert assigned.loc[1] == 0
    assert assigned.loc[2] == 3
    assert assigned.loc[3] == 3


def test_graph_endpoint_selection_prefers_bent_tail_tips() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
        ]
    )
    probabilities = np.ones(len(positions), dtype=np.float64)
    neighbors = [
        np.array([1, 4, -1, -1, -1, -1]),
        np.array([0, 2, -1, -1, -1, -1]),
        np.array([1, 3, -1, -1, -1, -1]),
        np.array([2, -1, -1, -1, -1, -1]),
        np.array([0, 5, -1, -1, -1, -1]),
        np.array([4, -1, -1, -1, -1, -1]),
    ]
    contactspace = StubContactSpace(positions, probabilities, neighbors)
    maps = Maps(_build_system(), [], contactspace)
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "pc0": positions[:, 0],
            "pc1": positions[:, 1],
        }
    )
    maps.features = ["pc0", "pc1"]
    graph = maps.build_graph(mode="realspace")

    result = maps.select_archetypes(
        2,
        feature_columns=["pc0", "pc1"],
        graph=graph,
        selection_mode="graph_endpoint",
        probability_weight=0.0,
        extremeness_weight=1.0,
        diversity_weight=0.25,
        min_probability_quantile=None,
    )

    assert set(result.selected_indexes.tolist()) == {3, 5}
    assert "endpoint_score" in result.candidate_table.columns


def test_multimaps_archetype_wrappers_register_and_propagate() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    neighbors = [
        np.array([1, -1, -1, -1, -1, -1]),
        np.array([0, -1, -1, -1, -1, -1]),
    ]

    map_a = Maps(
        _build_system(),
        [],
        StubContactSpace(positions, np.array([1.0, 0.2]), neighbors),
    )
    map_a.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "f1": [0.0, 0.1],
        }
    )
    map_a.features = ["f1"]

    map_b = Maps(
        _build_system(),
        [],
        StubContactSpace(positions + np.array([10.0, 0.0, 0.0]), np.array([1.0, 0.2]), neighbors),
    )
    map_b.data = pd.DataFrame(
        {
            "x": positions[:, 0] + 10.0,
            "y": positions[:, 1],
            "z": positions[:, 2],
            "f1": [5.0, 5.1],
        }
    )
    map_b.features = ["f1"]

    multimaps = MultiMaps([map_a, map_b], names=["a", "b"])

    selection = multimaps.select_archetypes(
        2,
        feature_columns=["f1"],
        min_probability_quantile=0.5,
    )
    assert len(selection.selected_indexes) == 2

    graph = multimaps.build_graph(
        mode="feature",
        feature_columns=["f1"],
        feature_k=1,
        connect_systems=True,
    )
    result = multimaps.propagate_archetypes(
        graph=graph,
        selected_indexes=selection.selected_indexes,
        confidence_threshold=0.4,
    )

    assert "assigned_archetype_index" in multimaps.data.columns
    assert "assigned_archetype_index" in map_a.data.columns
    assert "assigned_archetype_index" in map_b.data.columns
    assert np.all(result.assigned_archetype_indexes[result.assigned_archetype_indexes >= 0] >= 0)

    special_a = map_a.get_special_points(kind="archetype")
    special_b = map_b.get_special_points(kind="archetype")
    assert len(special_a) == 1
    assert len(special_b) == 1
