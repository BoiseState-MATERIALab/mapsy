import numpy as np
import pandas as pd
from ase import Atoms

from mapsy import Maps, MultiMaps
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
