from types import SimpleNamespace

import numpy as np
import pandas as pd
from ase import Atoms

from mapsy import Maps
from mapsy.boundaries import Boundary, ContactSpace
from mapsy.boundaries.ionic import IonicGeometry
from mapsy.data import GradientField, Grid, ScalarField, System
from mapsy.io.parser import ContactSpaceGenerator


class StubContactSpace:
    def __init__(self, positions: np.ndarray) -> None:
        self.data = pd.DataFrame(positions, columns=["x", "y", "z"])
        self._feature_columns: list[str] = []

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    def annotate(
        self,
        name: str,
        values: np.ndarray,
        *,
        as_feature: bool = True,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        self.data.loc[:, name] = array
        if as_feature and name not in self._feature_columns:
            self._feature_columns.append(name)
        elif not as_feature and name in self._feature_columns:
            self._feature_columns.remove(name)
        return array


class DummyBoundary(Boundary):
    def update(self) -> None:
        return

    def _build(self) -> None:
        return

    def _build_solvent_aware_boundary(self) -> None:
        return


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


def test_maps_atcontactspace_includes_contactspace_feature_annotations() -> None:
    system = _build_system()
    csmodel = SimpleNamespace(
        mode="system",
        cutoff=2,
        threshold=-1.0,
        side=1,
        spread=0.5,
        distance=1.0,
    )
    contactspace = ContactSpaceGenerator(csmodel).generate(system)

    maps = Maps(system, [], contactspace)
    annotation = np.linspace(0.0, 1.0, contactspace.nm)
    maps.annotate_contactspace("ionic_distance", annotation)
    maps.atpoints = lambda positions: pd.DataFrame(positions, columns=["x", "y", "z"])

    data = maps.atcontactspace()

    assert "ionic_distance" in data.columns
    assert "region" in data.columns
    assert "core_distance" in data.columns
    assert maps.features == ["ionic_distance"]
    assert "region" not in maps.features
    assert "core_distance" not in maps.features
    np.testing.assert_allclose(data["ionic_distance"].to_numpy(), annotation)


def test_contactspace_data_includes_boundary_derived_columns() -> None:
    system = _build_system()
    csmodel = SimpleNamespace(
        mode="system",
        cutoff=2,
        threshold=-1.0,
        side=1,
        spread=0.5,
        distance=1.0,
    )
    contactspace = ContactSpaceGenerator(csmodel).generate(system)

    expected = {
        "boundary_switch",
        "boundary_gradient_x",
        "boundary_gradient_y",
        "boundary_gradient_z",
    }

    assert expected.issubset(contactspace.data.columns)
    assert expected.issubset(contactspace.annotation_columns)
    assert set(contactspace.feature_columns).isdisjoint(expected)


def test_contactspace_core_distance_tracks_distance_from_interface() -> None:
    grid = Grid(scalars=[2, 2, 1], cell=np.diag([2.0, 2.0, 1.0]))
    boundary = DummyBoundary(mode="system", grid=grid)

    switch = ScalarField(grid)
    switch[:] = np.array(
        [
            [[0.25], [0.50]],
            [[0.75], [0.50]],
        ]
    )
    boundary.switch[:] = switch

    gradient = GradientField(grid)
    gradient[0, :, :, :] = 1.0
    gradient[1, :, :, :] = 0.0
    gradient[2, :, :, :] = 0.0
    boundary.gradient[:] = gradient

    contactspace = ContactSpace(
        boundary,
        tol=0.1,
        core_epsilon=1.0e-12,
    )

    assert "core_distance" in contactspace.data.columns
    assert "core_distance" in contactspace.annotation_columns
    assert "core_distance" not in contactspace.feature_columns
    np.testing.assert_allclose(
        np.sort(contactspace.data["core_distance"].to_numpy()),
        np.array([0.0, 0.0, 0.25, 0.25]),
    )


def test_contactspace_core_tolerance_defines_is_core_annotation() -> None:
    grid = Grid(scalars=[2, 2, 1], cell=np.diag([2.0, 2.0, 1.0]))
    boundary = DummyBoundary(mode="system", grid=grid)

    switch = ScalarField(grid)
    switch[:] = np.array(
        [
            [[0.40], [0.50]],
            [[0.70], [0.50]],
        ]
    )
    boundary.switch[:] = switch

    gradient = GradientField(grid)
    gradient[0, :, :, :] = 1.0
    gradient[1, :, :, :] = 0.0
    gradient[2, :, :, :] = 0.0
    boundary.gradient[:] = gradient

    contactspace = ContactSpace(
        boundary,
        tol=0.1,
        core_tolerance=0.15,
    )

    assert "is_core" in contactspace.data.columns
    assert "is_core" in contactspace.annotation_columns
    assert contactspace.data["is_core"].tolist().count(True) == 3


def test_maps_annotate_ionic_distance_updates_contactspace_and_maps_data() -> None:
    system = _build_system()
    metric = IonicGeometry(mode="muff", alpha=1.0, system=system)
    radius = metric.radii[0]
    center = system.atoms.positions[0]
    positions = np.array(
        [
            center,
            center + np.array([radius + 0.25, 0.0, 0.0]),
        ]
    )
    contactspace = StubContactSpace(positions)

    maps = Maps(system, [], contactspace)
    maps.data = pd.DataFrame(positions, columns=["x", "y", "z"])

    distances = maps.annotate_ionic_distance(radiusmode="muff", alpha=1.0)

    np.testing.assert_allclose(distances, np.array([-radius, 0.25]), atol=1e-8)
    np.testing.assert_allclose(contactspace.data["ionic_distance"].to_numpy(), distances)
    assert maps.data is not None
    np.testing.assert_allclose(maps.data["ionic_distance"].to_numpy(), distances)
    assert maps.features == ["ionic_distance"]


def test_maps_save_and_load_roundtrip_preserves_cached_data(tmp_path) -> None:
    system = _build_system()
    positions = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    contactspace = StubContactSpace(positions)
    contactspace.data.loc[:, "region"] = np.array([0, 1], dtype=np.int64)
    contactspace.data.loc[:, "core_distance"] = np.array([0.2, 0.3], dtype=np.float64)

    maps = Maps(system, [], contactspace)
    maps.data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "region": [0, 1],
            "core_distance": [0.2, 0.3],
            "f1": [0.1, 0.2],
        }
    )
    maps.features = ["f1"]
    maps.add_special_points([1], kind="adaptive", iteration=2, label_status="completed")

    path = maps.save(tmp_path / "maps.pkl")
    loaded = Maps.load(path)

    assert loaded.data is not None
    pd.testing.assert_frame_equal(loaded.data, maps.data)
    assert loaded.features == ["f1"]
    pd.testing.assert_frame_equal(
        loaded.get_special_points(kind="adaptive").reset_index(drop=True),
        maps.get_special_points(kind="adaptive").reset_index(drop=True),
    )
