from types import SimpleNamespace

import numpy as np
import pandas as pd
from ase import Atoms

from mapsy import Maps
from mapsy.boundaries.ionic import IonicGeometry
from mapsy.data import Grid, System
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
    assert maps.features == ["ionic_distance"]
    np.testing.assert_allclose(data["ionic_distance"].to_numpy(), annotation)


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
