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
    maps.atpoints = lambda positions, *, workers=None: pd.DataFrame(
        positions, columns=["x", "y", "z"]
    )

    data = maps.atcontactspace()

    assert "ionic_distance" in data.columns
    assert "region" in data.columns
    assert "signed_distance" in data.columns
    assert "core_distance" in data.columns
    assert "patch" in data.columns
    assert "layer" in data.columns
    assert maps.features == ["ionic_distance"]
    assert "region" not in maps.features
    assert "signed_distance" not in maps.features
    assert "core_distance" not in maps.features
    assert "patch" not in maps.features
    assert "layer" not in maps.features
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
        tol=0.3,
        core_epsilon=1.0e-12,
    )

    assert "core_distance" in contactspace.data.columns
    assert "signed_distance" in contactspace.data.columns
    assert "patch" in contactspace.data.columns
    assert "layer" in contactspace.data.columns
    assert "core_distance" in contactspace.annotation_columns
    assert "signed_distance" in contactspace.annotation_columns
    assert "patch" in contactspace.annotation_columns
    assert "layer" in contactspace.annotation_columns
    assert "core_distance" not in contactspace.feature_columns
    np.testing.assert_allclose(
        np.sort(contactspace.data["core_distance"].to_numpy()),
        np.array([0.0, 0.0, 0.25, 0.25]),
    )


def test_contactspace_layers_order_by_absolute_signed_distance() -> None:
    grid = Grid(scalars=[2, 2, 3], cell=np.diag([2.0, 2.0, 3.0]))
    boundary = DummyBoundary(mode="system", grid=grid)

    switch = ScalarField(grid)
    switch[:] = np.array(
        [
            [[0.30, 0.50, 0.70], [0.30, 0.50, 0.70]],
            [[0.30, 0.50, 0.70], [0.30, 0.50, 0.70]],
        ]
    )
    boundary.switch[:] = switch

    gradient = GradientField(grid)
    gradient[0, :, :, :] = 0.0
    gradient[1, :, :, :] = 0.0
    gradient[2, :, :, :] = 1.0
    boundary.gradient[:] = gradient

    contactspace = ContactSpace(boundary, tol=0.25, core_epsilon=1.0e-12)

    nonnegative_layers = np.sort(
        contactspace.data.loc[contactspace.data["layer"] >= 0, "layer"].unique()
    )
    np.testing.assert_array_equal(nonnegative_layers, np.array([0, 1, 2], dtype=np.int64))

    layer_by_z = contactspace.data.groupby("z")["layer"].first().to_numpy(dtype=np.int64)
    np.testing.assert_array_equal(layer_by_z, np.array([1, 0, 2], dtype=np.int64))

    layer_count_by_z = contactspace.data.groupby("z")["layer"].nunique()
    np.testing.assert_array_equal(layer_count_by_z.to_numpy(dtype=np.int64), np.array([1, 1, 1]))

    layer_distance_by_z = (
        contactspace.data.groupby("z")["layer_mean_distance"].first().to_numpy(dtype=np.float64)
    )
    np.testing.assert_allclose(layer_distance_by_z, np.array([-0.2, 0.0, 0.2], dtype=np.float64))


def test_layer_cluster_assignment_can_use_fixed_layer_count() -> None:
    patch_sizes = np.array([10, 12, 8, 9, 11, 10], dtype=np.int64)
    patch_means = np.array([-0.22, -0.20, -0.02, 0.00, 0.21, 0.24], dtype=np.float64)
    major_patch_ids = np.arange(patch_sizes.size, dtype=np.int64)

    layer_by_patch, layer_mean_distances = ContactSpace._assign_layers_from_patch_clusters(
        patch_sizes_by_id=patch_sizes,
        patch_mean_distances_by_id=patch_means,
        major_patch_ids=major_patch_ids,
        n_layers=3,
    )

    np.testing.assert_array_equal(layer_by_patch, np.array([1, 1, 0, 0, 2, 2], dtype=np.int64))
    np.testing.assert_allclose(
        layer_mean_distances[[0, 2, 4]],
        np.array([-0.20909091, -0.00941176, 0.22428571], dtype=np.float64),
        rtol=1.0e-6,
    )


def test_layer_auto_cluster_assignment_stops_after_large_distance_groups() -> None:
    patch_sizes = np.array([10, 12, 8, 9, 11, 10], dtype=np.int64)
    patch_means = np.array([-0.22, -0.20, -0.02, 0.00, 0.21, 0.24], dtype=np.float64)
    major_patch_ids = np.arange(patch_sizes.size, dtype=np.int64)

    layer_by_patch, layer_mean_distances = ContactSpace._assign_layers_from_patch_clusters(
        patch_sizes_by_id=patch_sizes,
        patch_mean_distances_by_id=patch_means,
        major_patch_ids=major_patch_ids,
        n_layers="auto",
    )

    np.testing.assert_array_equal(layer_by_patch, np.array([1, 1, 0, 0, 2, 2], dtype=np.int64))
    np.testing.assert_allclose(
        layer_mean_distances[[0, 2, 4]],
        np.array([-0.20909091, -0.00941176, 0.22428571], dtype=np.float64),
        rtol=1.0e-6,
    )


def test_contactspace_model_defaults_to_auto_layer_count() -> None:
    from mapsy.io.input.base import ContactSpaceModel

    model = ContactSpaceModel.parse_obj({"mode": "system"})

    assert model.n_layers == "auto"


def test_contactspace_model_rejects_layer_merge_tolerance() -> None:
    from pydantic import ValidationError

    from mapsy.io.input.base import ContactSpaceModel

    with np.testing.assert_raises(ValidationError):
        ContactSpaceModel.parse_obj({"mode": "system", "layer_merge_tolerance": 0.1})


def test_contactspace_threshold_uses_distance_metric_not_raw_gradient() -> None:
    grid = Grid(scalars=[2, 1, 1], cell=np.diag([2.0, 1.0, 1.0]))
    boundary = DummyBoundary(mode="system", grid=grid)

    switch = ScalarField(grid)
    switch[:] = np.array(
        [
            [[0.40]],
            [[0.70]],
        ]
    )
    boundary.switch[:] = switch

    gradient = GradientField(grid)
    gradient[0, :, :, :] = np.array(
        [
            [[2.0]],
            [[1.0]],
        ]
    )
    gradient[1, :, :, :] = 0.0
    gradient[2, :, :, :] = 0.0
    boundary.gradient[:] = gradient

    contactspace = ContactSpace(boundary, tol=0.06, core_epsilon=1.0e-12)

    assert contactspace.nm == 1
    np.testing.assert_allclose(contactspace.data["core_distance"].to_numpy(), np.array([0.05]))


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
