from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mapsy import (
    ClusterResult,
    ClusterScreeningResult,
    MultiMaps,
    MultiMapsFromFile,
    PCAAnalysisResult,
    PCAResult,
    plot_cluster_screening,
    plot_pca_scree,
)
from mapsy.io.parser import resolve_file_model, resolve_file_records


class FakeMaps:
    _metadata_columns = {
        "region",
        "signed_distance",
        "core_distance",
        "patch",
        "layer",
        "patch_size",
        "layer_size",
        "patch_mean_distance",
        "layer_mean_distance",
    }

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        nneighbors = np.full((len(frame), 6), -1, dtype=np.int64)
        self.contactspace = SimpleNamespace(nm=len(frame), neighbors=nneighbors)
        self.data: pd.DataFrame | None = None
        self.features: list[str] = []

    def atcontactspace(self) -> pd.DataFrame:
        self.data = self._frame.copy()
        self.features = [
            column
            for column in self.data.columns
            if column not in {"x", "y", "z", *self._metadata_columns}
        ]
        return self.data


def _write_xyz(path: Path, x: float) -> None:
    path.write_text(
        "\n".join(
            [
                "1",
                "5.0 5.0 5.0",
                f"H {x:.1f} 1.0 1.0",
            ]
        )
        + "\n"
    )


def _write_extxyz(path: Path, x: float) -> None:
    path.write_text(
        "\n".join(
            [
                "1",
                'Lattice="5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 5.0" '
                'Origin="0.0 0.0 0.0" Properties=species:S:1:pos:R:3',
                f"H {x:.1f} 1.0 1.0",
            ]
        )
        + "\n"
    )


def test_multimaps_combines_contact_space_data() -> None:
    map_a = FakeMaps(
        pd.DataFrame(
            {
                "x": [0.0, 1.0],
                "y": [0.0, 1.0],
                "z": [0.0, 0.0],
                "region": [0, 0],
                "core_distance": [0.1, 0.2],
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
                "region": [1, 1],
                "core_distance": [0.3, 0.4],
                "f1": [0.3, 0.4],
                "f2": [1.2, 1.3],
            }
        )
    )

    multimaps = MultiMaps([map_a, map_b], names=["a", "b"])
    combined = multimaps.atcontactspace()

    assert multimaps.features == ["f1", "f2"]
    assert "region" not in multimaps.features
    assert "core_distance" not in multimaps.features
    assert combined["system"].tolist() == ["a", "a", "b", "b"]
    assert combined["map_index"].tolist() == [0, 0, 1, 1]
    assert combined["point_index"].tolist() == [0, 1, 0, 1]
    assert combined["region"].tolist() == [0, 0, 1, 1]
    assert combined["core_distance"].tolist() == [0.1, 0.2, 0.3, 0.4]


def test_multimaps_save_and_load_roundtrip_preserves_cached_data(tmp_path: Path) -> None:
    map_a = FakeMaps(
        pd.DataFrame(
            {
                "x": [0.0, 1.0],
                "y": [0.0, 1.0],
                "z": [0.0, 0.0],
                "region": [0, 0],
                "core_distance": [0.1, 0.2],
                "f1": [0.1, 0.2],
            }
        )
    )
    map_b = FakeMaps(
        pd.DataFrame(
            {
                "x": [2.0, 3.0],
                "y": [2.0, 3.0],
                "z": [0.0, 0.0],
                "region": [1, 1],
                "core_distance": [0.3, 0.4],
                "f1": [0.3, 0.4],
            }
        )
    )

    multimaps = MultiMaps([map_a, map_b], names=["a", "b"])
    combined = multimaps.atcontactspace().copy()

    path = multimaps.save(tmp_path / "multimaps.pkl")
    loaded = MultiMaps.load(path)

    assert loaded.data is not None
    pd.testing.assert_frame_equal(loaded.data, combined)
    assert loaded.features == ["f1"]
    assert loaded.names == ["a", "b"]
    assert loaded.maps[0].data is not None
    assert loaded.maps[1].data is not None


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
    result = multimaps.reduce(npca=2)

    assert multimaps.data is not None
    assert {"pca0", "pca1"}.issubset(multimaps.data.columns)
    assert isinstance(result, PCAResult)
    assert result.transformed_columns == ["pca0", "pca1"]
    assert map_a.data is not None
    assert map_b.data is not None
    assert {"pca0", "pca1"}.issubset(map_a.data.columns)
    assert {"pca0", "pca1"}.issubset(map_b.data.columns)


def test_multimaps_reduce_can_fit_pca_on_layer() -> None:
    map_a = FakeMaps(
        pd.DataFrame(
            {
                "x": [0.0, 0.1],
                "y": [0.0, 0.1],
                "z": [0.0, 0.0],
                "layer": [0, 1],
                "f1": [0.0, 100.0],
                "f2": [1.0, 50.0],
            }
        )
    )
    map_b = FakeMaps(
        pd.DataFrame(
            {
                "x": [1.0, 1.1],
                "y": [1.0, 1.1],
                "z": [0.0, 0.0],
                "layer": [0, 1],
                "f1": [2.0, 200.0],
                "f2": [1.0, 60.0],
            }
        )
    )

    multimaps = MultiMaps([map_a, map_b])
    multimaps.atcontactspace()
    result = multimaps.reduce(npca=1, layer=0)

    assert result.npca == 1
    assert multimaps.pca_analysis_result is not None
    np.testing.assert_allclose(
        multimaps.pca_analysis_result.estimator.mean_,
        np.array([1.0, 1.0], dtype=np.float64),
    )
    assert multimaps.data is not None
    assert "pca0" in multimaps.data.columns


@pytest.mark.parametrize(
    "method",
    ["spectral", "gaussian_mixture", "kmeans", "agglomerative"],
)
def test_multimaps_cluster_propagates_labels(method: str) -> None:
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
    result = multimaps.cluster(nclusters=2, random_state=0, method=method)

    assert multimaps.data is not None
    assert "Cluster" in multimaps.data.columns
    assert multimaps.nclusters == 2
    assert multimaps.cluster_method == method
    assert isinstance(result, ClusterResult)
    assert multimaps.cluster_graph is not None
    assert multimaps.cluster_graph.shape == (2, 2)
    assert map_a.data is not None
    assert map_b.data is not None
    assert "Cluster" in map_a.data.columns
    assert "Cluster" in map_b.data.columns


def test_multimaps_cluster_can_fit_on_layer() -> None:
    map_a = FakeMaps(
        pd.DataFrame(
            {
                "x": [0.0, 0.1],
                "y": [0.0, 0.1],
                "z": [0.0, 0.0],
                "layer": [0, 1],
                "f1": [0.0, 100.0],
            }
        )
    )
    map_b = FakeMaps(
        pd.DataFrame(
            {
                "x": [1.0, 1.1],
                "y": [1.0, 1.1],
                "z": [0.0, 0.0],
                "layer": [0, 1],
                "f1": [10.0, 200.0],
            }
        )
    )

    multimaps = MultiMaps([map_a, map_b])
    multimaps.atcontactspace()
    result = multimaps.cluster(nclusters=2, method="kmeans", random_state=0, layer=1)

    assert result.metadata is not None
    assert result.metadata["layer"] == [1]
    assert multimaps.data is not None
    np.testing.assert_array_equal(
        multimaps.data["Cluster"].to_numpy(dtype=np.int64),
        np.array([-1, 0, -1, 1], dtype=np.int64),
    )


def test_multimaps_cluster_can_propagate_layer_labels() -> None:
    map_a = FakeMaps(
        pd.DataFrame(
            {
                "x": [0.0, 1.0],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
                "probability": [1.0, 1.0],
                "layer": [0, 1],
                "f1": [100.0, 0.0],
            }
        )
    )
    map_b = FakeMaps(
        pd.DataFrame(
            {
                "x": [2.0, 3.0],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
                "probability": [1.0, 1.0],
                "layer": [1, 0],
                "f1": [10.0, 200.0],
            }
        )
    )

    multimaps = MultiMaps([map_a, map_b])
    multimaps.atcontactspace()
    multimaps.build_graph(mode="realspace", feature_columns=["f1"])
    result = multimaps.cluster(
        nclusters=2,
        method="kmeans",
        random_state=0,
        layer=1,
        propagate=True,
        propagation_mode="region_grow",
    )

    assert result.metadata is not None
    assert result.metadata["propagate"] is True
    assert multimaps.data is not None
    labels = multimaps.data["Cluster"].to_numpy(dtype=np.int64)
    assert np.all(labels >= 0)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[3]


def test_multimaps_scatter_uses_shared_categorical_legend() -> None:
    plt.switch_backend("Agg")
    map_a = FakeMaps(
        pd.DataFrame(
            {
                "x": [0.0, 1.0],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
                "region": [0, 0],
                "Cluster": [0, 1],
                "f1": [0.0, 1.0],
            }
        )
    )
    map_b = FakeMaps(
        pd.DataFrame(
            {
                "x": [2.0, 3.0],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
                "region": [0, 0],
                "Cluster": [1, 2],
                "f1": [2.0, 3.0],
            }
        )
    )

    multimaps = MultiMaps([map_a, map_b], names=["a", "b"])
    multimaps.atcontactspace()
    multimaps.nclusters = 3
    fig, axs = multimaps.scatter(
        feature="Cluster",
        categorical=True,
        map_indexes=[0],
        s=10,
    )

    assert axs.shape == (1, 1)
    assert len(fig.legends) == 1
    labels = [text.get_text() for text in fig.legends[0].get_texts()]
    assert labels == ["Cluster = 0", "Cluster = 1", "Cluster = 2"]
    plt.close(fig)


def test_multimaps_scatter_core_projection_selects_center_per_map() -> None:
    plt.switch_backend("Agg")
    map_a = FakeMaps(
        pd.DataFrame(
            {
                "x": [0.0, 0.0, 1.0],
                "y": [0.0, 0.0, 0.0],
                "z": [0.0, 1.0, 0.0],
                "region": [0, 0, 0],
                "layer": [0, 0, 0],
                "probability": [1.0, 2.0, 1.0],
                "interface_center_distance": [0.4, 0.1, 0.2],
                "f1": [10.0, 20.0, 30.0],
            }
        )
    )
    map_b = FakeMaps(
        pd.DataFrame(
            {
                "x": [0.0, 0.0, 1.0],
                "y": [0.0, 0.0, 0.0],
                "z": [0.0, 1.0, 0.0],
                "region": [0, 0, 0],
                "layer": [0, 0, 0],
                "probability": [1.0, 1.0, 1.0],
                "interface_center_distance": [0.3, 0.05, 0.6],
                "f1": [100.0, 200.0, 300.0],
            }
        )
    )

    multimaps = MultiMaps([map_a, map_b], names=["a", "b"])
    multimaps.atcontactspace()
    fig, axs, projection = multimaps.scatter_core_projection(
        feature="f1",
        selector="center",
        distance_column="interface_center_distance",
        return_projection=True,
        s=10,
    )

    assert axs.shape == (1, 2)
    selected = projection.sort_values(["map_index", "x", "y"]).reset_index(drop=True)
    np.testing.assert_allclose(
        selected["f1"].to_numpy(dtype=np.float64),
        np.array([20.0, 30.0, 200.0, 300.0], dtype=np.float64),
    )
    assert selected["multiplicity"].tolist() == [2, 1, 2, 1]
    assert axs[0, 0].collections[0].norm.vmin == 20.0
    assert axs[0, 0].collections[0].norm.vmax == 300.0
    assert axs[0, 1].collections[0].norm.vmin == 20.0
    assert axs[0, 1].collections[0].norm.vmax == 300.0
    plt.close(fig)


def test_multimaps_sites_can_select_one_site_per_cluster_and_layer() -> None:
    map_a = FakeMaps(
        pd.DataFrame(
            {
                "x": [0.0, 1.0],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
                "region": [0, 0],
                "layer": [0, 1],
                "f1": [0.0, 0.5],
                "Cluster": [0, 0],
            }
        )
    )
    map_b = FakeMaps(
        pd.DataFrame(
            {
                "x": [2.0, 3.0],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
                "region": [0, 0],
                "layer": [0, 1],
                "f1": [10.0, 9.5],
                "Cluster": [1, 1],
            }
        )
    )

    multimaps = MultiMaps([map_a, map_b], names=["a", "b"])
    multimaps.atcontactspace()
    multimaps.cluster_features = ["f1"]
    multimaps.cluster_centers = np.array([[0.0], [10.0]], dtype=np.float64)
    multimaps.cluster_graph = np.zeros((2, 2), dtype=np.int64)
    multimaps.cluster_edges = np.zeros((2, 2), dtype=np.int64)
    multimaps.cluster_result = ClusterResult(
        method="kmeans",
        feature_columns=["f1"],
        scale=False,
        nclusters=2,
        labels=np.array([0, 0, 1, 1], dtype=np.int64),
        centers=np.array([[0.0], [10.0]], dtype=np.float64),
        sizes=np.array([2, 2], dtype=np.int64),
        random_state=0,
        metadata={},
    )

    selected = multimaps.sites(region=0, per_layer=True)

    assert selected["global_point_index"].tolist() == [0, 1, 2, 3]
    assert selected["layer"].tolist() == [0, 1, 0, 1]
    assert selected["cluster"].tolist() == [0, 0, 1, 1]


def test_multimaps_cluster_screening_tracks_method() -> None:
    plt.switch_backend("Agg")
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
    result = multimaps.cluster(method="gaussian_mixture", maxclusters=4, ntries=2)

    assert isinstance(result, ClusterScreeningResult)
    assert multimaps.cluster_screening is not None
    assert set(multimaps.cluster_screening["method"]) == {"gaussian_mixture"}
    assert multimaps.cluster_screening_method == "gaussian_mixture"
    fig, _, _ = plot_cluster_screening(result)
    assert fig is not None


def test_multimaps_reduce_screening_plot_helper() -> None:
    plt.switch_backend("Agg")
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
    result = multimaps.analyze_pca(scale=True)

    assert isinstance(result, PCAAnalysisResult)
    fig, _, _ = plot_pca_scree(result)
    assert fig is not None


def test_resolve_file_model_supports_folder_and_root(tmp_path: Path) -> None:
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir()
    _write_xyz(systems_dir / "sample_a.xyz", 1.0)
    _write_xyz(systems_dir / "sample_b.xyz", 2.0)

    filemodel = SimpleNamespace(
        fileformat="xyz+",
        name="",
        folder="systems",
        root="sample_",
        units="angstrom",
    )
    resolved = resolve_file_model(filemodel, basepath=tmp_path)

    assert resolved == [
        str((systems_dir / "sample_a.xyz").resolve()),
        str((systems_dir / "sample_b.xyz").resolve()),
    ]


def test_resolve_file_model_supports_multiple_folders_and_source_metadata(
    tmp_path: Path,
) -> None:
    folder_a = tmp_path / "cop-10"
    folder_b = tmp_path / "cop-15"
    folder_a.mkdir()
    folder_b.mkdir()
    _write_extxyz(folder_a / "sample_a.extxyz", 1.0)
    _write_extxyz(folder_b / "sample_b.extxyz", 2.0)

    filemodel = SimpleNamespace(
        fileformat="xyz+",
        name="",
        names=[],
        folder="",
        folders=["cop-10", "cop-15"],
        root="",
        pattern="*.extxyz",
        recursive=False,
        units="angstrom",
    )
    records = resolve_file_records(filemodel, basepath=tmp_path)

    assert resolve_file_model(filemodel, basepath=tmp_path) == [
        str((folder_a / "sample_a.extxyz").resolve()),
        str((folder_b / "sample_b.extxyz").resolve()),
    ]
    assert [record.source_folder_name for record in records] == ["cop-10", "cop-15"]
    assert [record.source_folder_number for record in records] == [10, 15]


def test_multimaps_from_file_loads_multiple_systems(tmp_path: Path) -> None:
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir()
    _write_xyz(systems_dir / "sample_a.xyz", 1.0)
    _write_xyz(systems_dir / "sample_b.xyz", 2.0)

    input_file = tmp_path / "multimaps.yaml"
    input_file.write_text(
        dedent(
            """
            control:
              debug: false
              verbosity: 0

            system:
              systemtype: ions
              file:
                fileformat: xyz+
                folder: systems
                root: sample_
                units: angstrom
              dimension: 2
              axis: 2

            contactspace:
              mode: system
              distance: 0.5
              spread: 0.5
              cutoff: 2
              threshold: 0.1

            symmetryfunctions:
              functions:
                - type: ac
                  cutoff: cos
                  radius: 2.0
                  order: 1
                  compositional: false
                  structural: true
                  radial: true
            """
        ).strip()
        + "\n"
    )

    multimaps = MultiMapsFromFile(str(input_file))

    assert len(multimaps.maps) == 2
    assert multimaps.names == ["sample_a", "sample_b"]

    combined = multimaps.atcontactspace()

    assert not combined.empty
    assert combined["system"].isin(["sample_a", "sample_b"]).all()


def test_multimaps_from_file_rejects_unknown_contactspace_keys(tmp_path: Path) -> None:
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir()
    _write_xyz(systems_dir / "sample_a.xyz", 1.0)

    input_file = tmp_path / "multimaps.yaml"
    input_file.write_text(
        dedent(
            """
            system:
              systemtype: ions
              file:
                fileformat: xyz+
                folder: systems
                root: sample_
                units: angstrom
              dimension: 2
              axis: 2

            contactspace:
              mode: system
              distance: 0.5
              spread: 0.5
              cutoff: 2
              threshold: 0.1
              layer_switch_tolerance: 0.1

            symmetryfunctions:
              functions:
                - type: ac
                  cutoff: cos
                  radius: 2.0
                  order: 1
                  compositional: false
                  structural: true
                  radial: true
            """
        ).strip()
        + "\n"
    )

    with pytest.raises(Exception, match="layer_switch_tolerance|extra fields not permitted"):
        MultiMapsFromFile(str(input_file))


def test_multimaps_from_file_preserves_source_folder_metadata(tmp_path: Path) -> None:
    folder_a = tmp_path / "cop-10"
    folder_b = tmp_path / "cop-15"
    folder_a.mkdir()
    folder_b.mkdir()
    _write_extxyz(folder_a / "sample_a.extxyz", 1.0)
    _write_extxyz(folder_b / "sample_b.extxyz", 2.0)

    input_file = tmp_path / "multimaps.yaml"
    input_file.write_text(
        dedent(
            """
            control:
              debug: false
              verbosity: 0

            system:
              systemtype: ions
              file:
                fileformat: xyz+
                folders:
                  - cop-10
                  - cop-15
                pattern: "*.extxyz"
                units: angstrom
              dimension: 2
              axis: 2

            contactspace:
              mode: system
              distance: 0.5
              spread: 0.5
              cutoff: 2
              threshold: 0.1

            symmetryfunctions:
              functions:
                - type: ac
                  cutoff: cos
                  radius: 2.0
                  order: [0]
                  compositional: false
                  structural: true
                  radial: true
            """
        ).strip()
        + "\n"
    )

    multimaps = MultiMapsFromFile(str(input_file))
    combined = multimaps.atcontactspace()

    assert len(multimaps.features) == 1
    assert "source_folder_number" not in multimaps.features
    assert combined.groupby("system")["source_folder_number"].first().to_dict() == {
        "sample_a": 10,
        "sample_b": 15,
    }
    assert set(combined["source_folder_name"]) == {"cop-10", "cop-15"}
    assert all("source_folder_number" in maps.data.columns for maps in multimaps.maps)
