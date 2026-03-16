from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mapsy import MultiMaps, MultiMapsFromFile
from mapsy.io.parser import resolve_file_model


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
                  order: [0]
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
