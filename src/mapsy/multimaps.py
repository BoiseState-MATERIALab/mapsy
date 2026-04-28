from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from yaml import SafeLoader, load

from mapsy.analysis import fit_clusters, fit_pca_analysis, project_pca, screen_clusters
from mapsy.clustering import (
    clustering_uses_random_state,
    normalize_cluster_method,
)
from mapsy.results import ClusterResult, ClusterScreeningResult, PCAAnalysisResult, PCAResult

if TYPE_CHECKING:
    from .maps import Maps


class MultiMaps:
    def __init__(
        self,
        maps: Sequence["Maps"],
        names: Sequence[str] | None = None,
    ) -> None:
        if not maps:
            raise ValueError("maps must contain at least one Maps instance")

        self.maps = list(maps)
        self.names = (
            list(names)
            if names is not None
            else [f"system_{index}" for index in range(len(self.maps))]
        )
        if len(self.names) != len(self.maps):
            raise ValueError("names and maps must have the same length")

        self.data: pd.DataFrame | None = None
        self.features: list[str] = []
        self.npca: int | None = None
        self.pca_analysis_result: PCAAnalysisResult | None = None
        self.pca_result: PCAResult | None = None

        self.nclusters: int = 0
        self.cluster_method: str = "spectral"
        self.cluster_screening_method: str | None = None
        self.cluster_result: ClusterResult | None = None
        self.cluster_features: list[str] = []
        self.cluster_centers: np.ndarray[Any, np.dtype[np.float64]] | None = None
        self.cluster_graph: np.ndarray[Any, np.dtype[np.int64]] | None = None
        self.cluster_edges: np.ndarray[Any, np.dtype[np.int64]] | None = None
        self.best_clusters: pd.DataFrame | None = None
        self.cluster_screening: pd.DataFrame | None = None

        self._slices: list[slice] = []

    def atcontactspace(self) -> pd.DataFrame:
        return self._build_dataset(recompute=True)

    def analyze_pca(self, scale: bool = False) -> PCAAnalysisResult:
        data = self._ensure_data()
        X = data[self.features].values.astype(np.float64)
        result = fit_pca_analysis(X, feature_columns=list(self.features), scale=scale)
        self.pca_analysis_result = result
        return result

    def reduce(self, npca: int | None = None, scale: bool = False) -> PCAAnalysisResult | PCAResult:
        if npca is None:
            return self.analyze_pca(scale=scale)
        data = self._ensure_data()
        analysis = self.analyze_pca(scale=scale)
        result = project_pca(analysis, data[self.features].values.astype(np.float64), npca=npca)
        self.npca = npca
        self.pca_result = result
        data.loc[:, result.transformed_columns] = result.transformed_values
        self._propagate_columns(result.transformed_columns)
        return result

    def cluster(
        self,
        nclusters: int | None = None,
        features: list[str] | None = None,
        method: str = "spectral",
        maxclusters: int = 20,
        ntries: int = 1,
        random_state: int | None = None,
        scale: bool = False,
    ) -> ClusterResult | ClusterScreeningResult:
        data = self._ensure_data()
        self.cluster_features = list(features) if features is not None else list(self.features)
        self.cluster_method = normalize_cluster_method(method)
        X = data[self.cluster_features].values.astype(np.float64)

        if nclusters is not None:
            chosen_random_state = self._select_random_state(nclusters, random_state)
            screening_result = (
                ClusterScreeningResult(
                    method=self.cluster_method,
                    feature_columns=list(self.cluster_features),
                    scale=scale,
                    table=self.cluster_screening.copy(),
                    best_by_db=self.best_clusters.copy(),
                    best_by_silhouette=self.cluster_screening.loc[
                        self.cluster_screening.groupby("nclusters")["silhouette_score"].idxmax()
                    ].copy(),
                )
                if self.cluster_screening is not None
                and self.best_clusters is not None
                and self.cluster_screening_method == self.cluster_method
                else None
            )
            result = fit_clusters(
                X,
                feature_columns=list(self.cluster_features),
                nclusters=nclusters,
                method=self.cluster_method,
                random_state=chosen_random_state,
                scale=scale,
                screening=screening_result,
            )
            data.loc[:, "Cluster"] = result.labels
            self.nclusters = nclusters
            self.cluster_result = result
            self.cluster_centers = result.centers
            self.cluster_graph = self._aggregate_graph(result.labels.astype(np.int64))
            self.cluster_edges = self.cluster_graph.copy()
            for i in range(nclusters):
                self.cluster_edges[i, i] = 0
            result.graph = self.cluster_graph
            result.edges = self.cluster_edges
            self._propagate_columns(["Cluster"])
            return result

        screening = screen_clusters(
            X,
            feature_columns=list(self.cluster_features),
            method=self.cluster_method,
            maxclusters=maxclusters,
            ntries=ntries,
            scale=scale,
        )
        self.cluster_screening = screening.table.copy()
        self.cluster_screening_method = screening.method
        self.best_clusters = screening.best_by_db.copy()
        return screening

    def _build_dataset(self, recompute: bool) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        reference_features: list[str] | None = None
        self._slices = []

        start = 0
        for index, (name, maps) in enumerate(zip(self.names, self.maps, strict=False)):
            frame = self._get_map_frame(maps, recompute).copy()
            current_features = self._get_map_features(maps)
            if reference_features is None:
                reference_features = current_features
            elif set(current_features) != set(reference_features):
                raise ValueError("All maps in MultiMaps must expose the same feature columns")

            ordered = frame.loc[:, ["x", "y", "z", *reference_features]].copy()
            ordered.insert(0, "point_index", np.arange(len(ordered), dtype=np.int64))
            ordered.insert(0, "map_index", index)
            ordered.insert(0, "system", name)
            frames.append(ordered)

            stop = start + len(ordered)
            self._slices.append(slice(start, stop))
            start = stop

        if reference_features is None:
            raise RuntimeError("No maps data available.")

        self.features = reference_features
        self.data = pd.concat(frames, ignore_index=True)
        return self.data

    def _ensure_data(self) -> pd.DataFrame:
        if self.data is None:
            return self._build_dataset(recompute=False)
        return self.data

    def _get_map_frame(self, maps: "Maps", recompute: bool) -> pd.DataFrame:
        if recompute or maps.data is None:
            if maps.contactspace is None:
                raise RuntimeError("Each Maps instance must define a contact space")
            return maps.atcontactspace()
        return maps.data

    def _get_map_features(self, maps: "Maps") -> list[str]:
        features = list(maps.features)
        if features:
            return features
        if maps.data is None:
            raise RuntimeError("No maps data available.")
        return [column for column in maps.data.columns if column not in {"x", "y", "z"}]

    def _propagate_columns(self, columns: list[str]) -> None:
        if self.data is None:
            raise RuntimeError("No MultiMaps data available.")

        for maps, data_slice in zip(self.maps, self._slices, strict=False):
            if maps.data is None:
                raise RuntimeError("Child Maps data is not initialized.")
            for column in columns:
                maps.data.loc[:, column] = self.data.iloc[data_slice][column].to_numpy()

    def _aggregate_graph(
        self, labels: np.ndarray[Any, np.dtype[np.int64]]
    ) -> np.ndarray[Any, np.dtype[np.int64]]:
        graph = np.zeros((self.nclusters, self.nclusters), dtype=np.int64)
        for maps, data_slice in zip(self.maps, self._slices, strict=False):
            local_labels = labels[data_slice]
            local_graph = maps.graph(local_labels)
            nrows, ncols = local_graph.shape
            graph[:nrows, :ncols] += local_graph
        return graph

    def _select_random_state(self, nclusters: int, random_state: int | None) -> int:
        if not clustering_uses_random_state(self.cluster_method):
            return 0
        if random_state is not None:
            return random_state
        if (
            self.best_clusters is not None
            and self.cluster_screening_method == self.cluster_method
            and nclusters in self.best_clusters["nclusters"].values
        ):
            return int(
                self.best_clusters[self.best_clusters["nclusters"] == nclusters][
                    "random_state"
                ].values[0]
            )
        return int(np.random.randint(0, 1000))


class MultiMapsFromFile(MultiMaps):
    def __init__(self, filename: str) -> None:
        from mapsy.io.parser import ContactSpaceGenerator, SystemParser
        from mapsy.maps import (
            Maps,
            _namespace_contactspace,
            _namespace_symmetryfunctions,
            _namespace_system,
        )
        from mapsy.symfunc.parser import SymmetryFunctionsParser

        with open(Path(filename).expanduser().resolve()) as handle:
            params = load(handle, SafeLoader) or {}

        control = params.get("control") or {}
        system = params.get("system")
        symmetryfunctions = params.get("symmetryfunctions")
        contactspace = params.get("contactspace")

        if system is None:
            raise RuntimeError("System section missing in input file")
        if symmetryfunctions is None:
            raise RuntimeError("Symmetry functions section missing in input file")
        if contactspace is None:
            raise RuntimeError("Contact space section missing in input file")

        basepath = Path(filename).expanduser().resolve().parent
        systemmodel = _namespace_system(system)
        system_parser = SystemParser(systemmodel, basepath=basepath)
        system_files = system_parser.filenames()

        if len(system_files) > 1 and systemmodel.properties:
            raise NotImplementedError(
                "MultiMapsFromFile does not support per-system properties yet."
            )

        maps_list = []
        for system in system_parser.parse_many():
            parsed_symmetryfunctions = SymmetryFunctionsParser(
                _namespace_symmetryfunctions(symmetryfunctions)
            ).parse()
            parsed_contactspace = ContactSpaceGenerator(
                _namespace_contactspace(contactspace)
            ).generate(system)
            maps_list.append(Maps(system, parsed_symmetryfunctions, parsed_contactspace))

        super().__init__(maps_list, names=[Path(path).stem for path in system_files])
        self.debug = bool(control.get("debug", False))
        self.verbosity = int(control.get("verbosity", 0))
