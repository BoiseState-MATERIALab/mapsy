import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from yaml import SafeLoader, load

from mapsy.analysis import (
    ArchetypePropagationMode,
    ArchetypeSelectionMode,
    FeatureConnectivityMode,
    GraphMode,
    aggregate_cluster_graph,
    build_point_graph,
    fit_clusters,
    fit_pca_analysis,
    project_pca,
    screen_clusters,
)
from mapsy.analysis import (
    propagate_archetypes as propagate_archetypes_on_graph,
)
from mapsy.analysis import (
    select_archetypes as select_feature_archetypes,
)
from mapsy.clustering import (
    clustering_uses_random_state,
    normalize_cluster_method,
)
from mapsy.maps import _coerce_layer_values
from mapsy.results import (
    ArchetypePropagationResult,
    ArchetypeSelectionResult,
    ClusterResult,
    ClusterScreeningResult,
    GraphResult,
    PCAAnalysisResult,
    PCAResult,
)

if TYPE_CHECKING:
    from .maps import Maps


class MultiMaps:
    _METADATA_COLUMNS = (
        "region",
        "layer",
        "source_file",
        "source_file_name",
        "source_file_stem",
        "source_folder",
        "source_folder_name",
        "source_folder_number",
    )

    def __init__(
        self,
        maps: Sequence["Maps"],
        names: Sequence[str] | None = None,
        metadata: Sequence[Mapping[str, Any]] | None = None,
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
        self.map_metadata = (
            [dict(item) for item in metadata] if metadata is not None else [{} for _ in self.maps]
        )
        if len(self.map_metadata) != len(self.maps):
            raise ValueError("metadata and maps must have the same length")

        self.data: pd.DataFrame | None = None
        self.features: list[str] = []
        self.npca: int | None = None
        self.pca_analysis_result: PCAAnalysisResult | None = None
        self.pca_result: PCAResult | None = None
        self.graph_result: GraphResult | None = None
        self.archetype_selection_result: ArchetypeSelectionResult | None = None
        self.archetype_propagation_result: ArchetypePropagationResult | None = None
        self.layer_ranking: pd.DataFrame | None = None

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

    def save(
        self,
        filename: str | Path,
        *,
        protocol: int = pickle.HIGHEST_PROTOCOL,
    ) -> Path:
        path = Path(filename).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=protocol)
        return path

    @classmethod
    def load(cls, filename: str | Path) -> "MultiMaps":
        path = Path(filename).expanduser().resolve()
        with path.open("rb") as handle:
            loaded = pickle.load(handle)
        if not isinstance(loaded, cls):
            raise TypeError(
                f"Pickle at {path} contains {type(loaded).__name__}, expected {cls.__name__}."
            )
        return loaded

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
        graph: GraphResult | None = None,
    ) -> ClusterResult | ClusterScreeningResult:
        data = self._ensure_data()
        self.cluster_features = list(features) if features is not None else list(self.features)
        self.cluster_method = normalize_cluster_method(method)
        X = data[self.cluster_features].values.astype(np.float64)
        if graph is not None and graph.matrix.shape[0] != len(X):
            raise ValueError(
                f"graph has {graph.matrix.shape[0]} nodes, expected {len(X)} to match multimaps.data."
            )

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
                graph=graph,
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
            graph=graph,
        )
        self.cluster_screening = screening.table.copy()
        self.cluster_screening_method = screening.method
        self.best_clusters = screening.best_by_db.copy()
        return screening

    def build_graph(
        self,
        *,
        mode: GraphMode = "hybrid",
        feature_columns: list[str] | None = None,
        node_weight_column: str = "probability",
        feature_k: int = 8,
        feature_connectivity: FeatureConnectivityMode = "knn",
        sigma_feature: float | None = None,
        realspace_weight: float = 1.0,
        feature_weight: float = 1.0,
        connect_systems: bool = False,
        normalize_node_weights: bool = True,
        use_node_weights_in_edges: bool = True,
        direction_columns: tuple[str, str, str] | None = None,
        directional_weight: float = 0.0,
        directional_power: float = 1.0,
    ) -> GraphResult:
        data = self._ensure_data()
        selected_features = (
            list(feature_columns) if feature_columns is not None else list(self.features)
        )
        node_table = data.loc[
            :, ["system", "map_index", "point_index", "x", "y", "z", *selected_features]
        ].copy()
        node_table.insert(0, "global_point_index", data.index.to_numpy(dtype=np.int64))
        required_columns = [node_weight_column]
        if directional_weight > 0.0:
            required_columns.extend(
                list(
                    direction_columns
                    or (
                        "boundary_gradient_x",
                        "boundary_gradient_y",
                        "boundary_gradient_z",
                    )
                )
            )
        for column in required_columns:
            if column in node_table.columns:
                continue
            if column in data.columns:
                node_table.loc[:, column] = data[column].to_numpy()
            else:
                node_table.loc[:, column] = self._collect_contactspace_column(column)

        neighbors = self._combine_contactspace_neighbors()
        result = build_point_graph(
            node_table,
            mode=mode,
            feature_columns=selected_features,
            neighbors=neighbors,
            node_weight_column=node_weight_column,
            feature_k=feature_k,
            feature_connectivity=feature_connectivity,
            sigma_feature=sigma_feature,
            realspace_weight=realspace_weight,
            feature_weight=feature_weight,
            connect_systems=connect_systems,
            system_ids=node_table["map_index"].to_numpy(dtype=np.int64),
            normalize_node_weights=normalize_node_weights,
            use_node_weights_in_edges=use_node_weights_in_edges,
            direction_columns=direction_columns,
            directional_weight=directional_weight,
            directional_power=directional_power,
        )
        self.graph_result = result
        return result

    def rank_layers(
        self,
        *,
        feature_columns: list[str] | None = None,
        scale_features: bool = True,
        completeness_weight: float = 1.0,
        variance_weight: float = 1.0,
        min_points: int = 1,
    ) -> pd.DataFrame:
        self._ensure_data()

        frames: list[pd.DataFrame] = []
        for map_index, (name, maps) in enumerate(zip(self.names, self.maps, strict=False)):
            ranking = maps.rank_layers(
                feature_columns=feature_columns,
                scale_features=scale_features,
                completeness_weight=completeness_weight,
                variance_weight=variance_weight,
                min_points=min_points,
            ).copy()
            ranking.insert(0, "system", name)
            ranking.insert(0, "map_index", map_index)
            frames.append(ranking)

        result = pd.concat(frames, ignore_index=True)
        self.layer_ranking = result.copy()
        return result

    def select_archetypes(
        self,
        n_archetypes: int,
        *,
        feature_columns: list[str] | None = None,
        graph: GraphResult | None = None,
        probability_column: str = "probability",
        region: int | None = None,
        layer: int | Sequence[int] | None = None,
        min_probability: float | None = None,
        min_probability_quantile: float | None = 0.75,
        scale_features: bool = True,
        selection_mode: ArchetypeSelectionMode = "feature_extreme",
        probability_weight: float = 1.0,
        extremeness_weight: float = 1.0,
        diversity_weight: float = 1.0,
        endpointness_weight: float = 1.0,
        geodesic_weight: float = 1.0,
        branching_weight: float = 0.5,
        register: bool = True,
        kind: str = "archetype",
        iteration: int | None = 0,
        label_status: str = "unlabeled",
        replace_kind: bool = True,
    ) -> ArchetypeSelectionResult:
        data = self._ensure_data()
        resolved_feature_columns = (
            list(feature_columns) if feature_columns is not None else list(self.features)
        )
        point_table = data.copy()
        point_table.loc[:, "global_point_index"] = data.index.to_numpy(dtype=np.int64)
        if probability_column not in point_table.columns:
            point_table.loc[:, probability_column] = self._collect_contactspace_column(
                probability_column
            )

        candidate_mask: np.ndarray[Any, np.dtype[np.bool_]] | None = None
        if region is not None or layer is not None:
            candidate_mask = np.ones(len(point_table), dtype=bool)
            if region is not None:
                candidate_mask &= (
                    self._collect_contactspace_column("region").astype(np.int64) == region
                )
            if layer is not None:
                layer_values = self._collect_contactspace_column("layer").astype(np.int64)
                requested_layers = _coerce_layer_values(layer)
                candidate_mask &= np.isin(layer_values, requested_layers)
                if min_probability is None and min_probability_quantile == 0.75:
                    min_probability_quantile = None

        selected_graph = graph if graph is not None else self.graph_result
        if selection_mode == "graph_endpoint" and selected_graph is None:
            raise RuntimeError(
                "selection_mode='graph_endpoint' requires a graph. "
                "Call multimaps.build_graph(...) first or pass graph explicitly."
            )

        result = select_feature_archetypes(
            point_table,
            n_archetypes=n_archetypes,
            feature_columns=resolved_feature_columns,
            probability_column=probability_column,
            point_index_column="global_point_index",
            candidate_mask=candidate_mask,
            min_probability=min_probability,
            min_probability_quantile=min_probability_quantile,
            scale_features=scale_features,
            selection_mode=selection_mode,
            graph=selected_graph,
            probability_weight=probability_weight,
            extremeness_weight=extremeness_weight,
            diversity_weight=diversity_weight,
            endpointness_weight=endpointness_weight,
            geodesic_weight=geodesic_weight,
            branching_weight=branching_weight,
        )
        self.archetype_selection_result = result

        if register:
            if replace_kind:
                for maps in self.maps:
                    maps.special_points.remove(kind=kind)
            self._register_archetypes_on_children(
                result.archetype_table,
                kind=kind,
                iteration=iteration,
                label_status=label_status,
            )

        return result

    def propagate_archetypes(
        self,
        *,
        graph: GraphResult | None = None,
        selected_indexes: npt.ArrayLike | None = None,
        propagation_mode: ArchetypePropagationMode = "diffusion",
        alpha: float = 0.9,
        max_iter: int = 500,
        tol: float = 1.0e-8,
        confidence_threshold: float = 0.5,
        margin_threshold: float = 0.0,
        propagation_realspace_scale: float = 1.0,
        propagation_feature_scale: float = 1.0,
        propagation_use_node_weights: bool = False,
        kind: str = "archetype",
        update_data: bool = True,
    ) -> ArchetypePropagationResult:
        data = self._ensure_data()
        if selected_indexes is None:
            if self.archetype_selection_result is None:
                raise RuntimeError(
                    "No archetype seeds available. Call multimaps.select_archetypes(...) first "
                    "or pass selected_indexes explicitly."
                )
            selected = self.archetype_selection_result.selected_indexes
        else:
            selected = np.asarray(selected_indexes, dtype=np.int64).reshape(-1)
        if selected.size == 0:
            raise RuntimeError("No archetype seeds available for propagation.")

        selected_graph = graph if graph is not None else self.graph_result
        if selected_graph is None:
            raise RuntimeError(
                "No graph available for propagation. Call multimaps.build_graph(...) first "
                "or pass graph explicitly."
            )
        if "global_point_index" not in selected_graph.node_table.columns:
            selected_graph.node_table.loc[:, "global_point_index"] = np.arange(
                len(selected_graph.node_table), dtype=np.int64
            )

        result = propagate_archetypes_on_graph(
            selected_graph,
            selected_indexes=selected,
            point_index_column="global_point_index",
            propagation_mode=propagation_mode,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            propagation_realspace_scale=propagation_realspace_scale,
            propagation_feature_scale=propagation_feature_scale,
            propagation_use_node_weights=propagation_use_node_weights,
        )
        self.archetype_propagation_result = result

        if update_data:
            assignment = result.assignment_table.set_index("global_point_index")
            columns = [
                "assigned_archetype_rank",
                "assigned_archetype_index",
                "archetype_confidence",
                "archetype_margin",
                "is_ambiguous",
            ]
            data.loc[assignment.index, columns] = assignment.loc[:, columns]
            self._propagate_columns(columns)

        valid_assignments = result.assigned_archetype_indexes[
            result.assigned_archetype_indexes >= 0
        ]
        assignment_table = result.assignment_table
        seed_rows = data.loc[selected, ["map_index", "point_index"]].copy()
        for map_index, group in seed_rows.groupby("map_index"):
            local_point_indexes = group["point_index"].to_numpy(dtype=np.int64)
            global_ids = group.index.to_numpy(dtype=np.int64)
            assigned_counts = np.array(
                [int(np.count_nonzero(valid_assignments == global_id)) for global_id in global_ids],
                dtype=np.int64,
            )
            mean_confidences = []
            for global_id in global_ids:
                point_mask = assignment_table["assigned_archetype_index"].to_numpy(
                    dtype=np.int64
                ) == int(global_id)
                mean_confidences.append(
                    float(assignment_table.loc[point_mask, "archetype_confidence"].mean())
                    if np.any(point_mask)
                    else 0.0
                )
            self.maps[int(map_index)].update_special_points(
                kind=kind,
                point_indexes=local_point_indexes,
                assigned_point_count=assigned_counts,
                mean_assignment_confidence=np.asarray(mean_confidences, dtype=np.float64),
            )

        return result

    def _build_dataset(self, recompute: bool) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        reference_features: list[str] | None = None
        self._slices = []

        start = 0
        for index, (name, maps) in enumerate(zip(self.names, self.maps, strict=False)):
            frame = self._get_map_frame(maps, recompute).copy()
            self._attach_map_metadata(index, maps, frame)
            current_features = self._get_map_features(maps)
            if reference_features is None:
                reference_features = current_features
            elif set(current_features) != set(reference_features):
                raise ValueError("All maps in MultiMaps must expose the same feature columns")

            metadata_columns = [
                column
                for column in self._METADATA_COLUMNS
                if column in frame.columns and column not in reference_features
            ]
            ordered = frame.loc[:, ["x", "y", "z", *metadata_columns, *reference_features]].copy()
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
        return [
            column
            for column in maps.data.columns
            if column
            not in {
                "x",
                "y",
                "z",
                *self._METADATA_COLUMNS,
                *getattr(maps, "_METADATA_COLUMNS", set()),
            }
        ]

    def _attach_map_metadata(self, index: int, maps: "Maps", frame: pd.DataFrame) -> None:
        metadata = self.map_metadata[index]
        if not metadata:
            return
        for column, value in metadata.items():
            frame.loc[:, column] = value
            if maps.data is not None:
                maps.data.loc[:, column] = value
        if maps.data is not None and hasattr(maps, "_refresh_features"):
            maps._refresh_features()

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
            if maps.contactspace is None:
                raise RuntimeError("Each Maps instance must define a contact space")
            local_graph = aggregate_cluster_graph(local_labels, maps.contactspace.neighbors)
            nrows, ncols = local_graph.shape
            graph[:nrows, :ncols] += local_graph
        return graph

    def _collect_contactspace_column(self, column: str) -> np.ndarray[Any, np.dtype[np.float64]]:
        values: list[np.ndarray[Any, np.dtype[np.float64]]] = []
        for maps in self.maps:
            if maps.contactspace is None:
                raise RuntimeError("Each Maps instance must define a contact space")
            if column not in maps.contactspace.data.columns:
                raise ValueError(
                    f"node_weight_column {column!r} not present in child contactspace.data."
                )
            values.append(maps.contactspace.data[column].to_numpy(dtype=np.float64))
        return np.concatenate(values).astype(np.float64, copy=False)

    def _register_archetypes_on_children(
        self,
        archetype_table: pd.DataFrame,
        *,
        kind: str,
        iteration: int | None,
        label_status: str,
    ) -> None:
        for map_index, group in archetype_table.groupby("map_index", sort=False):
            ordered = group.sort_values("selection_rank").reset_index(drop=True)
            metadata: dict[str, np.ndarray[Any, Any]] = {
                "selection_rank": ordered["selection_rank"].to_numpy(dtype=np.int64),
                "selection_score": ordered["selection_score"].to_numpy(dtype=np.float64),
                "probability_score": ordered["probability_score"].to_numpy(dtype=np.float64),
                "extremeness_score": ordered["extremeness_score"].to_numpy(dtype=np.float64),
                "diversity_score": ordered["diversity_score"].to_numpy(dtype=np.float64),
            }
            for column in [
                "euclidean_extremeness_score",
                "endpoint_score",
                "endpointness_score",
                "geodesic_score",
                "branching_score",
            ]:
                if column in ordered.columns:
                    metadata[column] = ordered[column].to_numpy(dtype=np.float64)
            self.maps[int(map_index)].add_special_points(
                ordered["point_index"].to_numpy(dtype=np.int64),
                kind=kind,
                iteration=iteration,
                label_status=label_status,
                replace_kind=False,
                **metadata,
            )

    def _combine_contactspace_neighbors(self) -> list[np.ndarray[Any, np.dtype[np.int64]]]:
        combined: list[np.ndarray[Any, np.dtype[np.int64]]] = []
        offset = 0
        for maps in self.maps:
            if maps.contactspace is None:
                raise RuntimeError("Each Maps instance must define a contact space")
            for row in maps.contactspace.neighbors:
                local_neighbors = np.asarray(row, dtype=np.int64).reshape(-1).copy()
                mask = local_neighbors >= 0
                local_neighbors[mask] += offset
                combined.append(local_neighbors)
            offset += len(maps.contactspace.neighbors)
        return combined

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
        file_records = system_parser.file_records()
        system_files = [record.path for record in file_records]

        if len(system_files) > 1 and systemmodel.properties:
            raise NotImplementedError(
                "MultiMapsFromFile does not support per-system properties yet."
            )

        maps_list = []
        for path in system_files:
            system = system_parser.parse_file(path)
            parsed_symmetryfunctions = SymmetryFunctionsParser(
                _namespace_symmetryfunctions(symmetryfunctions)
            ).parse()
            parsed_contactspace = ContactSpaceGenerator(
                _namespace_contactspace(contactspace)
            ).generate(system)
            maps_list.append(Maps(system, parsed_symmetryfunctions, parsed_contactspace))

        super().__init__(
            maps_list,
            names=[Path(path).stem for path in system_files],
            metadata=[record.metadata() for record in file_records],
        )
        self.debug = bool(control.get("debug", False))
        self.verbosity = int(control.get("verbosity", 0))
