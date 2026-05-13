from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from mapsy.results import GraphResult

GraphMode: TypeAlias = Literal["realspace", "feature", "hybrid"]
FeatureConnectivityMode: TypeAlias = Literal["knn", "knn_mst"]


def build_point_graph(
    node_table: pd.DataFrame,
    *,
    mode: GraphMode = "hybrid",
    feature_columns: list[str] | None = None,
    neighbors: list[npt.NDArray[np.int64]] | npt.NDArray[np.int64] | None = None,
    node_weight_column: str = "probability",
    feature_k: int = 8,
    feature_connectivity: FeatureConnectivityMode = "knn",
    sigma_feature: float | None = None,
    realspace_weight: float = 1.0,
    feature_weight: float = 1.0,
    connect_systems: bool = True,
    system_ids: npt.ArrayLike | None = None,
    normalize_node_weights: bool = True,
    use_node_weights_in_edges: bool = True,
    direction_columns: tuple[str, str, str] | None = None,
    directional_weight: float = 0.0,
    directional_power: float = 1.0,
) -> GraphResult:
    if node_table.empty:
        raise ValueError("node_table must contain at least one point.")
    if mode in {"realspace", "hybrid"} and neighbors is None:
        raise ValueError(f"neighbors must be provided for graph mode {mode!r}.")
    if mode in {"feature", "hybrid"} and not feature_columns:
        raise ValueError(f"feature_columns must be provided for graph mode {mode!r}.")
    if feature_k < 1:
        raise ValueError(f"feature_k must be positive, got {feature_k}.")
    if node_weight_column not in node_table.columns:
        raise ValueError(f"node_weight_column {node_weight_column!r} not present in node_table.")
    if directional_weight < 0.0:
        raise ValueError(f"directional_weight must be non-negative, got {directional_weight}.")
    if directional_power <= 0.0:
        raise ValueError(f"directional_power must be positive, got {directional_power}.")

    feature_columns = list(feature_columns or [])
    system_array = (
        np.asarray(system_ids).reshape(-1)
        if system_ids is not None
        else np.zeros(len(node_table), dtype=np.int64)
    )
    if system_array.size != len(node_table):
        raise ValueError(
            f"system_ids has length {system_array.size}, expected {len(node_table)} entries."
        )

    raw_node_weights = node_table[node_weight_column].to_numpy(dtype=np.float64)
    node_weights = _normalize_node_weights(raw_node_weights, normalize=normalize_node_weights)
    positions = node_table.loc[:, ["x", "y", "z"]].to_numpy(dtype=np.float64)
    direction_vectors = _extract_direction_vectors(
        node_table,
        direction_columns=direction_columns,
        directional_weight=directional_weight,
    )
    edge_accumulator: dict[tuple[int, int], dict[str, float]] = {}

    if mode in {"realspace", "hybrid"} and neighbors is not None:
        _accumulate_realspace_edges(
            edge_accumulator,
            neighbors=neighbors,
            realspace_weight=realspace_weight,
        )

    if mode in {"feature", "hybrid"} and feature_columns:
        features = node_table.loc[:, feature_columns].to_numpy(dtype=np.float64)
        _accumulate_feature_edges(
            edge_accumulator,
            features=features,
            feature_k=feature_k,
            feature_connectivity=feature_connectivity,
            sigma_feature=sigma_feature,
            feature_weight=feature_weight,
            system_ids=system_array,
            connect_systems=connect_systems,
        )

    edge_table = _build_edge_table(
        edge_accumulator,
        node_weights=node_weights,
        positions=positions,
        direction_vectors=direction_vectors,
        use_node_weights_in_edges=use_node_weights_in_edges,
        directional_weight=directional_weight,
        directional_power=directional_power,
    )
    matrix = _build_sparse_matrix(
        edge_table=edge_table,
        npoints=len(node_table),
    )

    node_frame = node_table.copy()
    node_frame.loc[:, "graph_node_weight"] = node_weights
    return GraphResult(
        mode=mode,
        feature_columns=feature_columns,
        node_weight_column=node_weight_column,
        node_table=node_frame,
        node_weights=node_weights,
        edge_table=edge_table,
        matrix=matrix,
        metadata={
            "connect_systems": connect_systems,
            "normalize_node_weights": normalize_node_weights,
            "use_node_weights_in_edges": use_node_weights_in_edges,
            "realspace_weight": realspace_weight,
            "feature_weight": feature_weight,
            "feature_k": feature_k,
            "feature_connectivity": feature_connectivity,
            "sigma_feature": sigma_feature,
            "direction_columns": direction_columns,
            "directional_weight": directional_weight,
            "directional_power": directional_power,
        },
    )


def _normalize_node_weights(
    values: npt.NDArray[np.float64],
    *,
    normalize: bool,
) -> npt.NDArray[np.float64]:
    if not normalize:
        return values.astype(np.float64, copy=True)
    vmax = float(np.max(values))
    if vmax <= 0.0:
        return np.ones_like(values, dtype=np.float64)
    return values.astype(np.float64, copy=True) / vmax


def _accumulate_realspace_edges(
    edge_accumulator: dict[tuple[int, int], dict[str, float]],
    *,
    neighbors: list[npt.NDArray[np.int64]] | npt.NDArray[np.int64],
    realspace_weight: float,
) -> None:
    for source, row in enumerate(neighbors):
        local_neighbors = np.asarray(row, dtype=np.int64).reshape(-1)
        for target in local_neighbors[local_neighbors >= 0]:
            i = int(source)
            j = int(target)
            if i >= j:
                continue
            key = (i, j)
            payload = edge_accumulator.setdefault(
                key,
                {"realspace_component": 0.0, "feature_component": 0.0},
            )
            payload["realspace_component"] += float(realspace_weight)


def _accumulate_feature_edges(
    edge_accumulator: dict[tuple[int, int], dict[str, float]],
    *,
    features: npt.NDArray[np.float64],
    feature_k: int,
    feature_connectivity: FeatureConnectivityMode,
    sigma_feature: float | None,
    feature_weight: float,
    system_ids: npt.NDArray[Any],
    connect_systems: bool,
) -> None:
    if connect_systems:
        _accumulate_feature_edges_subset(
            edge_accumulator,
            features=features,
            indices=np.arange(len(features), dtype=np.int64),
            feature_k=feature_k,
            feature_connectivity=feature_connectivity,
            sigma_feature=sigma_feature,
            feature_weight=feature_weight,
        )
        return

    for system_id in np.unique(system_ids):
        indexes = np.where(system_ids == system_id)[0].astype(np.int64, copy=False)
        if indexes.size < 2:
            continue
        _accumulate_feature_edges_subset(
            edge_accumulator,
            features=features[indexes],
            indices=indexes,
            feature_k=feature_k,
            feature_connectivity=feature_connectivity,
            sigma_feature=sigma_feature,
            feature_weight=feature_weight,
        )


def _accumulate_feature_edges_subset(
    edge_accumulator: dict[tuple[int, int], dict[str, float]],
    *,
    features: npt.NDArray[np.float64],
    indices: npt.NDArray[np.int64],
    feature_k: int,
    feature_connectivity: FeatureConnectivityMode,
    sigma_feature: float | None,
    feature_weight: float,
) -> None:
    npoints = len(indices)
    if npoints < 2:
        return

    nneighbors = min(feature_k + 1, npoints)
    model = NearestNeighbors(n_neighbors=nneighbors)
    model.fit(features)
    distances, neighbor_indices = model.kneighbors(features)

    positive_distances = distances[:, 1:][distances[:, 1:] > 0.0]
    if sigma_feature is None:
        sigma = float(np.median(positive_distances)) if positive_distances.size else 1.0
    else:
        sigma = float(sigma_feature)
    if sigma <= 0.0:
        sigma = 1.0

    for local_source, global_source in enumerate(indices):
        for local_target, distance in zip(
            neighbor_indices[local_source, 1:],
            distances[local_source, 1:],
            strict=False,
        ):
            global_target = int(indices[int(local_target)])
            if int(global_source) == global_target:
                continue
            key = (
                (int(global_source), global_target)
                if int(global_source) < global_target
                else (global_target, int(global_source))
            )
            payload = edge_accumulator.setdefault(
                key,
                {"realspace_component": 0.0, "feature_component": 0.0},
            )
            similarity = np.exp(-float(distance) ** 2 / (sigma**2))
            payload["feature_component"] += float(feature_weight) * float(similarity)

    if feature_connectivity == "knn_mst":
        _accumulate_feature_mst_edges(
            edge_accumulator,
            features=features,
            indices=indices,
            sigma=sigma,
            feature_weight=feature_weight,
        )


def _accumulate_feature_mst_edges(
    edge_accumulator: dict[tuple[int, int], dict[str, float]],
    *,
    features: npt.NDArray[np.float64],
    indices: npt.NDArray[np.int64],
    sigma: float,
    feature_weight: float,
) -> None:
    npoints = len(indices)
    if npoints < 2:
        return
    distances = cdist(features, features, metric="euclidean")
    mst = minimum_spanning_tree(distances).tocoo()
    for local_source, local_target, distance in zip(
        mst.row,
        mst.col,
        mst.data,
        strict=False,
    ):
        global_source = int(indices[int(local_source)])
        global_target = int(indices[int(local_target)])
        if global_source == global_target:
            continue
        key = (
            (global_source, global_target)
            if global_source < global_target
            else (global_target, global_source)
        )
        payload = edge_accumulator.setdefault(
            key,
            {"realspace_component": 0.0, "feature_component": 0.0},
        )
        similarity = max(float(np.exp(-float(distance) ** 2 / (sigma**2))), 1.0e-12)
        payload["feature_component"] += float(feature_weight) * float(similarity)


def _build_edge_table(
    edge_accumulator: dict[tuple[int, int], dict[str, float]],
    *,
    node_weights: npt.NDArray[np.float64],
    positions: npt.NDArray[np.float64],
    direction_vectors: npt.NDArray[np.float64] | None,
    use_node_weights_in_edges: bool,
    directional_weight: float,
    directional_power: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for (source, target), components in edge_accumulator.items():
        realspace_component = float(components["realspace_component"])
        feature_component = float(components["feature_component"])
        base_weight = realspace_component + feature_component
        if base_weight <= 0.0:
            continue
        directional_alignment = _directional_alignment(
            source,
            target,
            positions=positions,
            direction_vectors=direction_vectors,
        )
        directional_factor = 1.0 + directional_weight * (directional_alignment**directional_power)
        weighted_realspace = realspace_component * directional_factor
        node_factor = (
            float(np.sqrt(node_weights[source] * node_weights[target]))
            if use_node_weights_in_edges
            else 1.0
        )
        rows.append(
            {
                "source": source,
                "target": target,
                "realspace_component": realspace_component,
                "feature_component": feature_component,
                "directional_alignment": directional_alignment,
                "directional_factor": directional_factor,
                "node_factor": node_factor,
                "weight": (weighted_realspace + feature_component) * node_factor,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "realspace_component",
                "feature_component",
                "directional_alignment",
                "directional_factor",
                "node_factor",
                "weight",
            ]
        )
    return pd.DataFrame(rows).sort_values(["source", "target"]).reset_index(drop=True)


def _build_sparse_matrix(
    *,
    edge_table: pd.DataFrame,
    npoints: int,
) -> csr_matrix:
    if edge_table.empty:
        return csr_matrix((npoints, npoints), dtype=np.float64)
    sources = edge_table["source"].to_numpy(dtype=np.int64)
    targets = edge_table["target"].to_numpy(dtype=np.int64)
    weights = edge_table["weight"].to_numpy(dtype=np.float64)
    rows = np.concatenate([sources, targets])
    cols = np.concatenate([targets, sources])
    vals = np.concatenate([weights, weights])
    return csr_matrix((vals, (rows, cols)), shape=(npoints, npoints), dtype=np.float64)


def _extract_direction_vectors(
    node_table: pd.DataFrame,
    *,
    direction_columns: tuple[str, str, str] | None,
    directional_weight: float,
) -> npt.NDArray[np.float64] | None:
    if directional_weight == 0.0:
        return None
    columns = direction_columns or (
        "boundary_gradient_x",
        "boundary_gradient_y",
        "boundary_gradient_z",
    )
    missing = [column for column in columns if column not in node_table.columns]
    if missing:
        raise ValueError(
            "directional weighting requires direction columns " f"{columns}, missing {missing}."
        )
    return node_table.loc[:, list(columns)].to_numpy(dtype=np.float64)


def _directional_alignment(
    source: int,
    target: int,
    *,
    positions: npt.NDArray[np.float64],
    direction_vectors: npt.NDArray[np.float64] | None,
) -> float:
    if direction_vectors is None:
        return 0.0
    displacement = positions[target] - positions[source]
    displacement_norm = float(np.linalg.norm(displacement))
    if displacement_norm == 0.0:
        return 0.0
    edge_direction = displacement / displacement_norm

    source_direction = _normalize_vector(direction_vectors[source])
    target_direction = _normalize_vector(direction_vectors[target])
    combined = source_direction + target_direction
    combined_norm = float(np.linalg.norm(combined))
    reference = (
        combined / combined_norm
        if combined_norm > 0.0
        else (
            source_direction if float(np.linalg.norm(source_direction)) > 0.0 else target_direction
        )
    )
    reference_norm = float(np.linalg.norm(reference))
    if reference_norm == 0.0:
        return 0.0
    return float(abs(np.dot(edge_direction, reference / reference_norm)))


def _normalize_vector(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return np.zeros_like(vector, dtype=np.float64)
    return vector / norm
