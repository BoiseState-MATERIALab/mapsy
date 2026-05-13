from collections.abc import Sequence
from heapq import heappop, heappush
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components, dijkstra
from sklearn.preprocessing import StandardScaler

from mapsy.results import ArchetypePropagationResult, ArchetypeSelectionResult, GraphResult

ArchetypeSelectionMode = Literal["feature_extreme", "graph_endpoint"]
ArchetypePropagationMode = Literal["diffusion", "shortest_path", "watershed", "region_grow"]


def select_archetypes(
    point_table: pd.DataFrame,
    *,
    n_archetypes: int,
    feature_columns: Sequence[str],
    probability_column: str = "probability",
    point_index_column: str = "point_index",
    candidate_mask: npt.ArrayLike | None = None,
    min_probability: float | None = None,
    min_probability_quantile: float | None = 0.75,
    scale_features: bool = True,
    selection_mode: ArchetypeSelectionMode = "feature_extreme",
    graph: GraphResult | None = None,
    probability_weight: float = 1.0,
    extremeness_weight: float = 1.0,
    diversity_weight: float = 1.0,
    endpointness_weight: float = 1.0,
    geodesic_weight: float = 1.0,
    branching_weight: float = 0.5,
) -> ArchetypeSelectionResult:
    if point_table.empty:
        raise ValueError("point_table must contain at least one point.")
    if n_archetypes <= 0:
        raise ValueError(f"n_archetypes must be positive, got {n_archetypes}.")
    if probability_column not in point_table.columns:
        raise ValueError(f"probability_column {probability_column!r} not present in point_table.")
    resolved_feature_columns = list(feature_columns)
    if not resolved_feature_columns:
        raise ValueError("feature_columns must contain at least one column.")
    missing_features = [
        column for column in resolved_feature_columns if column not in point_table.columns
    ]
    if missing_features:
        raise ValueError(f"Missing feature columns in point_table: {missing_features}.")
    if min_probability_quantile is not None and not 0.0 <= min_probability_quantile <= 1.0:
        raise ValueError(
            "min_probability_quantile must lie in [0, 1], " f"got {min_probability_quantile}."
        )

    table = point_table.copy()
    if point_index_column not in table.columns:
        table.loc[:, point_index_column] = table.index.to_numpy(dtype=np.int64)
    point_indexes = table[point_index_column].to_numpy(dtype=np.int64)

    mask = np.ones(len(table), dtype=bool)
    if candidate_mask is not None:
        provided_mask = np.asarray(candidate_mask, dtype=bool).reshape(-1)
        if provided_mask.size != len(table):
            raise ValueError(
                f"candidate_mask has length {provided_mask.size}, expected {len(table)} entries."
            )
        mask &= provided_mask

    probabilities = table[probability_column].to_numpy(dtype=np.float64)
    if min_probability is not None:
        mask &= probabilities >= float(min_probability)
    if min_probability_quantile is not None:
        available = probabilities[mask]
        if available.size == 0:
            raise RuntimeError("No candidate points remain before probability quantile filtering.")
        threshold = float(np.quantile(available, min_probability_quantile))
        mask &= probabilities >= threshold

    candidate_positions = np.where(mask)[0].astype(np.int64, copy=False)
    if candidate_positions.size == 0:
        raise RuntimeError("No candidate points remain for archetype selection.")
    if n_archetypes > candidate_positions.size:
        raise ValueError(
            f"Requested {n_archetypes} archetypes, but only {candidate_positions.size} "
            "candidate points are available."
        )

    candidate_table = table.iloc[candidate_positions].copy().reset_index(drop=True)
    candidate_indexes = point_indexes[candidate_positions]
    candidate_features = candidate_table.loc[:, resolved_feature_columns].to_numpy(dtype=np.float64)
    if scale_features:
        candidate_features = StandardScaler().fit_transform(candidate_features)
    candidate_probabilities = candidate_table[probability_column].to_numpy(dtype=np.float64)
    probability_score = _normalize_component(candidate_probabilities)
    centroid = candidate_features.mean(axis=0)
    euclidean_extremeness_raw = np.linalg.norm(candidate_features - centroid, axis=1)
    euclidean_extremeness_score = _normalize_component(euclidean_extremeness_raw)
    endpoint_score = np.zeros(candidate_positions.size, dtype=np.float64)
    endpointness_score = np.zeros(candidate_positions.size, dtype=np.float64)
    geodesic_score = np.zeros(candidate_positions.size, dtype=np.float64)
    branching_score = np.zeros(candidate_positions.size, dtype=np.float64)
    if selection_mode == "graph_endpoint":
        if graph is None:
            raise ValueError("graph must be provided when selection_mode='graph_endpoint'.")
        endpoint_score, endpointness_score, geodesic_score, branching_score = (
            _compute_graph_endpoint_scores(
                graph,
                point_indexes=candidate_indexes,
                point_index_column=point_index_column,
                feature_vectors=candidate_features,
                endpointness_weight=endpointness_weight,
                geodesic_weight=geodesic_weight,
                branching_weight=branching_weight,
            )
        )
        extremeness_score = endpoint_score
    else:
        extremeness_score = euclidean_extremeness_score

    remaining = np.arange(candidate_positions.size, dtype=np.int64)
    selected_local: list[int] = []
    selected_feature_vectors: list[npt.NDArray[np.float64]] = []
    selection_records: list[dict[str, float | int]] = []

    for rank in range(n_archetypes):
        current_feature_vectors = candidate_features[remaining]
        current_probabilities = probability_score[remaining]
        current_extremeness = extremeness_score[remaining]

        if selected_feature_vectors:
            references = np.vstack(selected_feature_vectors)
            diff = current_feature_vectors[:, None, :] - references[None, :, :]
            min_distance = np.min(np.linalg.norm(diff, axis=2), axis=1)
            diversity_score = _normalize_component(min_distance)
        else:
            diversity_score = current_extremeness.copy()

        total_score = (
            probability_weight * current_probabilities
            + extremeness_weight * current_extremeness
            + diversity_weight * diversity_score
        )
        local_choice = int(np.argmax(total_score))
        selected_position = int(remaining[local_choice])
        selected_local.append(selected_position)
        selected_feature_vectors.append(candidate_features[selected_position].reshape(1, -1))
        selection_records.append(
            {
                point_index_column: int(candidate_indexes[selected_position]),
                "selection_rank": rank,
                "selection_score": float(total_score[local_choice]),
                "probability_score": float(current_probabilities[local_choice]),
                "extremeness_score": float(current_extremeness[local_choice]),
                "diversity_score": float(diversity_score[local_choice]),
            }
        )
        remaining = np.delete(remaining, local_choice)

    archetype_table = candidate_table.loc[selected_local].copy().reset_index(drop=True)
    selection_frame = pd.DataFrame(selection_records)
    archetype_table = archetype_table.merge(selection_frame, on=point_index_column, how="left")
    archetype_table = archetype_table.sort_values("selection_rank").reset_index(drop=True)
    candidate_table.loc[:, "probability_score"] = probability_score
    candidate_table.loc[:, "extremeness_score"] = extremeness_score
    candidate_table.loc[:, "euclidean_extremeness_score"] = euclidean_extremeness_score
    if selection_mode == "graph_endpoint":
        candidate_table.loc[:, "endpoint_score"] = endpoint_score
        candidate_table.loc[:, "endpointness_score"] = endpointness_score
        candidate_table.loc[:, "geodesic_score"] = geodesic_score
        candidate_table.loc[:, "branching_score"] = branching_score

    return ArchetypeSelectionResult(
        feature_columns=resolved_feature_columns,
        probability_column=probability_column,
        candidate_indexes=candidate_indexes.astype(np.int64, copy=False),
        selected_indexes=archetype_table[point_index_column].to_numpy(dtype=np.int64),
        candidate_table=candidate_table,
        archetype_table=archetype_table,
        metadata={
            "point_index_column": point_index_column,
            "min_probability": min_probability,
            "min_probability_quantile": min_probability_quantile,
            "scale_features": scale_features,
            "selection_mode": selection_mode,
            "probability_weight": probability_weight,
            "extremeness_weight": extremeness_weight,
            "diversity_weight": diversity_weight,
            "endpointness_weight": endpointness_weight,
            "geodesic_weight": geodesic_weight,
            "branching_weight": branching_weight,
            "graph_mode": None if graph is None else graph.mode,
        },
    )


def propagate_archetypes(
    graph: GraphResult,
    *,
    selected_indexes: npt.ArrayLike,
    point_index_column: str = "point_index",
    propagation_mode: ArchetypePropagationMode = "diffusion",
    alpha: float = 0.9,
    max_iter: int = 500,
    tol: float = 1.0e-8,
    confidence_threshold: float = 0.5,
    margin_threshold: float = 0.0,
    propagation_realspace_scale: float = 1.0,
    propagation_feature_scale: float = 1.0,
    propagation_use_node_weights: bool = False,
) -> ArchetypePropagationResult:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}.")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}.")
    if point_index_column not in graph.node_table.columns:
        raise ValueError(
            f"point_index_column {point_index_column!r} not present in graph.node_table."
        )

    node_point_indexes = graph.node_table[point_index_column].to_numpy(dtype=np.int64)
    selected = np.asarray(selected_indexes, dtype=np.int64).reshape(-1)
    if selected.size == 0:
        raise ValueError("selected_indexes must contain at least one archetype.")
    if np.unique(selected).size != selected.size:
        raise ValueError("selected_indexes must be unique.")

    point_to_row = {int(point_index): row for row, point_index in enumerate(node_point_indexes)}
    missing = [int(point_index) for point_index in selected if int(point_index) not in point_to_row]
    if missing:
        raise ValueError(f"selected_indexes not present in graph.node_table: {missing}.")

    seed_rows = np.array(
        [point_to_row[int(point_index)] for point_index in selected], dtype=np.int64
    )
    npoints = graph.matrix.shape[0]
    narchetypes = selected.size

    propagation_graph = _reweight_graph(
        graph,
        realspace_scale=propagation_realspace_scale,
        feature_scale=propagation_feature_scale,
        use_node_weights=propagation_use_node_weights,
    )
    target = np.zeros((npoints, narchetypes), dtype=np.float64)
    target[seed_rows, np.arange(narchetypes, dtype=np.int64)] = 1.0

    if propagation_mode == "diffusion":
        normalized_field = _diffusion_assignments(
            propagation_graph.matrix,
            target=target,
            seed_rows=seed_rows,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
        )
    elif propagation_mode == "shortest_path":
        normalized_field = _shortest_path_assignments(
            propagation_graph.matrix,
            seed_rows=seed_rows,
        )
    elif propagation_mode == "watershed":
        normalized_field = _watershed_assignments(
            propagation_graph.matrix,
            seed_rows=seed_rows,
        )
    elif propagation_mode == "region_grow":
        normalized_field = _region_grow_assignments(
            propagation_graph.matrix,
            seed_rows=seed_rows,
        )
    else:
        raise ValueError(f"Unsupported propagation_mode {propagation_mode!r}.")

    positive_mask = normalized_field.sum(axis=1) > 0.0
    assigned_ranks = np.full(npoints, -1, dtype=np.int64)
    confidence = np.zeros(npoints, dtype=np.float64)
    margin = np.zeros(npoints, dtype=np.float64)

    if narchetypes == 1:
        confidence = normalized_field[:, 0].copy()
        margin = confidence.copy()
        assigned_ranks[positive_mask] = 0
    else:
        assigned_ranks = np.argmax(normalized_field, axis=1).astype(np.int64, copy=False)
        confidence = normalized_field[np.arange(npoints), assigned_ranks].astype(
            np.float64, copy=False
        )
        sorted_scores = np.sort(normalized_field, axis=1)
        margin = sorted_scores[:, -1] - sorted_scores[:, -2]

    ambiguous_mask = (
        (~positive_mask) | (confidence < confidence_threshold) | (margin < margin_threshold)
    )
    assigned_ranks = assigned_ranks.astype(np.int64, copy=True)
    assigned_ranks[ambiguous_mask] = -1
    assigned_indexes = np.full(npoints, -1, dtype=np.int64)
    valid_mask = assigned_ranks >= 0
    assigned_indexes[valid_mask] = selected[assigned_ranks[valid_mask]]

    assignment_table = graph.node_table.copy()
    assignment_table.loc[:, "assigned_archetype_rank"] = assigned_ranks
    assignment_table.loc[:, "assigned_archetype_index"] = assigned_indexes
    assignment_table.loc[:, "archetype_confidence"] = confidence
    assignment_table.loc[:, "archetype_margin"] = margin
    assignment_table.loc[:, "is_ambiguous"] = ambiguous_mask
    for rank in range(narchetypes):
        assignment_table.loc[:, f"archetype_score_{rank}"] = normalized_field[:, rank]

    return ArchetypePropagationResult(
        selected_indexes=selected,
        assigned_archetype_ranks=assigned_ranks,
        assigned_archetype_indexes=assigned_indexes,
        confidence=confidence,
        margin=margin,
        soft_assignments=normalized_field,
        assignment_table=assignment_table,
        metadata={
            "point_index_column": point_index_column,
            "propagation_mode": propagation_mode,
            "alpha": alpha,
            "max_iter": max_iter,
            "tol": tol,
            "confidence_threshold": confidence_threshold,
            "margin_threshold": margin_threshold,
            "propagation_realspace_scale": propagation_realspace_scale,
            "propagation_feature_scale": propagation_feature_scale,
            "propagation_use_node_weights": propagation_use_node_weights,
        },
    )


def _normalize_component(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if values.size == 0:
        return values.astype(np.float64, copy=True)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmax, vmin):
        return np.ones_like(values, dtype=np.float64)
    return (values - vmin) / (vmax - vmin)


def _compute_graph_endpoint_scores(
    graph: GraphResult,
    *,
    point_indexes: npt.NDArray[np.int64],
    point_index_column: str,
    feature_vectors: npt.NDArray[np.float64],
    endpointness_weight: float,
    geodesic_weight: float,
    branching_weight: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    if point_index_column not in graph.node_table.columns:
        raise ValueError(
            f"point_index_column {point_index_column!r} not present in graph.node_table."
        )
    graph_point_indexes = graph.node_table[point_index_column].to_numpy(dtype=np.int64)
    point_to_row = {int(point_index): row for row, point_index in enumerate(graph_point_indexes)}
    missing = [
        int(point_index) for point_index in point_indexes if int(point_index) not in point_to_row
    ]
    if missing:
        raise ValueError(f"candidate point indexes not present in graph.node_table: {missing}.")

    candidate_rows = np.array(
        [point_to_row[int(point_index)] for point_index in point_indexes], dtype=np.int64
    )
    candidate_graph = graph.matrix[candidate_rows][:, candidate_rows].tocsr()
    degree = np.asarray(candidate_graph.sum(axis=1), dtype=np.float64).reshape(-1)
    branching_score = 1.0 - _normalize_component(degree)

    endpointness_score = np.zeros(candidate_graph.shape[0], dtype=np.float64)
    for row in range(candidate_graph.shape[0]):
        start = int(candidate_graph.indptr[row])
        stop = int(candidate_graph.indptr[row + 1])
        neighbors = candidate_graph.indices[start:stop]
        weights = candidate_graph.data[start:stop]
        if neighbors.size == 0:
            continue
        directions = feature_vectors[neighbors] - feature_vectors[row]
        norms = np.linalg.norm(directions, axis=1)
        valid = norms > 0.0
        if not np.any(valid):
            continue
        unit_directions = directions[valid] / norms[valid, None]
        normalized_weights = weights[valid] / np.sum(weights[valid])
        resultant = np.sum(unit_directions * normalized_weights[:, None], axis=0)
        endpointness_score[row] = float(np.linalg.norm(resultant))

    geodesic_score = np.zeros(candidate_graph.shape[0], dtype=np.float64)
    ncomponents, labels = connected_components(candidate_graph, directed=False, return_labels=True)
    for component in range(ncomponents):
        component_rows = np.where(labels == component)[0]
        if component_rows.size == 0:
            continue
        if component_rows.size == 1:
            geodesic_score[component_rows[0]] = 1.0
            continue
        component_degree = degree[component_rows]
        core_local = int(np.argmax(component_degree))
        component_graph = candidate_graph[component_rows][:, component_rows].copy().tocsr()
        if component_graph.nnz == 0:
            geodesic_score[component_rows] = 1.0
            continue
        component_graph.data = 1.0 / np.maximum(component_graph.data, 1.0e-12)
        distances = dijkstra(component_graph, directed=False, indices=core_local)
        finite = np.isfinite(distances)
        component_score = np.zeros(component_rows.size, dtype=np.float64)
        if np.any(finite):
            component_score[finite] = _normalize_component(distances[finite])
        geodesic_score[component_rows] = component_score

    endpoint_raw = (
        endpointness_weight * endpointness_score
        + geodesic_weight * geodesic_score
        + branching_weight * branching_score
    )
    endpoint_score = _normalize_component(endpoint_raw)
    return endpoint_score, endpointness_score, geodesic_score, branching_score


def _row_normalize(matrix: csr_matrix) -> csr_matrix:
    row_sums = np.asarray(matrix.sum(axis=1), dtype=np.float64).reshape(-1)
    inv = np.zeros_like(row_sums, dtype=np.float64)
    positive = row_sums > 0.0
    inv[positive] = 1.0 / row_sums[positive]
    return diags(inv).dot(matrix).tocsr()


def _diffusion_assignments(
    matrix: csr_matrix,
    *,
    target: npt.NDArray[np.float64],
    seed_rows: npt.NDArray[np.int64],
    alpha: float,
    max_iter: int,
    tol: float,
) -> npt.NDArray[np.float64]:
    normalized_matrix = _row_normalize(matrix)
    field = target.copy()
    for _ in range(max_iter):
        updated = alpha * normalized_matrix.dot(field) + (1.0 - alpha) * target
        updated[seed_rows, :] = target[seed_rows, :]
        delta = float(np.max(np.abs(updated - field)))
        field = updated
        if delta < tol:
            break
    row_sums = field.sum(axis=1)
    normalized_field = np.zeros_like(field)
    positive_mask = row_sums > 0.0
    normalized_field[positive_mask] = field[positive_mask] / row_sums[positive_mask, None]
    return normalized_field


def _shortest_path_assignments(
    matrix: csr_matrix,
    *,
    seed_rows: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    if matrix.nnz == 0:
        return np.zeros((matrix.shape[0], seed_rows.size), dtype=np.float64)
    cost_matrix = matrix.copy().tocsr()
    cost_matrix.data = 1.0 / np.maximum(cost_matrix.data, 1.0e-12)
    distances = dijkstra(cost_matrix, directed=False, indices=seed_rows)
    if distances.ndim == 1:
        distances = distances[np.newaxis, :]
    scores = np.zeros((matrix.shape[0], seed_rows.size), dtype=np.float64)
    finite = np.isfinite(distances)
    scores_t = scores.T
    scores_t[finite] = 1.0 / (1.0 + distances[finite])
    scores = scores_t.T
    scores[seed_rows, :] = 0.0
    scores[seed_rows, np.arange(seed_rows.size, dtype=np.int64)] = 1.0
    return _normalize_rows(scores)


def _watershed_assignments(
    matrix: csr_matrix,
    *,
    seed_rows: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    nscores = seed_rows.size
    scores = np.zeros((matrix.shape[0], nscores), dtype=np.float64)
    closeness = _shortest_path_assignments(matrix, seed_rows=seed_rows)
    for column, seed in enumerate(seed_rows):
        scores[:, column] = _widest_path_scores(matrix, int(seed))
    # Tie-break equal-capacity paths by ordinary closeness to the seed.
    scores = scores + 1.0e-6 * closeness
    scores[seed_rows, :] = 0.0
    scores[seed_rows, np.arange(seed_rows.size, dtype=np.int64)] = 1.0
    return _normalize_rows(scores)


def _region_grow_assignments(
    matrix: csr_matrix,
    *,
    seed_rows: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    if matrix.nnz == 0:
        return np.zeros((matrix.shape[0], seed_rows.size), dtype=np.float64)

    adjacency = matrix.copy().tocsr()
    adjacency.data = np.ones_like(adjacency.data, dtype=np.float64)
    hop_distances = dijkstra(adjacency, directed=False, indices=seed_rows)
    if hop_distances.ndim == 1:
        hop_distances = hop_distances[np.newaxis, :]

    closeness = _shortest_path_assignments(matrix, seed_rows=seed_rows)
    scores = np.zeros((matrix.shape[0], seed_rows.size), dtype=np.float64)

    for node in range(matrix.shape[0]):
        hop_column = hop_distances[:, node]
        finite = np.isfinite(hop_column)
        if not np.any(finite):
            continue
        best_hop = float(np.min(hop_column[finite]))
        contenders = np.where(np.isclose(hop_column, best_hop))[0]
        if contenders.size == 1:
            scores[node, int(contenders[0])] = 1.0
            continue
        tie_weights = closeness[node, contenders]
        if np.allclose(tie_weights.sum(), 0.0):
            scores[node, contenders] = 1.0 / contenders.size
        else:
            scores[node, contenders] = tie_weights / tie_weights.sum()

    scores[seed_rows, :] = 0.0
    scores[seed_rows, np.arange(seed_rows.size, dtype=np.int64)] = 1.0
    return scores


def _widest_path_scores(matrix: csr_matrix, seed_row: int) -> npt.NDArray[np.float64]:
    npoints = matrix.shape[0]
    scores = np.zeros(npoints, dtype=np.float64)
    scores[seed_row] = np.inf
    heap: list[tuple[float, int]] = [(-scores[seed_row], seed_row)]
    while heap:
        neg_score, node = heappop(heap)
        current_score = -neg_score
        if current_score < scores[node]:
            continue
        start = int(matrix.indptr[node])
        stop = int(matrix.indptr[node + 1])
        neighbors = matrix.indices[start:stop]
        weights = matrix.data[start:stop]
        for neighbor, weight in zip(neighbors, weights, strict=False):
            path_score = (
                float(weight) if np.isinf(current_score) else min(current_score, float(weight))
            )
            if path_score > scores[int(neighbor)]:
                scores[int(neighbor)] = path_score
                heappush(heap, (-path_score, int(neighbor)))
    finite = np.isfinite(scores)
    finite_scores = scores[finite]
    if finite_scores.size == 0:
        return np.zeros_like(scores)
    max_finite = (
        float(np.max(finite_scores[~np.isinf(finite_scores)]))
        if np.any(~np.isinf(finite_scores))
        else 1.0
    )
    normalized = scores.copy()
    if np.isinf(normalized[seed_row]):
        normalized[seed_row] = max(max_finite, 1.0)
    return normalized


def _normalize_rows(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    row_sums = scores.sum(axis=1)
    normalized = np.zeros_like(scores)
    positive = row_sums > 0.0
    normalized[positive] = scores[positive] / row_sums[positive, None]
    return normalized


def _reweight_graph(
    graph: GraphResult,
    *,
    realspace_scale: float,
    feature_scale: float,
    use_node_weights: bool,
) -> GraphResult:
    if realspace_scale < 0.0 or feature_scale < 0.0:
        raise ValueError("Propagation graph scales must be non-negative.")
    edge_table = graph.edge_table.copy()
    if {
        "realspace_component",
        "feature_component",
        "directional_factor",
        "node_factor",
    }.issubset(edge_table.columns):
        node_factor = (
            edge_table["node_factor"].to_numpy(dtype=np.float64)
            if use_node_weights
            else np.ones(len(edge_table), dtype=np.float64)
        )
        edge_table.loc[:, "weight"] = (
            edge_table["realspace_component"].to_numpy(dtype=np.float64)
            * edge_table["directional_factor"].to_numpy(dtype=np.float64)
            * realspace_scale
            + edge_table["feature_component"].to_numpy(dtype=np.float64) * feature_scale
        ) * node_factor
    else:
        edge_table.loc[:, "weight"] = edge_table["weight"].to_numpy(dtype=np.float64)
    edge_table = edge_table.loc[edge_table["weight"] > 0.0].reset_index(drop=True)
    matrix = _sparse_matrix_from_edge_table(edge_table, npoints=graph.matrix.shape[0])
    metadata = dict(graph.metadata or {})
    metadata["propagation_realspace_scale"] = realspace_scale
    metadata["propagation_feature_scale"] = feature_scale
    metadata["propagation_use_node_weights"] = use_node_weights
    return GraphResult(
        mode=graph.mode,
        feature_columns=list(graph.feature_columns),
        node_weight_column=graph.node_weight_column,
        node_table=graph.node_table.copy(),
        node_weights=graph.node_weights.copy(),
        edge_table=edge_table,
        matrix=matrix,
        metadata=metadata,
    )


def _sparse_matrix_from_edge_table(
    edge_table: pd.DataFrame,
    *,
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
