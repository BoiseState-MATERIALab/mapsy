from heapq import heappop, heappush

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from mapsy.analysis.archetypes import (
    ArchetypePropagationMode,
    _diffusion_assignments,
    _normalize_rows,
    _reweight_graph,
)
from mapsy.clustering import (
    clustering_uses_random_state,
    fit_predict_clusters,
    normalize_cluster_method,
)
from mapsy.results import ClusterResult, ClusterScreeningResult, GraphResult


def fit_clusters(
    X: npt.NDArray[np.float64],
    *,
    feature_columns: list[str],
    nclusters: int,
    method: str = "spectral",
    random_state: int | None = None,
    scale: bool = False,
    screening: ClusterScreeningResult | None = None,
    graph: GraphResult | None = None,
) -> ClusterResult:
    normalized_method = normalize_cluster_method(method)
    scaler = StandardScaler() if scale else None
    X_scaled = scaler.fit_transform(X) if scaler is not None else X
    effective_random_state = (
        random_state if clustering_uses_random_state(normalized_method) else None
    )
    labels = fit_predict_clusters(
        X_scaled,
        method=normalized_method,
        nclusters=nclusters,
        random_state=effective_random_state,
        graph_matrix=graph.matrix if graph is not None else None,
    )
    labels = _renumber_labels_by_cluster_center(X, labels)
    centers = _cluster_centers(X, labels, feature_columns)
    sizes = _cluster_sizes(labels)
    return ClusterResult(
        method=normalized_method,
        feature_columns=list(feature_columns),
        scale=scale,
        nclusters=nclusters,
        labels=labels,
        centers=centers,
        sizes=sizes,
        random_state=effective_random_state,
        screening=screening,
        metadata={"scaler": scaler, "graph_mode": None if graph is None else graph.mode},
    )


def screen_clusters(
    X: npt.NDArray[np.float64],
    *,
    feature_columns: list[str],
    method: str = "spectral",
    maxclusters: int = 20,
    ntries: int = 1,
    scale: bool = False,
    graph: GraphResult | None = None,
) -> ClusterScreeningResult:
    normalized_method = normalize_cluster_method(method)
    scaler = StandardScaler() if scale else None
    X_scaled = scaler.fit_transform(X) if scaler is not None else X

    cluster_random_states: list[int] = []
    cluster_sizes: list[int] = []
    silhouette_scores: list[float] = []
    db_indexes: list[float] = []

    for nclusters in range(2, maxclusters):
        if clustering_uses_random_state(normalized_method):
            random_states = np.random.randint(0, 1000, ntries)
        else:
            random_states = np.array([0], dtype=np.int64)
        for random_state in random_states:
            effective_random_state = (
                int(random_state) if clustering_uses_random_state(normalized_method) else None
            )
            labels = fit_predict_clusters(
                X_scaled,
                method=normalized_method,
                nclusters=nclusters,
                random_state=effective_random_state,
                graph_matrix=graph.matrix if graph is not None else None,
            )
            actual_nclusters = int(len(np.unique(labels)))
            cluster_random_states.append(int(random_state))
            cluster_sizes.append(actual_nclusters)
            silhouette_scores.append(float(silhouette_score(X_scaled, labels)))
            db_indexes.append(float(davies_bouldin_score(X_scaled, labels)))

    table = pd.DataFrame(
        {
            "method": normalized_method,
            "nclusters": cluster_sizes,
            "random_state": cluster_random_states,
            "silhouette_score": silhouette_scores,
            "db_index": db_indexes,
        }
    )
    best_by_db = table.loc[table.groupby("nclusters")["db_index"].idxmin()].copy()
    best_by_silhouette = table.loc[table.groupby("nclusters")["silhouette_score"].idxmax()].copy()
    return ClusterScreeningResult(
        method=normalized_method,
        feature_columns=list(feature_columns),
        scale=scale,
        table=table,
        best_by_db=best_by_db,
        best_by_silhouette=best_by_silhouette,
    )


def _cluster_centers(
    X: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
    feature_columns: list[str],
) -> npt.NDArray[np.float64]:
    frame = pd.DataFrame(X, columns=feature_columns)
    frame.loc[:, "Cluster"] = labels
    return frame.groupby("Cluster")[feature_columns].mean().to_numpy(dtype=np.float64)


def _renumber_labels_by_cluster_center(
    X: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    unique_labels = np.unique(labels)
    centers = np.vstack([np.mean(X[labels == label], axis=0) for label in unique_labels]).astype(
        np.float64, copy=False
    )
    sort_keys = tuple(centers[:, column] for column in range(centers.shape[1] - 1, -1, -1))
    label_order = np.lexsort(sort_keys)
    remap = {
        int(unique_labels[old_index]): int(new_label)
        for new_label, old_index in enumerate(label_order)
    }
    return np.array([remap[int(label)] for label in labels], dtype=np.int64)


def _cluster_sizes(labels: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    _, counts = np.unique(labels, return_counts=True)
    return counts.astype(np.int64, copy=False)


def aggregate_cluster_graph(
    labels: npt.NDArray[np.int64],
    neighbors: list[npt.NDArray[np.int64]] | npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    valid_labels = labels[labels >= 0]
    if valid_labels.size == 0:
        return np.zeros((0, 0), dtype=np.int64)
    nclusters = int(np.max(valid_labels)) + 1
    graph = np.zeros((nclusters, nclusters), dtype=np.int64)
    for i, ci in enumerate(labels):
        if ci < 0:
            continue
        local_neighbors = np.asarray(neighbors[i], dtype=np.int64).reshape(-1)
        for j in local_neighbors[local_neighbors >= 0]:
            cj = labels[int(j)]
            if cj < 0:
                continue
            graph[int(ci), int(cj)] += 1
            graph[int(cj), int(ci)] += 1
    return graph // 2


def propagate_cluster_labels(
    graph: GraphResult,
    *,
    seed_mask: npt.ArrayLike,
    seed_labels: npt.ArrayLike,
    propagation_mode: ArchetypePropagationMode = "diffusion",
    alpha: float = 0.9,
    max_iter: int = 500,
    tol: float = 1.0e-8,
    confidence_threshold: float = 0.0,
    margin_threshold: float = 0.0,
    propagation_realspace_scale: float = 1.0,
    propagation_feature_scale: float = 1.0,
    propagation_use_node_weights: bool = False,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
    npt.NDArray[np.float64],
]:
    seed_mask_array = np.asarray(seed_mask, dtype=bool).reshape(-1)
    labels_array = np.asarray(seed_labels, dtype=np.int64).reshape(-1)
    npoints = graph.matrix.shape[0]
    if seed_mask_array.size != npoints:
        raise ValueError(f"seed_mask has length {seed_mask_array.size}, expected {npoints}.")
    if labels_array.size != npoints:
        raise ValueError(f"seed_labels has length {labels_array.size}, expected {npoints}.")

    seed_rows = np.flatnonzero(seed_mask_array).astype(np.int64, copy=False)
    if seed_rows.size == 0:
        raise ValueError("seed_mask must select at least one seed point.")
    selected_labels = labels_array[seed_rows]
    if np.any(selected_labels < 0):
        raise ValueError("seed_labels must be non-negative on the selected seed rows.")

    nclusters = int(np.max(selected_labels)) + 1
    propagation_graph = _reweight_graph(
        graph,
        realspace_scale=propagation_realspace_scale,
        feature_scale=propagation_feature_scale,
        use_node_weights=propagation_use_node_weights,
    )

    if propagation_mode == "diffusion":
        target = np.zeros((npoints, nclusters), dtype=np.float64)
        target[seed_rows, selected_labels] = 1.0
        normalized_field = _diffusion_assignments(
            propagation_graph.matrix,
            target=target,
            seed_rows=seed_rows,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
        )
    else:
        if propagation_mode == "shortest_path":
            normalized_field = _cluster_shortest_path_assignments(
                propagation_graph.matrix,
                seed_rows=seed_rows,
                seed_labels=selected_labels,
                nclusters=nclusters,
            )
        elif propagation_mode == "watershed":
            normalized_field = _cluster_watershed_assignments(
                propagation_graph.matrix,
                seed_rows=seed_rows,
                seed_labels=selected_labels,
                nclusters=nclusters,
            )
        elif propagation_mode == "region_grow":
            normalized_field = _cluster_region_grow_assignments(
                propagation_graph.matrix,
                seed_rows=seed_rows,
                seed_labels=selected_labels,
                nclusters=nclusters,
            )
        else:
            raise ValueError(f"Unsupported propagation_mode {propagation_mode!r}.")

    positive_mask = normalized_field.sum(axis=1) > 0.0
    confidence = np.zeros(npoints, dtype=np.float64)
    margin = np.zeros(npoints, dtype=np.float64)
    assigned_labels = np.full(npoints, -1, dtype=np.int64)

    if nclusters == 1:
        confidence = normalized_field[:, 0].copy()
        margin = confidence.copy()
        assigned_labels[positive_mask] = 0
    else:
        best_labels = np.argmax(normalized_field, axis=1).astype(np.int64, copy=False)
        confidence = normalized_field[np.arange(npoints), best_labels].astype(
            np.float64, copy=False
        )
        sorted_scores = np.sort(normalized_field, axis=1)
        margin = sorted_scores[:, -1] - sorted_scores[:, -2]
        assigned_labels[positive_mask] = best_labels[positive_mask]

    ambiguous_mask = (
        (~positive_mask) | (confidence < confidence_threshold) | (margin < margin_threshold)
    )
    assigned_labels[ambiguous_mask] = -1
    return assigned_labels, confidence, margin, ambiguous_mask, normalized_field


def _cluster_shortest_path_assignments(
    matrix: csr_matrix,
    *,
    seed_rows: npt.NDArray[np.int64],
    seed_labels: npt.NDArray[np.int64],
    nclusters: int,
) -> npt.NDArray[np.float64]:
    scores = np.zeros((matrix.shape[0], nclusters), dtype=np.float64)
    if matrix.nnz == 0:
        return _normalize_rows(_set_seed_identity(scores, seed_rows, seed_labels))

    cost_matrix = matrix.copy().tocsr()
    cost_matrix.data = 1.0 / np.maximum(cost_matrix.data, 1.0e-12)
    for cluster_id in range(nclusters):
        distances = _multi_seed_distances(
            cost_matrix,
            seed_rows[seed_labels == cluster_id],
        )
        finite = np.isfinite(distances)
        scores[finite, cluster_id] = 1.0 / (1.0 + distances[finite])

    return _normalize_rows(_set_seed_identity(scores, seed_rows, seed_labels))


def _cluster_watershed_assignments(
    matrix: csr_matrix,
    *,
    seed_rows: npt.NDArray[np.int64],
    seed_labels: npt.NDArray[np.int64],
    nclusters: int,
) -> npt.NDArray[np.float64]:
    scores = np.zeros((matrix.shape[0], nclusters), dtype=np.float64)
    if matrix.nnz == 0:
        return _normalize_rows(_set_seed_identity(scores, seed_rows, seed_labels))

    closeness = _cluster_shortest_path_assignments(
        matrix,
        seed_rows=seed_rows,
        seed_labels=seed_labels,
        nclusters=nclusters,
    )
    for cluster_id in range(nclusters):
        scores[:, cluster_id] = _multi_seed_widest_path_scores(
            matrix,
            seed_rows[seed_labels == cluster_id],
        )

    # Tie-break equal-capacity paths by ordinary closeness to the cluster.
    scores = scores + 1.0e-6 * closeness
    return _normalize_rows(_set_seed_identity(scores, seed_rows, seed_labels))


def _cluster_region_grow_assignments(
    matrix: csr_matrix,
    *,
    seed_rows: npt.NDArray[np.int64],
    seed_labels: npt.NDArray[np.int64],
    nclusters: int,
) -> npt.NDArray[np.float64]:
    scores = np.zeros((matrix.shape[0], nclusters), dtype=np.float64)
    if matrix.nnz == 0:
        return _set_seed_identity(scores, seed_rows, seed_labels)

    adjacency = matrix.copy().tocsr()
    adjacency.data = np.ones_like(adjacency.data, dtype=np.float64)
    hop_distances = np.zeros((matrix.shape[0], nclusters), dtype=np.float64)
    for cluster_id in range(nclusters):
        hop_distances[:, cluster_id] = _multi_seed_distances(
            adjacency,
            seed_rows[seed_labels == cluster_id],
        )

    closeness = _cluster_shortest_path_assignments(
        matrix,
        seed_rows=seed_rows,
        seed_labels=seed_labels,
        nclusters=nclusters,
    )

    for node in range(matrix.shape[0]):
        finite = np.isfinite(hop_distances[node])
        if not np.any(finite):
            continue
        best_hop = float(np.min(hop_distances[node, finite]))
        contenders = np.where(np.isclose(hop_distances[node], best_hop))[0]
        if contenders.size == 1:
            scores[node, int(contenders[0])] = 1.0
            continue
        tie_weights = closeness[node, contenders]
        if np.allclose(tie_weights.sum(), 0.0):
            scores[node, contenders] = 1.0 / contenders.size
        else:
            scores[node, contenders] = tie_weights / tie_weights.sum()

    return _set_seed_identity(scores, seed_rows, seed_labels)


def _set_seed_identity(
    scores: npt.NDArray[np.float64],
    seed_rows: npt.NDArray[np.int64],
    seed_labels: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    scores[seed_rows, :] = 0.0
    scores[seed_rows, seed_labels] = 1.0
    return scores


def _multi_seed_distances(
    matrix: csr_matrix,
    seed_rows: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    if seed_rows.size == 0:
        return np.full(matrix.shape[0], np.inf, dtype=np.float64)
    distances = dijkstra(
        matrix,
        directed=False,
        indices=seed_rows,
        min_only=True,
    )
    return np.asarray(distances, dtype=np.float64).reshape(-1)


def _multi_seed_widest_path_scores(
    matrix: csr_matrix,
    seed_rows: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    npoints = matrix.shape[0]
    scores = np.zeros(npoints, dtype=np.float64)
    heap: list[tuple[float, int]] = []
    for seed_row in seed_rows:
        scores[int(seed_row)] = np.inf
        heappush(heap, (-scores[int(seed_row)], int(seed_row)))

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
    normalized[np.isinf(normalized)] = max(max_finite, 1.0)
    return normalized
