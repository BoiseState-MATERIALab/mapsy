import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from mapsy.clustering import (
    clustering_uses_random_state,
    fit_predict_clusters,
    normalize_cluster_method,
)
from mapsy.results import ClusterResult, ClusterScreeningResult


def fit_clusters(
    X: npt.NDArray[np.float64],
    *,
    feature_columns: list[str],
    nclusters: int,
    method: str = "spectral",
    random_state: int | None = None,
    scale: bool = False,
    screening: ClusterScreeningResult | None = None,
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
    )
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
        metadata={"scaler": scaler},
    )


def screen_clusters(
    X: npt.NDArray[np.float64],
    *,
    feature_columns: list[str],
    method: str = "spectral",
    maxclusters: int = 20,
    ntries: int = 1,
    scale: bool = False,
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


def _cluster_sizes(labels: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    _, counts = np.unique(labels, return_counts=True)
    return counts.astype(np.int64, copy=False)
