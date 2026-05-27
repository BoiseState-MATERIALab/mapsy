from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.spatial.distance as distance
from scipy.linalg import qr as scipy_qr


def normalize_point_selection_method(method: str) -> str:
    normalized = str(method).lower()
    aliases = {
        "adaptive": "greedy",
        "maximin": "greedy",
        "pivot_qr": "pivoted_qr",
        "qr": "pivoted_qr",
        "pivot_cholesky": "pivoted_cholesky",
        "cholesky": "pivoted_cholesky",
        "rbf": "pivoted_cholesky",
        "rbf_cholesky": "pivoted_cholesky",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"greedy", "pivoted_qr", "pivoted_cholesky"}:
        raise ValueError(
            "method must be one of {'greedy', 'pivoted_qr', 'pivoted_cholesky'}, "
            f"got {method!r}."
        )
    return normalized


def normalize_site_selection_method(method: str) -> str:
    normalized = str(method).lower()
    aliases = {
        "cluster": "cluster_centroid",
        "clusters": "cluster_centroid",
        "centroid": "cluster_centroid",
        "centroids": "cluster_centroid",
        "cluster_centroids": "cluster_centroid",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized == "cluster_centroid":
        return normalized
    return normalize_point_selection_method(normalized)


def pivoted_qr_selection(
    candidate_features: npt.ArrayLike,
    npoints: int,
    *,
    seed_features: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    features = _as_feature_matrix(candidate_features)
    if features.shape[1] == 0:
        raise ValueError("pivoted_qr selection requires at least one feature column.")
    if npoints > features.shape[0]:
        raise ValueError(
            f"Requested {npoints} points, but only {features.shape[0]} candidates are available."
        )

    columns = features.T.copy()
    seeds = _as_feature_matrix(seed_features, width=features.shape[1])
    if seeds.size:
        seed_columns = seeds.T
        basis, _ = np.linalg.qr(seed_columns, mode="reduced")
        if basis.size:
            columns = columns - basis @ (basis.T @ columns)

    _, r_matrix, pivots = scipy_qr(columns, mode="economic", pivoting=True)
    selected = np.asarray(pivots[:npoints], dtype=np.int64)
    scores = np.zeros(npoints, dtype=np.float64)
    diagonal = np.abs(np.diag(r_matrix))
    ndiag = min(npoints, diagonal.size)
    if ndiag:
        scores[:ndiag] = diagonal[:ndiag]
    if ndiag < npoints:
        scores[ndiag:] = np.linalg.norm(columns[:, selected[ndiag:]], axis=0)
    return selected, scores


def pivoted_cholesky_rbf_selection(
    candidate_features: npt.ArrayLike,
    npoints: int,
    *,
    seed_features: npt.ArrayLike | None = None,
    gamma: float | str | None = "median",
    tolerance: float = 1.0e-12,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], float]:
    features = _as_feature_matrix(candidate_features)
    if features.shape[1] == 0:
        raise ValueError("pivoted_cholesky selection requires at least one feature column.")
    if npoints > features.shape[0]:
        raise ValueError(
            f"Requested {npoints} points, but only {features.shape[0]} candidates are available."
        )

    gamma_value = _resolve_rbf_gamma(features, gamma)
    ncandidates = features.shape[0]
    residual = np.ones(ncandidates, dtype=np.float64)
    seeds = _as_feature_matrix(seed_features, width=features.shape[1])
    if seeds.size:
        seed_kernel = np.exp(-gamma_value * distance.cdist(features, seeds, "sqeuclidean"))
        residual -= np.max(seed_kernel * seed_kernel, axis=1)
        residual = np.clip(residual, 0.0, None)

    selected: list[int] = []
    scores: list[float] = []
    factors: list[npt.NDArray[np.float64]] = []
    available = np.ones(ncandidates, dtype=bool)

    for _ in range(npoints):
        pivot_scores = np.where(available, residual, -np.inf)
        pivot = int(np.argmax(pivot_scores))
        pivot_score = float(pivot_scores[pivot])
        selected.append(pivot)
        scores.append(max(pivot_score, 0.0))
        available[pivot] = False

        if pivot_score <= tolerance:
            residual[pivot] = 0.0
            continue

        column = np.exp(
            -gamma_value * distance.cdist(features, features[pivot : pivot + 1], "sqeuclidean")
        ).reshape(-1)
        if factors:
            factor_matrix = np.column_stack(factors)
            column = column - factor_matrix @ factor_matrix[pivot, :]
        column = column / np.sqrt(pivot_score)
        factors.append(column)
        residual = np.clip(residual - column * column, 0.0, None)
        residual[~available] = 0.0

    return (
        np.asarray(selected, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
        gamma_value,
    )


def _as_feature_matrix(
    values: npt.ArrayLike | None,
    *,
    width: int | None = None,
) -> npt.NDArray[np.float64]:
    if values is None:
        ncols = 0 if width is None else width
        return np.zeros((0, ncols), dtype=np.float64)
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError("feature matrix must be two-dimensional.")
    if width is not None and matrix.shape[1] != width:
        raise ValueError(
            f"seed feature width {matrix.shape[1]} does not match candidate width {width}."
        )
    return matrix


def _resolve_rbf_gamma(features: npt.NDArray[np.float64], gamma: float | str | None) -> float:
    nfeatures = int(features.shape[1])
    if gamma is None:
        return 1.0 / max(nfeatures, 1)
    if isinstance(gamma, str):
        normalized = gamma.lower()
        if normalized in {"scale", "auto"}:
            return 1.0 / max(nfeatures, 1)
        if normalized != "median":
            raise ValueError("gamma must be a float, None, or one of {'median', 'scale', 'auto'}.")
        sample = features
        if len(sample) > 1000:
            indexes = np.linspace(0, len(sample) - 1, 1000).astype(np.int64)
            sample = sample[indexes]
        distances = distance.pdist(sample, metric="sqeuclidean")
        positive = distances[distances > 0.0]
        if positive.size == 0:
            return 1.0
        return 1.0 / float(np.median(positive))
    gamma_value = float(gamma)
    if gamma_value <= 0.0:
        raise ValueError("gamma must be positive.")
    return gamma_value


def selection_metadata_columns(method: str) -> list[str]:
    normalized = normalize_point_selection_method(method)
    columns = [
        "selection_rank",
        "selection_score",
        "selection_method",
        "real_space_score",
        "feature_space_score",
        "energy_score",
        "uncertainty_score",
        "pivot_score",
    ]
    if normalized == "pivoted_cholesky":
        columns.append("kernel_gamma")
    return columns
