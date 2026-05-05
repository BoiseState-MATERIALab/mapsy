from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_matrix, diags
from sklearn.preprocessing import StandardScaler

from mapsy.results import ArchetypePropagationResult, ArchetypeSelectionResult, GraphResult


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
    probability_weight: float = 1.0,
    extremeness_weight: float = 1.0,
    diversity_weight: float = 1.0,
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
    extremeness_raw = np.linalg.norm(candidate_features - centroid, axis=1)
    extremeness_score = _normalize_component(extremeness_raw)

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
            "probability_weight": probability_weight,
            "extremeness_weight": extremeness_weight,
            "diversity_weight": diversity_weight,
        },
    )


def propagate_archetypes(
    graph: GraphResult,
    *,
    selected_indexes: npt.ArrayLike,
    point_index_column: str = "point_index",
    alpha: float = 0.9,
    max_iter: int = 500,
    tol: float = 1.0e-8,
    confidence_threshold: float = 0.5,
    margin_threshold: float = 0.0,
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

    normalized_matrix = _row_normalize(graph.matrix)
    target = np.zeros((npoints, narchetypes), dtype=np.float64)
    target[seed_rows, np.arange(narchetypes, dtype=np.int64)] = 1.0
    field = target.copy()

    for _ in range(max_iter):
        updated = alpha * normalized_matrix.dot(field) + (1.0 - alpha) * target
        updated[seed_rows, :] = target[seed_rows, :]
        delta = float(np.max(np.abs(updated - field)))
        field = updated
        if delta < tol:
            break

    row_sums = field.sum(axis=1)
    positive_mask = row_sums > 0.0
    normalized_field = np.zeros_like(field)
    normalized_field[positive_mask] = field[positive_mask] / row_sums[positive_mask, None]

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
            "alpha": alpha,
            "max_iter": max_iter,
            "tol": tol,
            "confidence_threshold": confidence_threshold,
            "margin_threshold": margin_threshold,
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


def _row_normalize(matrix: csr_matrix) -> csr_matrix:
    row_sums = np.asarray(matrix.sum(axis=1), dtype=np.float64).reshape(-1)
    inv = np.zeros_like(row_sums, dtype=np.float64)
    positive = row_sums > 0.0
    inv[positive] = 1.0 / row_sums[positive]
    return diags(inv).dot(matrix).tocsr()
