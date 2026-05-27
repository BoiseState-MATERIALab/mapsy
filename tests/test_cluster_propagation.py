import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from mapsy.analysis.clustering import propagate_cluster_labels
from mapsy.multimaps import _nearest_reference_indexes
from mapsy.results import GraphResult


def _chain_graph(npoints: int) -> GraphResult:
    sources = np.arange(npoints - 1, dtype=np.int64)
    targets = sources + 1
    weights = np.ones(npoints - 1, dtype=np.float64)
    edge_table = pd.DataFrame(
        {
            "source": sources,
            "target": targets,
            "weight": weights,
        }
    )
    rows = np.concatenate([sources, targets])
    cols = np.concatenate([targets, sources])
    vals = np.concatenate([weights, weights])
    matrix = csr_matrix((vals, (rows, cols)), shape=(npoints, npoints), dtype=np.float64)
    return GraphResult(
        mode="realspace",
        feature_columns=[],
        node_weight_column="probability",
        node_table=pd.DataFrame({"point_index": np.arange(npoints, dtype=np.int64)}),
        node_weights=np.ones(npoints, dtype=np.float64),
        edge_table=edge_table,
        matrix=matrix,
    )


@pytest.mark.parametrize("propagation_mode", ["shortest_path", "watershed", "region_grow"])
def test_cluster_label_propagation_uses_many_seeds_by_cluster(
    propagation_mode: str,
) -> None:
    graph = _chain_graph(8)
    seed_rows = np.array([0, 1, 2, 5, 6, 7], dtype=np.int64)
    seed_labels = np.full(8, -1, dtype=np.int64)
    seed_labels[seed_rows] = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    seed_mask = seed_labels >= 0

    labels, confidence, margin, ambiguous, scores = propagate_cluster_labels(
        graph,
        seed_mask=seed_mask,
        seed_labels=seed_labels,
        propagation_mode=propagation_mode,
    )

    assert scores.shape == (8, 2)
    assert labels.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert np.all(confidence > 0.0)
    assert np.all(margin >= 0.0)
    assert not np.any(ambiguous)


def test_nearest_reference_indexes_returns_one_reference_per_point() -> None:
    points = np.array([[0.1], [9.8], [4.9]], dtype=np.float64)
    references = np.array([[0.0], [5.0], [10.0]], dtype=np.float64)

    nearest = _nearest_reference_indexes(points, references)

    assert nearest.tolist() == [0, 2, 1]
