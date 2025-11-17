from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(
    features: np.ndarray,
    n_neighbors: int,
    metric: Literal["euclidean", "cosine"] = "euclidean",
) -> sparse.csr_matrix:
    """Return a symmetric adjacency matrix using a k-NN graph."""

    n_samples = features.shape[0]
    if n_samples < 2:
        raise ValueError("At least two samples are required to build a graph.")

    k = min(max(1, n_neighbors), n_samples - 1)
    # Include self in neighbor list so we can drop it below.
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nbrs.fit(features)
    distances, indices = nbrs.kneighbors(features)

    rows = []
    cols = []
    weights = []

    for node in range(n_samples):
        # Skip the first neighbor (self) and convert distances to similarity weights.
        for neighbor_idx, dist in zip(indices[node, 1:], distances[node, 1:]):
            rows.append(node)
            cols.append(neighbor_idx)
            weight = float(np.exp(-dist)) if metric == "euclidean" else float(1 - dist)
            weights.append(max(0.0, weight))

    adjacency = sparse.coo_matrix((weights, (rows, cols)), shape=(n_samples, n_samples))
    adjacency = adjacency.maximum(adjacency.transpose()).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency
