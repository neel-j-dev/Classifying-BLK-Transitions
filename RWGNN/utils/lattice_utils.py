from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


# -------------------------------
# Dataset Metadata and Lattice Utilities
# -------------------------------
def load_metadata(dataset_dir: Path | str) -> Optional[Dict]:
    """Load dataset metadata.json if present."""

    dataset_dir = Path(dataset_dir)
    metadata_path = dataset_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open() as f:
            return json.load(f)
    return None


# -------------------------------
# Lattice Graph Utilities
# -------------------------------

def infer_lattice_shape(num_features: int, metadata: Optional[dict]) -> Tuple[int, int]:
    """Infer lattice shape from metadata or assume square from feature length."""

    if metadata and "feature_shape" in metadata:
        shape = metadata["feature_shape"]
        return int(shape[0]), int(shape[1])
    side = int(round(np.sqrt(num_features)))
    if side * side != num_features:
        raise ValueError(f"Cannot infer square lattice from feature length={num_features}.")
    return side, side


def build_lattice_edge_index(lattice_shape: Tuple[int, int], periodic: bool = True) -> torch.Tensor:
    """Create undirected edges for a 2D lattice."""

    rows, cols = lattice_shape
    edges = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            neighbors = [
                ((r + 1) % rows, c),
                ((r - 1) % rows, c),
                (r, (c + 1) % cols),
                (r, (c - 1) % cols),
            ] if periodic else [
                (r + 1, c) if r + 1 < rows else None,
                (r - 1, c) if r - 1 >= 0 else None,
                (r, c + 1) if c + 1 < cols else None,
                (r, c - 1) if c - 1 >= 0 else None,
            ]
            for nbr in neighbors:
                if nbr is None:
                    continue
                nbr_idx = nbr[0] * cols + nbr[1]
                edges.append((idx, nbr_idx))
                edges.append((nbr_idx, idx))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_edge_attr(edge_index: torch.Tensor, value: float = 1.0) -> torch.Tensor:
    """Return edge attributes matching edge_index with a constant value."""

    return torch.full((edge_index.size(1), 1), fill_value=value, dtype=torch.float32)


def make_lattice_graphs(
    features: np.ndarray,
    temps: np.ndarray,
    lattice_shape: Tuple[int, int],
    edge_index: torch.Tensor,
    labels: Optional[np.ndarray] = None,
) -> List[Data]:
    """Convert lattice samples into PyG graphs with per-node features + optional labels."""

    num_nodes = lattice_shape[0] * lattice_shape[1]
    if features.shape[1] % num_nodes != 0:
        raise ValueError(f"Feature length {features.shape[1]} is not divisible by lattice nodes {num_nodes}.")
    per_node_dim = features.shape[1] // num_nodes

    edge_attr = build_edge_attr(edge_index)
    graphs: List[Data] = []
    for idx, (x_arr, temp) in enumerate(zip(features, temps)):
        x = torch.from_numpy(x_arr.reshape(num_nodes, per_node_dim)).float()
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            temp=torch.tensor(temp, dtype=torch.float32),
        )
        if labels is not None:
            data.y = torch.tensor(labels[idx], dtype=torch.long)
        graphs.append(data)
    return graphs
