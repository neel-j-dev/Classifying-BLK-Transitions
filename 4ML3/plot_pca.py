"""
Compute and visualize PCA of graph embeddings from the trained PyG lattice models.

This script loads a trained artifact, rebuilds the corresponding model (GCN classifier
or attention-based temperature regressor), extracts graph-level embeddings from the
last convolution block (before the head), and plots a 2-D PCA scatter colored by
phase labels (or temperatures).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from data_utils import load_split
from models import build_pyg_lattice_model


def load_metadata(dataset_path: Path) -> dict | None:
    meta_path = dataset_path.parent / "metadata.json"
    if meta_path.exists():
        with meta_path.open() as f:
            return json.load(f)
    return None


def infer_lattice_shape(num_features: int, metadata: dict | None) -> Tuple[int, int]:
    if metadata and "feature_shape" in metadata:
        shape = metadata["feature_shape"]
        return int(shape[0]), int(shape[1])
    side = int(round(np.sqrt(num_features)))
    if side * side != num_features:
        raise ValueError(f"Cannot infer square lattice from feature length={num_features}.")
    return side, side


def build_lattice_edge_index(lattice_shape: Tuple[int, int], periodic: bool = True) -> torch.Tensor:
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


def make_graphs(features: np.ndarray, temps: np.ndarray, labels: np.ndarray, edge_index: torch.Tensor, lattice_shape: Tuple[int, int]) -> List[Data]:
    num_nodes = lattice_shape[0] * lattice_shape[1]
    if features.shape[1] != num_nodes:
        raise ValueError(f"Feature length {features.shape[1]} does not match lattice nodes {num_nodes}.")

    edge_attr = torch.ones(edge_index.size(1), 1, dtype=torch.float32)
    graphs: List[Data] = []
    for x_arr, temp, label in zip(features, temps, labels):
        x = torch.from_numpy(x_arr.reshape(-1, 1)).float()
        graphs.append(
            Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(label, dtype=torch.long),
                temp=torch.tensor(temp, dtype=torch.float32),
            )
        )
    return graphs


def compute_embeddings(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    embeddings = []
    labels = []
    temps = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if not hasattr(model, "embed"):
                raise AttributeError("Model missing embed() needed for PCA extraction.")
            z = model.embed(data)
            embeddings.append(z.cpu())
            labels.append(data.y.view(-1).cpu())
            temps.append(data.temp.view(-1).cpu())
    embed_np = torch.cat(embeddings).numpy()
    labels_np = torch.cat(labels).numpy()
    temps_np = torch.cat(temps).numpy()
    return embed_np, labels_np, temps_np


def plot_pca(embeddings: np.ndarray, color_values: np.ndarray, color_label: str, output: Path, title: str):
    pca = PCA(n_components=2)
    proj = pca.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        proj[:, 0],
        proj[:, 1],
        c=color_values,
        cmap="plasma",
        s=28,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.2,
    )
    cb = plt.colorbar(scatter, ax=ax)
    cb.set_label(color_label)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved PCA plot to {output}")


def main():
    parser = argparse.ArgumentParser(description="PCA visualization of PyG lattice graph embeddings.")
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/pyg_lattice_model.joblib"))
    parser.add_argument("--dataset", type=Path, default=Path("../XYModel/blt_dataset/train.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/pca_embeddings.png"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = joblib.load(args.artifact)
    model_cfg = artifact.get("model_config", {"input_dim": 1, "hidden_dim": 64, "num_layers": 2, "dropout": 0.1, "heads": 2})
    model_type = artifact.get("model_type", "gcn")

    features, temps, labels = load_split(args.dataset)
    metadata = load_metadata(args.dataset)
    lattice_shape = tuple(artifact.get("lattice_shape", infer_lattice_shape(features.shape[1], metadata)))
    periodic = bool(artifact.get("periodic", False))
    edge_index = artifact.get("edge_index")
    if edge_index is None:
        edge_index = build_lattice_edge_index(lattice_shape, periodic=periodic)
    else:
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)

    graphs = make_graphs(features, temps, labels, edge_index, lattice_shape)
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)

    model = build_pyg_lattice_model(
        model_type=model_type,
        input_dim=model_cfg.get("input_dim", 1),
        hidden_dim=model_cfg.get("hidden_dim", 64),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.1),
        heads=model_cfg.get("heads", 2),
    ).to(device)

    state_dict = artifact.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Artifact is missing model_state_dict needed for embeddings.")
    model.load_state_dict(state_dict)

    embeddings, labels_out, temps_out = compute_embeddings(model, loader, device)

    if model_type == "attn_temp":
        color_values = temps_out
        color_label = "Temperature"
    else:
        color_values = labels_out
        color_label = "Phase label"

    plot_title = f"PCA of embeddings ({model_type})"
    plot_pca(embeddings, color_values, color_label, args.output, plot_title)


if __name__ == "__main__":
    main()
