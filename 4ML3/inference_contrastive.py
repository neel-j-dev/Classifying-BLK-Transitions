"""Compute embeddings with a trained contrastive lattice encoder and optionally save them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from data_utils import load_split
from models import build_contrastive_lattice_model


def load_metadata(dataset_dir: Path) -> dict | None:
    metadata_path = dataset_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open() as f:
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


def make_graphs(features: np.ndarray, temps: np.ndarray, lattice_shape: Tuple[int, int], edge_index: torch.Tensor) -> List[Data]:
    num_nodes = lattice_shape[0] * lattice_shape[1]
    if features.shape[1] != num_nodes:
        raise ValueError(f"Feature length {features.shape[1]} does not match lattice nodes {num_nodes}.")
    edge_attr = torch.ones(edge_index.size(1), 1, dtype=torch.float32)
    graphs: List[Data] = []
    for x_arr, temp in zip(features, temps):
        x = torch.from_numpy(x_arr.reshape(-1, 1)).float()
        graphs.append(
            Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                temp=torch.tensor(temp, dtype=torch.float32),
            )
        )
    return graphs


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained contrastive lattice encoder.")
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/contrastive_model.joblib"))
    parser.add_argument("--dataset", type=Path, default=Path("../XYModel/blt_dataset/test.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/contrastive_embeddings.npz"))
    parser.add_argument("--pca-plot", type=Path, default=None, help="Optional path to save PCA scatter of embeddings colored by temperature.")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, temps, labels = load_split(args.dataset)
    metadata = load_metadata(args.dataset.parent)

    artifact = joblib.load(args.artifact)
    model_cfg = artifact.get(
        "model_config",
        {"input_dim": 1, "hidden_dim": 128, "num_layers": 3, "projection_dim": 64, "dropout": 0.1},
    )
    lattice_shape = tuple(artifact.get("lattice_shape", infer_lattice_shape(features.shape[1], metadata)))
    periodic = bool(artifact.get("periodic", True))
    edge_index = artifact.get("edge_index")
    if edge_index is None:
        edge_index = build_lattice_edge_index(lattice_shape, periodic=periodic)
    else:
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)

    graphs = make_graphs(features, temps, lattice_shape, edge_index)
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)

    model = build_contrastive_lattice_model(
        input_dim=model_cfg.get("input_dim", 1),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_layers=model_cfg.get("num_layers", 3),
        projection_dim=model_cfg.get("projection_dim", 64),
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)
    state_dict = artifact.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Artifact is missing model_state_dict needed for inference.")
    model.load_state_dict(state_dict)
    model.eval()

    embeddings = []
    temps_out = []
    labels_out = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            z = model.embed(data)
            embeddings.append(z.cpu())
            temps_out.append(data.temp.view(-1).cpu())
            if hasattr(data, "y") and data.y is not None:
                labels_out.append(data.y.view(-1).cpu())

    embeddings_np = torch.cat(embeddings).numpy()
    temps_np = torch.cat(temps_out).numpy()
    labels_np = torch.cat(labels_out).numpy() if labels_out else np.zeros(len(embeddings_np), dtype=int)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, embeddings=embeddings_np, temperatures=temps_np, labels=labels_np)
    print(f"Saved contrastive embeddings to {args.output}")

    if args.pca_plot is not None:
        pca = PCA(n_components=2)
        proj = pca.fit_transform(embeddings_np)
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            proj[:, 0],
            proj[:, 1],
            c=temps_np,
            cmap="viridis",
            s=30,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.2,
        )
        cb = plt.colorbar(scatter, ax=ax)
        cb.set_label("Temperature")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        ax.set_title("Contrastive embeddings PCA (colored by temperature)")
        fig.tight_layout()
        args.pca_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.pca_plot, dpi=200)
        plt.close(fig)
        print(f"Saved PCA plot to {args.pca_plot}")


if __name__ == "__main__":
    main()
