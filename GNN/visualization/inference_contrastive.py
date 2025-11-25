"""Compute embeddings with a trained contrastive lattice encoder and optionally save them."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import torch
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from GNN.utils.data_utils import load_split
from GNN.utils.lattice_utils import build_lattice_edge_index, infer_lattice_shape, load_metadata, make_lattice_graphs
from GNN.models.models import build_contrastive_lattice_model


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained contra   stive lattice encoder.")
    parser.add_argument("--artifact", type=Path, default=Path("GNN/artifacts/contrastive_model.joblib"))
    parser.add_argument("--dataset", type=Path, default=Path("XYModel/blt_dataset/test.npz"))
    parser.add_argument("--output", type=Path, default=Path("GNN/artifacts/contrastive_embeddings.npz"))
    parser.add_argument("--pca-plot", type=Path, default=None, help="Optional path to save PCA scatter of embeddings colored by temperature.")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, temps, labels = load_split(args.dataset)
    metadata = load_metadata(args.dataset.parent)

    artifact = joblib.load(args.artifact)
    model_cfg = artifact.get(
        "model_config",
        {"input_dim": 2, "hidden_dim": 128, "num_layers": 3, "projection_dim": 64, "dropout": 0.1},
    )
    lattice_shape = tuple(artifact.get("lattice_shape", infer_lattice_shape(features.shape[1], metadata)))
    periodic = bool(artifact.get("periodic", True))
    edge_index = artifact.get("edge_index")
    if edge_index is None:
        edge_index = build_lattice_edge_index(lattice_shape, periodic=periodic)
    else:
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)

    graphs = make_lattice_graphs(features, temps, lattice_shape, edge_index, labels=None)
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)

    model = build_contrastive_lattice_model(
        input_dim=model_cfg.get("input_dim", 2),
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
        for data in tqdm(loader, desc="Embedding", leave=False):
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
