"""UMAP visualization of PyG lattice graph embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader

import umap

from GNN.utils.data_utils import load_split
from GNN.utils.lattice_utils import build_lattice_edge_index, infer_lattice_shape, load_metadata, make_lattice_graphs
from GNN.models.models import build_pyg_lattice_model


def main():
    parser = argparse.ArgumentParser(description="UMAP visualization of PyG lattice graph embeddings.")
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/pyg_lattice_model.joblib"))
    parser.add_argument("--dataset", type=Path, default=Path("../XYModel/blt_dataset/train.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/umap_embeddings.png"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", type=str, default="euclidean")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = joblib.load(args.artifact)
    model_cfg = artifact.get("model_config", {"input_dim": 2, "hidden_dim": 64, "num_layers": 2, "dropout": 0.1, "heads": 2})
    model_type = artifact.get("model_type", "gcn")

    features, temps, labels = load_split(args.dataset)
    metadata = load_metadata(args.dataset.parent)
    lattice_shape = tuple(artifact.get("lattice_shape", infer_lattice_shape(features.shape[1], metadata)))
    periodic = bool(artifact.get("periodic", False))
    edge_index = artifact.get("edge_index")
    if edge_index is None:
        edge_index = build_lattice_edge_index(lattice_shape, periodic=periodic)
    else:
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)

    graphs = make_lattice_graphs(features, temps, lattice_shape, edge_index, labels=labels)
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)

    if model_type == "attn_temp":
        # Use the attention regressor to embed
        model_type = "attn_temp"
    model = build_pyg_lattice_model(
        model_type=model_type,
        input_dim=model_cfg.get("input_dim", 2),
        hidden_dim=model_cfg.get("hidden_dim", 64),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.1),
        heads=model_cfg.get("heads", 2),
    ).to(device)

    state_dict = artifact.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Artifact is missing model_state_dict needed for embeddings.")
    model.load_state_dict(state_dict)
    model.eval()

    embeddings = []
    temps_out = []
    labels_out = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if hasattr(model, "embed"):
                z = model.embed(data)
            else:
                logits = model(data)
                z = logits
            embeddings.append(z.cpu())
            temps_out.append(data.temp.view(-1).cpu())
            if hasattr(data, "y"):
                labels_out.append(data.y.view(-1).cpu())

    embeddings_np = torch.cat(embeddings).numpy()
    temps_np = torch.cat(temps_out).numpy()
    labels_np = torch.cat(labels_out).numpy() if labels_out else np.zeros(len(embeddings_np), dtype=int)

    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=0,
    )
    proj = reducer.fit_transform(embeddings_np)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        proj[:, 0],
        proj[:, 1],
        c=temps_np if model_type == "attn_temp" else labels_np,
        cmap="viridis" if model_type == "attn_temp" else "plasma",
        s=30,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.2,
    )
    cb = plt.colorbar(scatter, ax=ax)
    cb.set_label("Temperature" if model_type == "attn_temp" else "Phase label")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"UMAP embeddings ({model_type})")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Saved UMAP plot to {args.output}")


if __name__ == "__main__":
    main()
