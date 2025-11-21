"""Train a contrastive lattice graph encoder (SimCLR-style) on lattice configurations."""

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


def augment_graph(data: Data, noise_std: float) -> Data:
    """Simple augmentation: add Gaussian noise to node features."""

    noisy_x = data.x + noise_std * torch.randn_like(data.x)
    return Data(
        x=noisy_x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        temp=data.temp,
        batch=data.batch if hasattr(data, "batch") else None,
    )


def train_epoch(model, loader, optimizer, device, noise_std: float) -> float:
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for data in loader:
        data = data.to(device)
        data1 = augment_graph(data, noise_std)
        data2 = augment_graph(data, noise_std)
        data1 = data1.to(device)
        data2 = data2.to(device)

        optimizer.zero_grad()
        z1 = model(data1)
        z2 = model(data2)
        loss = model.info_nce_loss(z1, z2)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * data.num_graphs
        total_graphs += data.num_graphs
    return total_loss / max(1, total_graphs)


def main():
    parser = argparse.ArgumentParser(description="Contrastive training for lattice graph encoder.")
    parser.add_argument("--dataset", type=Path, default=Path("../XYModel/blt_dataset/train.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/contrastive_model.joblib"))
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.05, help="Stddev for feature noise augmentation.")
    parser.add_argument("--no-periodic", action="store_true", default=False, help="Disable periodic boundary edges.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and lattice info
    features, temps, _ = load_split(args.dataset)
    metadata = load_metadata(args.dataset.parent)
    lattice_shape = infer_lattice_shape(features.shape[1], metadata)
    edge_index = build_lattice_edge_index(lattice_shape, periodic=not args.no_periodic)
    graphs = make_graphs(features, temps, lattice_shape, edge_index)

    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=True)

    model = build_contrastive_lattice_model(
        input_dim=1,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, device, noise_std=args.noise_std)
        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs:
            print(f"Epoch {epoch}/{args.epochs} - contrastive_loss={loss:.4f}")

    artifact = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": 1,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "projection_dim": args.projection_dim,
            "dropout": args.dropout,
        },
        "lattice_shape": lattice_shape,
        "periodic": not args.no_periodic,
        "edge_index": edge_index,
        "noise_std": args.noise_std,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    print(f"Saved contrastive model to {args.output}")


if __name__ == "__main__":
    main()
