from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, TransformerConv, global_mean_pool

from data_utils import load_split
from metrics import estimate_critical_temperature, phase_metrics


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

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def make_graphs(
    features: np.ndarray,
    temps: np.ndarray,
    labels: np.ndarray,
    lattice_shape: Tuple[int, int],
    edge_index: torch.Tensor,
) -> List[Data]:
    """Convert each lattice sample into a PyG graph with per-node angles."""

    num_nodes = lattice_shape[0] * lattice_shape[1]
    if features.shape[1] != num_nodes:
        raise ValueError(
            f"Feature length {features.shape[1]} does not match lattice nodes {num_nodes}."
        )

    data_list: List[Data] = []
    edge_attr = torch.ones(edge_index.size(1), 1, dtype=torch.float32)
    for x_arr, temp, label in zip(features, temps, labels):
        x = torch.from_numpy(x_arr.reshape(-1, 1)).float()
        y = torch.tensor(label, dtype=torch.long)
        temp_tensor = torch.tensor(temp, dtype=torch.float32)
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            temp=temp_tensor,
        )
        data_list.append(data)
    return data_list


class GridGraphClassifier(nn.Module):
    """GCN-based graph classifier with global mean pooling."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        self.convs = nn.ModuleList()
        dims: Sequence[int] = [input_dim] + [hidden_dim] * num_layers
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.convs.append(GCNConv(in_dim, out_dim, add_self_loops=False, normalize=True))
        self.dropout = dropout
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.head(x)


class AttentionLatticeClassifier(nn.Module):
    """Attention-based temperature regressor that uses node + edge features (no graph-level temp input)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        edge_attr_dim: int = 1,
        heads: int = 2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        dims: Sequence[int] = [input_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList()
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.convs.append(
                TransformerConv(
                    in_dim,
                    out_dim,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=edge_attr_dim,
                )
            )
        self.dropout = dropout
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, data: Data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.size(1), 1, device=x.device, dtype=x.dtype)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.head(x)


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data)
        if logits.shape[-1] == 1:
            loss = F.mse_loss(logits.view(-1), data.temp.view(-1))
        else:
            loss = F.cross_entropy(logits, data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * data.num_graphs
        total_graphs += data.num_graphs
    return total_loss / max(1, total_graphs)


def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    all_temps = []
    all_pred_temps = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data)
            if logits.shape[-1] == 1:
                preds = logits.view(-1)
                all_pred_temps.append(preds.cpu())
                all_temps.append(data.temp.view(-1).cpu())
            else:
                probs = torch.softmax(logits, dim=-1)[:, 1]
                all_probs.append(probs.cpu())
                all_labels.append(data.y.view(-1).cpu())
                all_temps.append(data.temp.view(-1).cpu())

    if all_pred_temps:
        preds_np = torch.cat(all_pred_temps).numpy()
        temps_np = torch.cat(all_temps).numpy()
        mae = float(np.mean(np.abs(preds_np - temps_np)))
        rmse = float(np.sqrt(np.mean((preds_np - temps_np) ** 2)))
        metrics = {"mae": mae, "rmse": rmse}
        return preds_np, temps_np, temps_np, metrics

    if not all_probs:
        return np.array([]), np.array([]), np.array([]), {}

    probs_np = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy()
    temps_np = torch.cat(all_temps).numpy()

    metrics = phase_metrics(labels_np, probs_np)
    metrics["t_c"] = estimate_critical_temperature(temps_np, probs_np)
    return probs_np, labels_np, temps_np, metrics


def add_split_arg(parser: argparse.ArgumentParser, name: str, default_filename: str):
    parser.add_argument(
        f"--{name}",
        type=Path,
        default=None,
        help=f"Optional path to a {name} split; defaults to dataset_dir/{default_filename}.",
    )


def resolve_split_path(dataset_dir: Path, cli_path: Path | None, filename: str) -> Path:
    return cli_path if cli_path is not None else dataset_dir / filename


def print_metrics(metrics: dict):
    if "accuracy" in metrics:
        print(
            f"[{metrics['split']}] accuracy={metrics['accuracy']:.3f} "
            f"f1={metrics['f1']:.3f} t_c={metrics['t_c']:.3f} "
            f"roc_auc={metrics['roc_auc']:.3f}"
        )
    else:
        print(
            f"[{metrics['split']}] mae={metrics['mae']:.4f} "
            f"rmse={metrics['rmse']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a PyTorch Geometric model on lattice graphs (one graph per sample)."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("../XYModel/blt_dataset"))
    parser.add_argument("--model-type", choices=["gcn", "attn_temp"], default="attn_temp",
                        help="gcn=phase classifier, attn_temp=attention regressor for temperature.")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--n-epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/pyg_lattice_model.joblib"))
    parser.add_argument("--no-periodic", action="store_true", help="Disable periodic boundary edges.", default=True)
    parser.add_argument("--heads", type=int, default=2, help="Number of attention heads (attn model).")
    add_split_arg(parser, "val", "val.npz")
    add_split_arg(parser, "test", "test.npz")
    args = parser.parse_args()

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata = load_metadata(args.dataset_dir)
    periodic = not args.no_periodic

    # Build datasets
    split_paths = {
        "train": resolve_split_path(args.dataset_dir, None, "train.npz"),
        "val": resolve_split_path(args.dataset_dir, args.val, "val.npz"),
        "test": resolve_split_path(args.dataset_dir, args.test, "test.npz"),
    }

    if not split_paths["train"].exists():
        raise FileNotFoundError(f"Train split not found at {split_paths['train']}")

    # Load train first to infer lattice shape from features or metadata.
    train_features, train_temps, train_labels = load_split(split_paths["train"])
    lattice_shape = infer_lattice_shape(train_features.shape[1], metadata)
    edge_index = build_lattice_edge_index(lattice_shape, periodic=periodic)

    def load_graph_split(path: Path):
        features, temps, labels = load_split(path)
        return make_graphs(features, temps, labels, lattice_shape, edge_index)

    train_graphs = load_graph_split(split_paths["train"])
    val_graphs = load_graph_split(split_paths["val"]) if split_paths["val"].exists() else []
    test_graphs = load_graph_split(split_paths["test"]) if split_paths["test"].exists() else []

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    # Separate eval loaders to keep a deterministic ordering for metrics/temperatures.
    train_eval_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False) if val_graphs else None
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False) if test_graphs else None

    if args.model_type == "gcn":
        model = GridGraphClassifier(
            input_dim=1,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
    else:
        model = AttentionLatticeClassifier(
            input_dim=1,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            edge_attr_dim=1,
            heads=args.heads,
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.n_epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        if epoch % max(1, args.n_epochs // 40) == 0 or epoch == args.n_epochs:
            msg = f"Epoch {epoch}/{args.n_epochs} - train_loss={loss:.4f}"
            if val_loader is not None:
                _, _, _, val_metrics = evaluate(model, val_loader, device)
                if val_metrics:
                    if "accuracy" in val_metrics:
                        msg += f" val_acc={val_metrics['accuracy']:.3f} val_f1={val_metrics['f1']:.3f}"
                    else:
                        msg += f" val_mae={val_metrics['mae']:.4f} val_rmse={val_metrics['rmse']:.4f}"
            print(msg)

    metrics_list = []
    splits_to_eval = [("train", train_eval_loader, train_temps)]
    if val_loader is not None:
        splits_to_eval.append(("val", val_loader, None))
    if test_loader is not None:
        splits_to_eval.append(("test", test_loader, None))

    split_outputs = {}
    for split_name, loader, temps_override in splits_to_eval:
        probs, labels, temps, metrics = evaluate(model, loader, device)
        if temps_override is not None:
            temps = temps_override  # keeps alignment with saved ordering if needed
        metrics["split"] = split_name
        metrics_list.append(metrics)
        split_outputs[split_name] = {"probs": probs, "labels": labels, "temps": temps}
        print_metrics(metrics)

    artifact_path = args.artifact
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": 1,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "heads": args.heads,
        },
        "model_type": args.model_type,
        "lattice_shape": lattice_shape,
        "periodic": periodic,
        "edge_index": edge_index,
        "metrics": metrics_list,
        "train_temperatures": split_outputs["train"]["temps"],
        "train_probabilities": split_outputs["train"]["probs"],
        "train_labels": split_outputs["train"]["labels"],
    }
    joblib.dump(artifact, artifact_path)
    print(f"Saved trained PyG lattice model to {artifact_path}")


if __name__ == "__main__":
    main()
