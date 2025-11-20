"""Inference and visualization for PyG lattice graph models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, TransformerConv, global_mean_pool

from data_utils import load_split
from metrics import estimate_critical_temperature, phase_metrics


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

    graphs: List[Data] = []
    edge_attr = torch.ones(edge_index.size(1), 1, dtype=torch.float32)
    for x_arr, temp, label in zip(features, temps, labels):
        x = torch.from_numpy(x_arr.reshape(-1, 1)).float()
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(label, dtype=torch.long),
            temp=torch.tensor(temp, dtype=torch.float32),
        )
        graphs.append(data)
    return graphs


class GridGraphClassifier(nn.Module):
    """GCN-based graph classifier; mirrors train_pyg configuration."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        self.convs = nn.ModuleList()
        dims = [input_dim] + [hidden_dim] * num_layers
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
    """Attention-based classifier matching the train_pyg attention model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        temp_embed_dim: int = 8,
        edge_attr_dim: int = 1,
        heads: int = 2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        self.temp_mlp = nn.Sequential(
            nn.Linear(1, temp_embed_dim),
            nn.ReLU(),
        )
        dims = [input_dim + temp_embed_dim] + [hidden_dim] * num_layers
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
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, data: Data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.size(1), 1, device=x.device, dtype=x.dtype)
        temp_emb = self.temp_mlp(data.temp.view(-1, 1))
        temp_per_node = temp_emb[data.batch]
        x = torch.cat([x, temp_per_node], dim=-1)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.head(x)


def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    all_temps = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.append(probs.cpu())
            all_labels.append(data.y.view(-1).cpu())
            all_temps.append(data.temp.view(-1).cpu())

    if not all_probs:
        return np.array([]), np.array([]), np.array([]), {}

    probs_np = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy()
    temps_np = torch.cat(all_temps).numpy()
    metrics = phase_metrics(labels_np, probs_np)
    metrics["t_c"] = estimate_critical_temperature(temps_np, probs_np)
    return probs_np, labels_np, temps_np, metrics


def smooth_probabilities(temperatures: np.ndarray, probs: np.ndarray, window: int = 72) -> np.ndarray:
    if window <= 1 or len(probs) < 3:
        return probs
    order = np.argsort(temperatures)
    probs_sorted = probs[order]
    kernel = np.ones(window, dtype=float) / window
    smoothed_sorted = np.convolve(probs_sorted, kernel, mode="same")
    smoothed = np.empty_like(probs_sorted)
    smoothed[:] = smoothed_sorted
    result = np.empty_like(probs)
    result[order] = smoothed
    return result


def plot_probability_scatter(ax, temps: np.ndarray, probs: np.ndarray, labels: np.ndarray, t_c: float, true_tc: float | None, method: str):
    scatter = ax.scatter(
        temps,
        probs,
        c=probs,
        cmap="plasma",
        s=45,
        edgecolor="black",
        linewidth=0.25,
        alpha=0.9,
        label="Graphs",
    )
    cb = plt.colorbar(scatter, ax=ax)
    cb.set_label("Predicted probability (phase=1)")

    predicted = probs >= 0.5
    misclassified = predicted != labels
    if np.any(misclassified):
        ax.scatter(
            temps[misclassified],
            probs[misclassified],
            facecolors="none",
            edgecolor="#222222",
            marker="x",
            linewidth=1.2,
            s=60,
            label="Misclassified",
        )

    ax.axvline(t_c, color="#222222", linestyle="--", label=f"Estimated T_c ({method})={t_c:.3f}")
    if true_tc is not None:
        ax.axvline(true_tc, color="#0b4f6c", linestyle="-.", label=f"Reference T_c={true_tc:.3f}")
    ax.axhline(0.5, color="#666666", linestyle=":")
    order = np.argsort(temps)
    temps_sorted = temps[order]
    probs_smoothed = smooth_probabilities(temps, probs)
    ax.plot(temps_sorted, probs_smoothed[order], color="#444444", linewidth=1.2, label="Moving average")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("P(phase = vortex)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=8)


def plot_probability_heatmap(ax, temps: np.ndarray, probs: np.ndarray, labels: np.ndarray):
    order = np.argsort(temps)
    ordered_temps = temps[order]
    ordered_probs = probs[order]
    ordered_labels = labels[order]
    heat_data = np.vstack([ordered_probs, ordered_labels])
    im = ax.imshow(
        heat_data,
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Value")
    ax.set_yticks([0, 1], ["Predicted probability", "True label"])
    ax.set_yticks([0.5, 1.5], minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    tick_positions = np.linspace(0, len(order) - 1, 6, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{ordered_temps[idx]:.2f}" for idx in tick_positions], rotation=45, ha="right")
    ax.set_xlabel("Temperature (sorted)")
    ax.set_title("Graph-wise heatmap")

def plot_temperature_windows(axes, temps: np.ndarray, probs: np.ndarray, labels: np.ndarray, t_c: float):
    order = np.argsort(temps)
    temps_sorted = temps[order]
    probs_sorted = probs[order]
    labels_sorted = labels[order]
    splits = np.array_split(np.arange(len(temps_sorted)), len(axes))

    for ax, idxs in zip(axes, splits):
        if len(idxs) == 0:
            continue
        t_chunk = temps_sorted[idxs]
        p_chunk = probs_sorted[idxs]
        l_chunk = labels_sorted[idxs]
        ax.scatter(
            t_chunk,
            p_chunk,
            c=p_chunk,
            cmap="plasma",
            s=35,
            edgecolor="black",
            linewidth=0.2,
            alpha=0.9,
        )
        ax.axvline(t_c, color="#555555", linestyle="--", linewidth=1)
        ax.axhline(0.5, color="#777777", linestyle=":", linewidth=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(t_chunk.min() - 0.02, t_chunk.max() + 0.02)
        ax.set_title(f"T ∈ [{t_chunk.min():.2f}, {t_chunk.max():.2f}]")
        ax.set_xlabel("Temperature")
        if ax is axes[0]:
            ax.set_ylabel("P(phase=1)")
        predicted = p_chunk >= 0.5
        misclassified = predicted != l_chunk
        if np.any(misclassified):
            ax.scatter(
                t_chunk[misclassified],
                p_chunk[misclassified],
                facecolors="none",
                edgecolor="#222222",
                marker="x",
                linewidth=1.0,
                s=50,
            )


def load_true_tc(dataset_path: Path, cli_tc: float | None) -> float | None:
    if cli_tc is not None:
        return cli_tc
    metadata_path = dataset_path.parent / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open() as f:
            metadata = json.load(f)
        return metadata.get("critical_temp")
    return None


def main():
    parser = argparse.ArgumentParser(description="Visualize PyG lattice graph predictions.")
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/pyg_lattice_model.joblib"))
    parser.add_argument("--dataset", type=Path, default=Path("../XYModel/blt_dataset/train.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/inference_plot.png"))
    parser.add_argument("--true-tc", type=float, default=None, help="Optional reference critical temperature.")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = joblib.load(args.artifact)
    model_cfg = artifact.get(
        "model_config",
        {"input_dim": 1, "hidden_dim": 64, "num_layers": 2, "dropout": 0.1, "temp_embed_dim": 8, "heads": 2},
    )
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
    model = None
    if model_type == "attn":
        model = AttentionLatticeClassifier(
            input_dim=model_cfg.get("input_dim", 1),
            hidden_dim=model_cfg.get("hidden_dim", 64),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.1),
            temp_embed_dim=model_cfg.get("temp_embed_dim", 8),
            edge_attr_dim=1,
            heads=model_cfg.get("heads", 2),
        ).to(device)
    else:
        model = GridGraphClassifier(
            input_dim=model_cfg.get("input_dim", 1),
            hidden_dim=model_cfg.get("hidden_dim", 64),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.1),
        ).to(device)
    state_dict = artifact.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Artifact is missing model_state_dict needed for inference.")
    model.load_state_dict(state_dict)

    probs, labels_out, temps_out, _ = evaluate(model, loader, device)
    true_tc = load_true_tc(args.dataset, args.true_tc)
    probs_smoothed = smooth_probabilities(temps_out, probs)
    t_c = estimate_critical_temperature(temps_out, probs_smoothed)
    print(f"Estimated critical temperature (smoothed crossing): {t_c:.3f}")
    if true_tc is not None:
        print(f"Reference critical temperature: {true_tc:.3f}")

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9])
    ax_scatter = fig.add_subplot(gs[0, :2])
    ax_heatmap = fig.add_subplot(gs[0, 2])
    window_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]

    plot_probability_scatter(ax_scatter, temps_out, probs, labels_out, t_c, true_tc, method="smoothed crossing")
    plot_probability_heatmap(ax_heatmap, temps_out, probs, labels_out)
    plot_temperature_windows(window_axes, temps_out, probs, labels_out, t_c)

    fig.suptitle("PyG lattice inference: probability view", fontsize=14)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Saved visualization to {args.output}")

if __name__ == "__main__":
    main()
