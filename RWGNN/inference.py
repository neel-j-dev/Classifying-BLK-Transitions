from __future__ import annotations

"""Train, validate, and test a Random-Walk-PE GNN for BLT detection."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


class RandomWalkGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            self.convs.append(GCNConv(in_dim, hidden_dim))
        self.dropout = dropout
        self.phase_head = nn.Linear(hidden_dim, 2)
        self.temp_head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x = data.x
        if hasattr(data, "rw_pe"):
            x = torch.cat([x, data.rw_pe], dim=-1)
        for conv in self.convs:
            x = conv(x, data.edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, data.batch)
        phase_logits = self.phase_head(x)
        temp_pred = self.temp_head(x).squeeze(-1)
        return phase_logits, temp_pred


def load_graph_split(path: Path) -> List:
    graphs = torch.load(path, weights_only=False)
    if not isinstance(graphs, list):
        raise ValueError(f"Expected a list of Data objects in {path}")
    return graphs


def step_epoch(model, loader, optimizer, device, temp_weight: float) -> float:
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        phase_logits, temp_pred = model(batch)
        cls_loss = F.cross_entropy(phase_logits, batch.y)
        temp_loss = F.mse_loss(temp_pred, batch.temp)
        loss = cls_loss + temp_weight * temp_loss
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * batch.num_graphs
        total_graphs += batch.num_graphs
    return total_loss / max(total_graphs, 1)


def evaluate(model, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_temps: List[torch.Tensor] = []
    all_temp_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            phase_logits, temp_pred = model(batch)
            all_logits.append(phase_logits.cpu())
            all_temps.append(batch.temp.cpu())
            all_temp_preds.append(temp_pred.cpu())
            all_labels.append(batch.y.cpu())

    logits = torch.cat(all_logits)
    temps = torch.cat(all_temps)
    temp_preds = torch.cat(all_temp_preds)
    labels = torch.cat(all_labels)

    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    mae = mean_absolute_error(temps, temp_preds)
    rmse = float(np.sqrt(mean_squared_error(temps, temp_preds)))
    metrics = {"accuracy": acc, "f1": f1, "mae": mae, "rmse": rmse}

    return preds.numpy(), temps.numpy(), temp_preds.numpy(), probs.numpy(), metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Phase confusion matrix")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_temperature_regression(true_t: np.ndarray, pred_t: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.scatter(true_t, pred_t, alpha=0.6, label="samples")
    slope, intercept = np.polyfit(true_t, pred_t, 1)
    xs = np.linspace(true_t.min(), true_t.max(), 100)
    plt.plot(xs, slope * xs + intercept, color="red", label=f"y={slope:.2f}x+{intercept:.2f}")
    plt.xlabel("True temperature")
    plt.ylabel("Predicted temperature")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def estimate_critical_temperature(temps: np.ndarray, phase_probs: np.ndarray) -> float:
    """
    Estimate the critical temperature as the point where the predicted phase
    probability crosses 0.5. If no crossing exists, return the mean
    temperature as a fallback.
    """

    if phase_probs.ndim != 2 or phase_probs.shape[1] < 2:
        raise ValueError("phase_probs must be shape (N, 2) with class probabilities")

    order = np.argsort(temps)
    t_sorted = temps[order]
    p_sorted = phase_probs[order, 1]

    mask = (p_sorted >= 0.5).astype(int)
    crossing_idx = np.where(np.diff(mask) != 0)[0]
    if len(crossing_idx) == 0:
        return float(t_sorted.mean())

    idx = crossing_idx[0]
    t0, t1 = t_sorted[idx], t_sorted[idx + 1]
    p0, p1 = p_sorted[idx], p_sorted[idx + 1]
    if p1 == p0:
        return float((t0 + t1) / 2.0)

    alpha = (0.5 - p0) / (p1 - p0)
    return float(t0 + alpha * (t1 - t0))


def main():
    parser = argparse.ArgumentParser(description="Train and test RWPE GNN for BLT transitions.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("blt_rwgnn_dataset"))
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temp-weight", type=float, default=1.0, help="Weight for temperature MSE in loss.")
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/rwgnn_model.joblib"))
    parser.add_argument("--confusion-path", type=Path, default=Path("artifacts/confusion_matrix.png"))
    parser.add_argument("--regression-path", type=Path, default=Path("artifacts/temperature_regression.png"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    metadata_path = args.dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata.json at {metadata_path}; run gen_dataset_rwgnn.py first.")
    metadata: Dict = json.loads(metadata_path.read_text())

    train_graphs = load_graph_split(args.dataset_dir / "train.pt")
    val_graphs = load_graph_split(args.dataset_dir / "val.pt")
    test_graphs = load_graph_split(args.dataset_dir / "test.pt")

    walk_length = metadata["walk_length"]
    node_feat_dim = metadata.get("features_per_node", 2)
    input_dim = node_feat_dim + walk_length

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    model = RandomWalkGNN(input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = step_epoch(model, train_loader, optimizer, device, temp_weight=args.temp_weight)
        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                _, _, _, _, val_metrics = evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch}/{args.epochs} - loss={loss:.4f} "
                f"val_acc={val_metrics['accuracy']:.3f} val_f1={val_metrics['f1']:.3f} "
                f"val_mae={val_metrics['mae']:.4f} val_rmse={val_metrics['rmse']:.4f}"
            )

    # Final evaluations
    def collect_metrics(split: str, loader):
        preds, true_t, pred_t, probs, metrics = evaluate(model, loader, device)
        metrics["split"] = split
        return preds, true_t, pred_t, probs, metrics

    train_preds, train_t, train_pred_t, train_probs, train_metrics = collect_metrics("train", train_loader)
    val_preds, val_t, val_pred_t, val_probs, val_metrics = collect_metrics("val", val_loader)
    test_preds, test_t, test_pred_t, test_probs, test_metrics = collect_metrics("test", test_loader)

    for metrics in [train_metrics, val_metrics, test_metrics]:
        print(
            f"[{metrics['split']}] acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} "
            f"mae={metrics['mae']:.4f} rmse={metrics['rmse']:.4f}"
        )

    # Confusion matrix and regression plot for test split
    # Ensure labels are 1D before plotting; individual graphs may store a scalar label.
    y_true_test = np.array([int(batch.y.view(-1)[0].cpu()) for batch in test_graphs])
    plot_confusion_matrix(y_true_test, test_preds, args.confusion_path)
    plot_temperature_regression(test_t, test_pred_t, args.regression_path)
    critical_temp = estimate_critical_temperature(test_t, test_probs)
    print(f"Estimated critical temperature (predicted) ≈ {critical_temp:.4f}")
    print(f"Saved confusion matrix to {args.confusion_path}")
    print(f"Saved temperature regression plot to {args.regression_path}")

    artifact = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "temp_weight": args.temp_weight,
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": [train_metrics, val_metrics, test_metrics],
        "metadata": metadata,
        "predicted_critical_temp": critical_temp,
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.artifact)
    print(f"Stored model and metrics at {args.artifact}")


if __name__ == "__main__":
    main()