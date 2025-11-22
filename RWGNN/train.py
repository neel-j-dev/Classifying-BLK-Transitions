"""Training script for the Random Walk GNN phase classifier and Tc estimator.

The pipeline can either consume pre-generated NPZ splits (see
``XYModel/generate_blt_dataset.py``) or draw fresh samples from the XY
Metropolis simulator. The model predicts the BLT phase, the underlying
temperature, and provides an uncertainty-aware estimate of the critical
temperature interval.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from RWGNN.data import SplitData, generate_from_simulator, load_xy_splits
from RWGNN.model import RandomWalkGNN


@dataclass
class Metrics:
    phase_acc: float
    temp_mae: float
    tc_mae: float


def _make_loaders(splits: SplitData, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return (
        DataLoader(splits.train, batch_size=batch_size, shuffle=True),
        DataLoader(splits.val, batch_size=batch_size),
        DataLoader(splits.test, batch_size=batch_size),
    )


def _split_graphs(graphs, train_ratio: float, val_ratio: float, seed: int) -> SplitData:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(graphs))
    rng.shuffle(indices)
    train_end = int(train_ratio * len(graphs))
    val_end = train_end + int(val_ratio * len(graphs))
    train_idx, val_idx, test_idx = indices[:train_end], indices[train_end:val_end], indices[val_end:]
    graphs_array = np.array(graphs, dtype=object)
    return SplitData(train=list(graphs_array[train_idx]), val=list(graphs_array[val_idx]), test=list(graphs_array[test_idx]))


def train_epoch(model: RandomWalkGNN, loader: DataLoader, optimizer: Adam, device: torch.device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss, _ = model.compute_losses(
            outputs,
            phase_target=batch.y,
            temp_target=batch.temperature,
            tc_target=batch.tc_target,
        )
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * batch.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model: RandomWalkGNN, loader: DataLoader, device: torch.device) -> Metrics:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    temp_errors = []
    tc_errors = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            loss, _ = model.compute_losses(
                outputs,
                phase_target=batch.y,
                temp_target=batch.temperature,
                tc_target=batch.tc_target,
            )
            total_loss += float(loss) * batch.num_graphs

            preds = outputs["phase_logits"].argmax(dim=-1)
            correct += int((preds == batch.y.view(-1)).sum())
            total += batch.num_graphs

            temp_errors.append(torch.abs(outputs["temp_mean"].squeeze(-1) - batch.temperature.view(-1)))
            tc_errors.append(torch.abs(outputs["tc_mean"].squeeze(-1) - batch.tc_target.view(-1)))

    phase_acc = correct / max(total, 1)
    temp_mae = torch.cat(temp_errors).mean().item() if temp_errors else 0.0
    tc_mae = torch.cat(tc_errors).mean().item() if tc_errors else 0.0
    return Metrics(phase_acc=phase_acc, temp_mae=temp_mae, tc_mae=tc_mae)


def fit(model: RandomWalkGNN, loaders: Tuple[DataLoader, DataLoader, DataLoader], device: torch.device, epochs: int, lr: float):
    optimizer = Adam(model.parameters(), lr=lr)
    train_loader, val_loader, _ = loaders

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_acc={val_metrics.phase_acc:.3f} | val_temp_mae={val_metrics.temp_mae:.3f} | "
            f"val_tc_mae={val_metrics.tc_mae:.3f}"
        )


def _load_or_generate(args: argparse.Namespace) -> Tuple[SplitData, int]:
    if args.use_simulator:
        temps = np.linspace(args.min_temp, args.max_temp, args.num_temps)
        graphs = generate_from_simulator(
            temperatures=temps,
            samples_per_temp=args.samples_per_temp,
            lattice_shape=(args.lattice_size, args.lattice_size),
            steps=args.steps,
            iters_per_step=args.iters_per_step,
            seed=args.seed,
            coupling=args.coupling,
            walk_length=args.walk_length,
            num_walks=args.num_walks,
            critical_temperature=args.critical_temp,
        )
        splits = _split_graphs(graphs, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
        feature_size = graphs[0].num_node_features
    else:
        data_dir = Path(args.dataset_dir)
        splits = load_xy_splits(data_dir, walk_length=args.walk_length, num_walks=args.num_walks)
        feature_size = splits.train[0].num_node_features
    return splits, feature_size


def _inference_demo(model: RandomWalkGNN, loader: DataLoader, device: torch.device, *, max_batches: int = 1):
    """Print a small Tc interval preview for sanity checking."""

    model.eval()
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch.to(device)
        phase_probs, lower, upper, tc_mean = model.predict_with_interval(batch)
        print("Sample phase probabilities (ordered, disordered):", phase_probs[:3].cpu().numpy())
        print("Tc mean ± 95% interval for first batch entries:")
        for j in range(min(3, batch.num_graphs)):
            print(f"  {tc_mean[j].item():.3f} -> [{lower[j].item():.3f}, {upper[j].item():.3f}]")


def main():
    parser = argparse.ArgumentParser(description="Train Random Walk GNN for BLT phase detection.")
    parser.add_argument("--dataset-dir", type=str, default="XYModel/blt_dataset", help="Directory with train/val/test NPZ splits.")
    parser.add_argument("--use-simulator", action="store_true", help="Generate fresh samples instead of loading NPZ splits.")
    parser.add_argument("--min-temp", type=float, default=0.3)
    parser.add_argument("--max-temp", type=float, default=1.5)
    parser.add_argument("--num-temps", type=int, default=120)
    parser.add_argument("--samples-per-temp", type=int, default=2)
    parser.add_argument("--lattice-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--iters-per-step", type=int, default=20000)
    parser.add_argument("--critical-temp", type=float, default=0.89)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--walk-length", type=int, default=4)
    parser.add_argument("--num-walks", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden-channels", type=int, default=128)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits, feature_size = _load_or_generate(args)
    train_loader, val_loader, test_loader = _make_loaders(splits, args.batch_size)

    model = RandomWalkGNN(in_channels=feature_size, hidden_channels=args.hidden_channels).to(device)
    fit(model, (train_loader, val_loader, test_loader), device, epochs=args.epochs, lr=args.lr)

    test_metrics = evaluate(model, test_loader, device)
    print(
        f"Test accuracy={test_metrics.phase_acc:.3f}, "
        f"Temp MAE={test_metrics.temp_mae:.3f}, Tc MAE={test_metrics.tc_mae:.3f}"
    )
    _inference_demo(model, val_loader, device)

    model_path = Path("artifacts") / "rwg_nn.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "in_channels": feature_size}, model_path)
    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    main()
