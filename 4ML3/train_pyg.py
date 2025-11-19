from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from data_utils import FEATURE_NAMES
from pipeline import run_split


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
    print(
        f"[{metrics['split']}] accuracy={metrics['accuracy']:.3f} "
        f"f1={metrics['f1']:.3f} t_c={metrics['t_c']:.3f} "
        f"roc_auc={metrics['roc_auc']:.3f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a PyTorch Geometric model on BLT data."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("../blt_dataset_large"))
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/pyg_model.joblib"))
    add_split_arg(parser, "val", "val.npz")
    add_split_arg(parser, "test", "test.npz")
    args = parser.parse_args()

    splits_to_run = [
        ("train", resolve_split_path(args.dataset_dir, None, "train.npz")),
    ]
    val_path = resolve_split_path(args.dataset_dir, args.val, "val.npz")
    if val_path.exists():
        splits_to_run.append(("val", val_path))
    test_path = resolve_split_path(args.dataset_dir, args.test, "test.npz")
    if test_path.exists():
        splits_to_run.append(("test", test_path))

    models = []
    metrics_list = []

    for split_name, split_path in splits_to_run:
        probs, metrics, temps, model, scaler = run_split(
            split_path,
            split_name=split_name,
            model_type="pyg",
            hidden_dim=args.hidden_dim,
            n_epochs=args.n_epochs,
            learning_rate=args.lr,
            random_state=args.random_state,
            n_neighbors=args.n_neighbors,
            metric=args.metric,
        )
        models.append((split_name, model))
        metrics_list.append((split_name, metrics, temps, probs, scaler))

    for _, metrics, _, _, _ in metrics_list:
        print_metrics(metrics)

    # Persist the training artifacts for reuse in downstream scripts (e.g., inference).
    train_split = next((item for item in metrics_list if item[0] == "train"), None)
    if train_split is None:
        raise RuntimeError("Training split did not run successfully.")

    _, train_metrics, train_temps, train_probs, train_scaler = train_split
    train_model = next((model for name, model in models if name == "train"), None)

    artifact_path = args.artifact
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": train_model,
        "scaler": train_scaler,
        "n_neighbors": args.n_neighbors,
        "metric": args.metric,
        "model_type": "pyg",
        "feature_names": FEATURE_NAMES,
        "train_temperatures": train_temps,
        "train_probabilities": train_probs,
        "train_metrics": train_metrics,
    }
    joblib.dump(artifact, artifact_path)
    print(f"Saved trained PyG model to {artifact_path}")


if __name__ == "__main__":
    main()
