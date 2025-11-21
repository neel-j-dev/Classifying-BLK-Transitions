from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from GNN.utils.data_utils import FEATURE_NAMES
from GNN.__legacy__.pipeline import run_split


def main():
    parser = argparse.ArgumentParser(description="Train a scikit-network GNN on BLT data.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("../XYModel/blt_dataset"))
    parser.add_argument("--model-type", choices=["gnn", "gat"], default="gnn")
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/gnn_model.joblib"))
    parser.add_argument("--val", type=Path, default=None, help="Optional path to a validation split.")
    args = parser.parse_args()

    train_path = args.dataset_dir / "train.npz"
    train_probs, train_metrics, train_temps, model, scaler = run_split(
        train_path,
        split_name="train",
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        random_state=args.random_state,
        n_neighbors=args.n_neighbors,
        metric=args.metric,
    )

    all_metrics = [train_metrics]
    val_path = args.val if args.val is not None else args.dataset_dir / "val.npz"
    if val_path.exists():
        _, val_metrics, _, _, _ = run_split(
            val_path,
            split_name="val",
            model_type=args.model_type,
            hidden_dim=args.hidden_dim,
            n_epochs=args.n_epochs,
            learning_rate=args.lr,
            random_state=args.random_state,
            n_neighbors=args.n_neighbors,
            metric=args.metric,
        )
        all_metrics.append(val_metrics)

    for metric in all_metrics:
        print(f"[{metric['split']}] accuracy={metric['accuracy']:.3f} f1={metric['f1']:.3f} "
              f"t_c={metric['t_c']:.3f} roc_auc={metric['roc_auc']:.3f}")

    artifact_path = args.artifact
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "scaler": scaler,
        "n_neighbors": args.n_neighbors,
        "metric": args.metric,
        "model_type": args.model_type,
        "feature_names": FEATURE_NAMES,
        "train_temperatures": train_temps,
        "train_probabilities": train_probs,
    }
    joblib.dump(artifact, artifact_path)
    print(f"Saved trained {args.model_type} model to {artifact_path}")


if __name__ == "__main__":
    main()
