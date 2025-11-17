from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import run_split


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate a GNN on the validation split.")
    parser.add_argument("--split", type=Path, default=Path("../blt_dataset/val.npz"))
    parser.add_argument("--model-type", choices=["gnn", "gat"], default="gnn")
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=1)
    args = parser.parse_args()

    _, stats, _, _, _ = run_split(
        args.split,
        split_name="val",
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        random_state=args.random_state,
        n_neighbors=args.n_neighbors,
        metric=args.metric,
    )
    print(f"[val] accuracy={stats['accuracy']:.3f} f1={stats['f1']:.3f} "
          f"t_c={stats['t_c']:.3f} roc_auc={stats['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
