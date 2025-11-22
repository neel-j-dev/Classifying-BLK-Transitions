from __future__ import annotations

"""Convert XYModel lattice samples into PyG graphs with random-walk positional encodings."""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch_geometric.transforms import AddRandomWalkPE

from utils.data_utils import load_split
from utils.lattice_utils import (
    build_lattice_edge_index,
    infer_lattice_shape,
    load_metadata,
    make_lattice_graphs,
)


def convert_split(
    split_name: str,
    split_path: Path,
    lattice_shape,
    edge_index,
    output_dir: Path,
    walk_length: int,
    periodic: bool,
):
    """Load a dense split, attach RWPE, and save PyG graphs."""

    features, temps, labels = load_split(split_path)
    graphs = make_lattice_graphs(features, temps, lattice_shape, edge_index, labels=labels)
    rw_transform = AddRandomWalkPE(walk_length=walk_length, attr_name="rw_pe")
    graphs = [rw_transform(g) for g in graphs]

    target_path = output_dir / f"{split_name}.pt"
    torch.save(graphs, target_path)
    print(f"Saved {len(graphs)} graphs to {target_path} (periodic_edges={periodic}).")


def main():
    parser = argparse.ArgumentParser(description="Build RWPE-enhanced PyG graphs for BLT detection.")
    parser.add_argument("--source-dir", type=Path, default=Path("../XYModel/blt_dataset"), help="Directory with *.npz splits.")
    parser.add_argument("--output-dir", type=Path, default=Path("RWGNN/blt_rwgnn_dataset"), help="Where to write *.pt splits.")
    parser.add_argument("--walk-length", type=int, default=16, help="Random walk length for positional encodings.")
    parser.add_argument("--no-periodic", action="store_true", help="Disable periodic boundary conditions.")
    parser.add_argument("--train-name", type=str, default="train.npz")
    parser.add_argument("--val-name", type=str, default="val.npz")
    parser.add_argument("--test-name", type=str, default="test.npz")
    args = parser.parse_args()

    src_dir: Path = args.source_dir
    metadata: Dict | None = load_metadata(src_dir)
    if metadata is None:
        raise FileNotFoundError(f"metadata.json not found under {src_dir}; generate the base dataset first.")

    train_path = src_dir / args.train_name
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train split at {train_path}; run XYModel/generate_blt_dataset.py.")

    train_features, _, _ = load_split(train_path)
    lattice_shape = infer_lattice_shape(train_features.shape[1], metadata)
    periodic = not args.no_periodic
    edge_index = build_lattice_edge_index(lattice_shape, periodic=periodic)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, filename in [("train", args.train_name), ("val", args.val_name), ("test", args.test_name)]:
        split_path = src_dir / filename
        if not split_path.exists():
            print(f"Skipping missing split {filename}")
            continue
        convert_split(
            split_name=name,
            split_path=split_path,
            lattice_shape=lattice_shape,
            edge_index=edge_index,
            output_dir=args.output_dir,
            walk_length=args.walk_length,
            periodic=periodic,
        )

    rw_metadata = {
        "source_dataset": str(src_dir),
        "output_dir": str(args.output_dir),
        "lattice_shape": lattice_shape,
        "periodic_edges": periodic,
        "walk_length": args.walk_length,
        "features_per_node": metadata.get("features_per_node", 2),
        "critical_temp": metadata.get("critical_temp"),
        "min_temp": metadata.get("min_temp"),
        "max_temp": metadata.get("max_temp"),
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(rw_metadata, indent=2))
    print(f"Wrote RWGNN metadata to {metadata_path}")


if __name__ == "__main__":
    main()
