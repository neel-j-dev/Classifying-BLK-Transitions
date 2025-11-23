"""Utility for building BLT transition datasets from the XY model simulation.

This script relies on the original XY Metropolis simulation (xy.py) that lives
outside the 4ML3 folder. It now captures the full lattice configuration at each
temperature, treating every lattice point as a graph node whose feature is the
angular value at that position. The generated splits can be consumed by the GNN
pipelines living under 4ML3/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from xy import XYModelMetropolisSimulation


def extract_lattice_vectors(sim: XYModelMetropolisSimulation) -> np.ndarray:
    """Flatten the latest lattice snapshot into per-node 2D vectors (cos, sin)."""

    angles = 2 * np.pi * sim.L.astype(np.float64)
    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)
    stacked = np.stack([cos_vals, sin_vals], axis=-1)  # shape (H, W, 2)
    return stacked.reshape(-1, 2)


def generate_samples(
    min_temp: float,
    max_temp: float,
    num_temps: int,
    samples_per_temp: int,
    lattice_shape: Tuple[int, int],
    steps: int,
    iters_per_step: int,
    seed: int,
    coupling: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the XY simulation at many temperatures and collect observables."""

    rng = np.random.default_rng(seed)
    temps = np.linspace(min_temp, max_temp, num_temps)
    records: List[np.ndarray] = []
    temp_targets: List[float] = []

    for temp in temps:
        print(f"Generating samples at T={temp:.3f}...")
        beta = 1.0 / temp
        for _ in range(samples_per_temp):
            sim = XYModelMetropolisSimulation(
                lattice_shape=lattice_shape,
                beta=beta,
                J=coupling,
                random_state=rng.integers(0, 1_000_000_000),
            )
            sim.simulate(steps=steps, iters_per_step=iters_per_step)
            records.append(extract_lattice_vectors(sim).reshape(-1))
            temp_targets.append(temp)

    return np.vstack(records), np.array(temp_targets)


def split_indices(num_samples: int, train_ratio: float, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    train_end = int(train_ratio * num_samples)
    val_end = train_end + int(val_ratio * num_samples)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return train_idx, val_idx, test_idx


def save_split(path: Path, features: np.ndarray, temps: np.ndarray, labels: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=features, temperatures=temps, labels=labels)


def main():
    parser = argparse.ArgumentParser(description="Generate BLT dataset splits.")
    parser.add_argument("--output-dir", type=Path, default=Path("XYModel/blt_dataset"))
    parser.add_argument("--min-temp", type=float, default=0.3)
    parser.add_argument("--max-temp", type=float, default=1.5)
    parser.add_argument("--num-temps", type=int, default=1000)
    parser.add_argument("--samples-per-temp", type=int, default=10)
    parser.add_argument("--lattice-size", type=int, default=50)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--iters-per-step", type=int, default=40000)
    parser.add_argument("--critical-temp", type=float, default=0.89)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    lattice_shape = (args.lattice_size, args.lattice_size)

    features, temps = generate_samples(
        min_temp=args.min_temp,
        max_temp=args.max_temp,
        num_temps=args.num_temps,
        samples_per_temp=args.samples_per_temp,
        lattice_shape=lattice_shape,
        steps=args.steps,
        iters_per_step=args.iters_per_step,
        seed=args.seed,
        coupling=args.coupling,
    )

    labels = (temps >= args.critical_temp).astype(np.int32)
    train_idx, val_idx, test_idx = split_indices(
        num_samples=len(temps),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    output_dir: Path = args.output_dir
    save_split(output_dir / "train.npz", features[train_idx], temps[train_idx], labels[train_idx])
    save_split(output_dir / "val.npz", features[val_idx], temps[val_idx], labels[val_idx])
    save_split(output_dir / "test.npz", features[test_idx], temps[test_idx], labels[test_idx])

    metadata = {
        "min_temp": args.min_temp,
        "max_temp": args.max_temp,
        "num_temps": args.num_temps,
        "samples_per_temp": args.samples_per_temp,
        "lattice_shape": lattice_shape,
        "steps": args.steps,
        "iters_per_step": args.iters_per_step,
        "critical_temp": args.critical_temp,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "feature_type": "lattice_vectors",
        "feature_shape": lattice_shape,
        "feature_length": int(features.shape[1]),
        "features_per_node": 2,
    }

    metadata_path = output_dir / "metadata.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Saved train/val/test splits to '{output_dir}' with {len(temps)} samples.")


if __name__ == "__main__":
    main()
