#!/usr/bin/env python3
"""Run a vanilla diffusion map on stored Ising configurations."""

from __future__ import annotations

import argparse
import csv
import pathlib
from typing import Dict, Optional

import numpy as np

from diffusion_map import diffusion_map


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "XYModel" / "ising_model_configurations.npy"
DEFAULT_METADATA = REPO_ROOT / "XYModel" / "ising_model_metadata.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the diffusion map algorithm directly to stored Ising configurations. "
            "Results are saved as an .npz archive along with an optional scatter plot."
        )
    )
    parser.add_argument(
        "--config-path",
        type=pathlib.Path,
        default=DEFAULT_CONFIG,
        help=f"Path to the numpy array of Ising configurations (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--metadata-path",
        type=pathlib.Path,
        default=DEFAULT_METADATA,
        help=f"Optional metadata CSV (default: {DEFAULT_METADATA})",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Kernel bandwidth. If omitted, the median pairwise squared distance is used.",
    )
    parser.add_argument(
        "--epsilon-samples",
        type=int,
        default=2000,
        help="Number of samples to use when estimating epsilon (ignored when epsilon is provided).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Randomly subsample to this many configurations (default: use all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for the subsampling procedure.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=10,
        help="Number of eigenvectors/eigenvalues to compute.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=3,
        help="Number of diffusion map dimensions to retain (excluding the trivial eigenvector).",
    )
    parser.add_argument(
        "--diffusion-time",
        type=int,
        default=1,
        help="Diffusion time exponent applied to eigenvalues.",
    )
    parser.add_argument(
        "--use-sparse",
        action="store_true",
        help="Use sparse eigendecomposition (recommended only for large datasets).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("simple_diffusion_results.npz"),
        help="Output .npz file to store embedding, eigenvalues, and metadata.",
    )
    parser.add_argument(
        "--plot",
        type=pathlib.Path,
        default=None,
        help="Optional path to save a ψ₁–ψ₂ scatter plot colored by temperature.",
    )
    return parser.parse_args()


def load_metadata(metadata_path: pathlib.Path) -> Optional[Dict[str, np.ndarray]]:
    if metadata_path is None or not metadata_path.exists():
        return None

    records: Dict[str, list[float]] = {"temperature": [], "magnetization": []}
    with metadata_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            records["temperature"].append(float(row["temperature"]))
            records["magnetization"].append(float(row["magnetization"]))

    return {key: np.asarray(values, dtype=np.float64) for key, values in records.items()}


def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    norms = np.sum(X ** 2, axis=1, keepdims=True)
    dists = norms + norms.T - 2 * X @ X.T
    np.maximum(dists, 0.0, out=dists)
    return dists


def estimate_epsilon(
    X: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> float:
    if X.shape[0] <= 1:
        raise ValueError("Need at least two samples to estimate epsilon.")

    if sample_size is not None and X.shape[0] > sample_size:
        subset_idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        subset = X[subset_idx]
    else:
        subset = X

    dists = _pairwise_sq_dists(subset)
    triu = dists[np.triu_indices_from(dists, k=1)]
    epsilon = float(np.median(triu))
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Failed to compute a positive epsilon; please provide one manually.")
    return epsilon


def subset_indices(n_total: int, max_samples: Optional[int], rng: np.random.Generator) -> np.ndarray:
    if max_samples is None or max_samples >= n_total:
        return np.arange(n_total)
    return np.sort(rng.choice(n_total, size=max_samples, replace=False))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    configs = np.load(args.config_path)
    n_total = configs.shape[0]
    flattened = configs.reshape(n_total, -1).astype(np.float64)
    selected_idx = subset_indices(n_total, args.max_samples, rng)
    X_subset = flattened[selected_idx]

    epsilon = args.epsilon
    if epsilon is None:
        epsilon = estimate_epsilon(X_subset, args.epsilon_samples, rng)
        print(f"Estimated epsilon from pairwise distances: {epsilon:.6g}")
    else:
        print(f"Using provided epsilon: {epsilon:.6g}")

    metadata = load_metadata(args.metadata_path)
    if metadata is not None and metadata["temperature"].shape[0] != n_total:
        print(
            "Warning: metadata size does not match number of configurations. "
            "Metadata will be ignored."
        )
        metadata = None

    print(f"Running diffusion map on {X_subset.shape[0]} configs, {X_subset.shape[1]} features each...")
    results = diffusion_map(
        X_subset,
        epsilon=epsilon,
        n_components=args.n_components,
        n_dimensions=args.dimensions,
        t=args.diffusion_time,
        N=X_subset.shape[1],
        use_sparse=args.use_sparse,
        verbose=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_payload = {
        "embedding": results["embedding"],
        "eigenvalues": results["eigenvalues"],
        "eigenvectors": results["eigenvectors"],
        "subset_indices": selected_idx,
        "epsilon": np.asarray([epsilon], dtype=np.float64),
    }
    if metadata is not None:
        save_payload["temperature"] = metadata["temperature"][selected_idx]
        save_payload["magnetization"] = metadata["magnetization"][selected_idx]
    np.savez(args.output, **save_payload)
    print(f"Saved diffusion outputs to {args.output}")

    if args.plot is not None:
        if metadata is None:
            colors = selected_idx
            color_label = "Sample index"
            cmap = "viridis"
        else:
            colors = metadata["temperature"][selected_idx]
            color_label = "Temperature T/J"
            cmap = "plasma"

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            print(f"Matplotlib is not installed; skipping scatter plot ({exc}).")
            return

        coords = results["embedding"]
        if coords.shape[1] < 2:
            raise ValueError("Need at least two diffusion coordinates to render a scatter plot.")

        plt.figure(figsize=(7, 6))
        sc = plt.scatter(coords[:, 0], coords[:, 1], c=colors, cmap=cmap, s=35, alpha=0.85)
        plt.xlabel("ψ₁")
        plt.ylabel("ψ₂")
        plt.title("Diffusion map of Ising configurations")
        plt.grid(True, alpha=0.25)
        cbar = plt.colorbar(sc)
        cbar.set_label(color_label)
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=200)
        plt.close()
        print(f"Saved ψ₁–ψ₂ scatter plot to {args.plot}")


if __name__ == "__main__":
    main()
