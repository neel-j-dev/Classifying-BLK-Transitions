"""
Automatic tuning of diffusion-map hyperparameters (epsilon and cluster count).

Implements the adjusted-MSE heuristic from Kerr et al. (Automatic Learning of
Topological Phase Boundaries) to jointly select the Gaussian kernel bandwidth
epsilon and the number of k-means clusters. This module mirrors the notebook
logic so it can be reused from scripts or the CLI.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

from diffusion_map import (
    compute_eigendecomposition,
    compute_gaussian_kernel,
    construct_diffusion_map_embedding,
    construct_transition_matrix,
)


@dataclass
class TuningResult:
    epsilon: float
    n_clusters: int
    mse: float
    epsilon_idx: int
    cluster_idx: int
    eigenvalues: np.ndarray
    labels: np.ndarray


def adjusted_mse(K: np.ndarray, labels: np.ndarray) -> float:
    """Adjusted MSE between similarity matrix and an ideal block-diagonal target."""
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    ideal = (labels[:, None] == labels[None, :]).astype(float)
    diff_sq = (K - ideal) ** 2
    mse = 0.0
    for lbl in unique_labels:
        mask = labels == lbl
        # Weight each cluster equally regardless of size.
        mse += diff_sq[mask].sum() / mask.sum()
    return (n_clusters - 1) / n_clusters * mse


def resolution_grid_from_data(
    X: np.ndarray,
    num: int = 10,
    low_q: float = 5.0,
    high_q: float = 95.0,
) -> np.ndarray:
    """Construct a logarithmic epsilon grid from data quantiles."""
    pairwise_sq = pdist(X, metric="sqeuclidean")
    nonzero = pairwise_sq[pairwise_sq > 0]
    if nonzero.size == 0:
        raise ValueError("All pairwise distances are zero; cannot build epsilon grid.")
    low = np.percentile(nonzero, low_q)
    high = np.percentile(pairwise_sq, high_q)
    eps_min = max(low / (2 * X.shape[1]), 1e-12)
    eps_max = max(high / (2 * X.shape[1]), eps_min * 10)
    return np.logspace(np.log10(eps_min), np.log10(eps_max), num=num)


def scan_resolution_and_clusters(
    X: np.ndarray,
    epsilon_values: Sequence[float],
    cluster_options: Sequence[int],
    t: int = 1,
    n_components: int = 15,
    random_state: Optional[int] = 0,
    n_jobs: Optional[int] = None,
    use_torch: bool = False,
    device: Optional[str] = None,
) -> Tuple[TuningResult, List[TuningResult], np.ndarray]:
    """Evaluate adjusted MSE over a grid of epsilon and k-means cluster counts."""
    results: List[TuningResult] = []
    best: Optional[TuningResult] = None
    mse_surface = np.full((len(cluster_options), len(epsilon_values)), np.nan)

    def _evaluate_single_epsilon(eps_idx: int, eps: float) -> Tuple[int, List[TuningResult]]:
        K = compute_gaussian_kernel(
            X,
            epsilon=eps,
            N=X.shape[1],
            use_torch=use_torch,
            device=device,
            return_torch=False,
        )
        P = construct_transition_matrix(K)
        evals, evecs = compute_eigendecomposition(
            P,
            n_components=min(n_components, X.shape[0] - 1),
            use_torch=use_torch,
            device=device,
            return_torch=False,
        )
        n_dims = min(3, len(evals) - 1)
        embedding = construct_diffusion_map_embedding(
            evals, evecs, n_dimensions=n_dims, t=t, skip_first=True
        )

        eps_results: List[TuningResult] = []
        for n_idx, n_clusters in enumerate(cluster_options):
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
            labels = kmeans.fit_predict(embedding)
            mse_val = adjusted_mse(K, labels)
            eps_results.append(
                TuningResult(
                    epsilon=eps,
                    n_clusters=n_clusters,
                    mse=mse_val,
                    epsilon_idx=eps_idx,
                    cluster_idx=n_idx,
                    eigenvalues=evals,
                    labels=labels,
                )
            )
        return eps_idx, eps_results

    if n_jobs is None or n_jobs == 1:
        eps_outputs = [_evaluate_single_epsilon(idx, eps) for idx, eps in enumerate(epsilon_values)]
    else:
        eps_outputs = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_evaluate_single_epsilon)(idx, eps) for idx, eps in enumerate(epsilon_values)
        )

    for eps_idx, eps_results in eps_outputs:
        for result in eps_results:
            results.append(result)
            mse_surface[result.cluster_idx, eps_idx] = result.mse
            if best is None or result.mse < best.mse:
                best = result

    if best is None:
        raise RuntimeError("Tuning scan produced no results.")
    return best, results, mse_surface


def tune_epsilon(
    X: np.ndarray,
    cluster_options: Sequence[int] = (2, 3, 4, 5, 6),
    epsilon_values: Optional[Sequence[float]] = None,
    max_samples: int = 500,
    random_state: Optional[int] = 0,
    t: int = 1,
    n_components: int = 15,
    n_jobs: Optional[int] = None,
    use_torch: bool = False,
    device: Optional[str] = None,
) -> Tuple[TuningResult, List[TuningResult], np.ndarray, np.ndarray]:
    """End-to-end helper: subsample, build epsilon grid, and run the heuristic."""
    rng = np.random.default_rng(random_state)
    if X.shape[0] > max_samples:
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X_used = X[idx]
    else:
        X_used = X

    if epsilon_values is None:
        epsilon_values = resolution_grid_from_data(X_used)
    epsilon_values = np.asarray(list(epsilon_values), dtype=float)

    best, results, mse_surface = scan_resolution_and_clusters(
        X_used,
        epsilon_values,
        cluster_options,
        t=t,
        n_components=n_components,
        random_state=random_state,
        n_jobs=n_jobs,
        use_torch=use_torch,
        device=device,
    )
    return best, results, mse_surface, epsilon_values


def _load_npy(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Automatic tuning of diffusion-map epsilon and cluster count."
    )
    parser.add_argument(
        "npy",
        type=Path,
        help="Path to a .npy file containing spin configurations (rows = samples).",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6],
        help="Candidate cluster counts for k-means.",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=None,
        help="Optional explicit epsilon list. If omitted, a grid is inferred from data.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum samples to use during tuning (random subset).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for subsampling and k-means.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel workers for the epsilon/k scan (joblib backend).",
    )
    parser.add_argument(
        "--use-torch",
        action="store_true",
        help="Use PyTorch (and CUDA if available) for kernel and eigendecomposition.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional torch device (e.g., 'cuda' or 'cuda:0'). Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to save the best tuning result.",
    )
    args = parser.parse_args(argv)

    X = _load_npy(args.npy)
    best, _, _, eps_grid = tune_epsilon(
        X,
        cluster_options=args.clusters,
        epsilon_values=args.epsilons,
        max_samples=args.max_samples,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        use_torch=args.use_torch,
        device=args.device,
    )

    print("Tuning complete.")
    print(f"  Best epsilon : {best.epsilon:.6f}")
    print(f"  Best clusters: {best.n_clusters}")
    print(f"  Adjusted MSE : {best.mse:.6f}")
    print(f"  Epsilon grid : [{eps_grid.min():.3e}, {eps_grid.max():.3e}] ({len(eps_grid)} values)")

    if args.output:
        payload = asdict(best)
        payload["eigenvalues"] = best.eigenvalues.tolist()
        payload["labels"] = best.labels.tolist()
        payload["epsilon_grid"] = eps_grid.tolist()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"✓ Saved result to {args.output}")


if __name__ == "__main__":
    main()
