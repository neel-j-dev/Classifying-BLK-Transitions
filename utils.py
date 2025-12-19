from __future__ import annotations

import dataclasses
import struct
from pathlib import Path
from typing import Callable, Protocol

import numpy as np


@dataclasses.dataclass(frozen=True)
class Dataset:
    temps: np.ndarray
    X: np.ndarray
    obs: dict[str, np.ndarray]
    configs: np.ndarray | None = None


def load_ising_configs_bin(path: str | Path, block: int | None = None) -> Dataset:
    path = Path(path)
    with path.open("rb") as f:
        magic = f.read(8)
        if not magic.startswith(b"ISCFG1"):
            raise ValueError(f"Unexpected magic {magic!r} in {path}")

        lattice_L, n_temps, n_samples, align_z2 = struct.unpack("<4i", f.read(16))
        n_temps = int(n_temps)
        n_samples = int(n_samples)
        size = int(lattice_L) * int(lattice_L)

        temps = np.empty(n_temps, dtype=np.float32)
        configs = np.empty((n_temps, n_samples, size), dtype=np.int8)
        mabs = np.empty((n_temps, n_samples), dtype=np.float32)
        energy = np.empty((n_temps, n_samples), dtype=np.float32)

        rec_dtype = np.dtype(
            [
                ("spins", np.int8, size),
                ("mabs", np.float32),
                ("energy", np.float32),
            ]
        )

        for ti in range(n_temps):
            t_bytes = f.read(4)
            if len(t_bytes) != 4:
                raise EOFError(f"Unexpected EOF while reading temperature header (ti={ti})")
            temps[ti] = struct.unpack("<f", t_bytes)[0]

            rec = np.fromfile(f, dtype=rec_dtype, count=n_samples)
            if rec.size != n_samples:
                raise EOFError(f"Unexpected EOF while reading records for T={temps[ti]}")
            configs[ti] = rec["spins"]
            mabs[ti] = rec["mabs"]
            energy[ti] = rec["energy"]

    order = np.argsort(temps)
    temps = temps[order]
    configs = configs[order]
    mabs = mabs[order]
    energy = energy[order]

    X = configs.astype(np.float32)
    if block is not None:
        block = int(block)
        if lattice_L % block != 0:
            raise ValueError(f"L={lattice_L} must be divisible by block={block}")
        Lb = lattice_L // block
        X = X.reshape(n_temps, n_samples, lattice_L, lattice_L)
        X = X.reshape(n_temps, n_samples, Lb, block, Lb, block).mean(axis=(3, 5))
        X = X.reshape(n_temps, n_samples, -1)

    obs = {
        "magnetization": mabs,
        "energy_density": energy,
        "_align_z2": np.array(bool(align_z2)),
        "_lattice_L": np.array(int(lattice_L)),
    }
    return Dataset(temps=temps.astype(float), X=X.astype(np.float32), obs=obs, configs=configs)


def flatten_dataset(ds: Dataset, *, use_raw_configs_if_present: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (temps_per_sample, X_per_sample)."""
    temps = np.repeat(np.asarray(ds.temps, dtype=float), ds.X.shape[1])
    if use_raw_configs_if_present and ds.configs is not None:
        X = ds.configs.reshape(ds.configs.shape[0] * ds.configs.shape[1], -1)
    else:
        X = ds.X.reshape(ds.X.shape[0] * ds.X.shape[1], -1)
    return temps, X


def choose_stratified_indices(temps: np.ndarray, per_temp: int, rng: np.random.Generator) -> np.ndarray:
    temps = np.asarray(temps, dtype=float)
    uniq = np.unique(temps)
    blocks: list[np.ndarray] = []
    per = int(per_temp)
    for t in uniq:
        w = np.where(np.isclose(temps, float(t)))[0]
        if w.size == 0:
            continue
        take = min(per, int(w.size))
        blocks.append(rng.choice(w, size=take, replace=False))
    if not blocks:
        return np.arange(temps.size)
    return np.concatenate(blocks)


class PairwiseDist2Matrix(Protocol):
    def __call__(self, X: np.ndarray) -> np.ndarray: ...


def ising_z2_quotient_l2_dist2_matrix(X: np.ndarray) -> np.ndarray:

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("Expected X with shape (n_samples, n_features)")
    n = int(X.shape[0])
    if n == 0:
        return np.empty((0, 0), dtype=np.float32)

    Xf = X.astype(np.float32, copy=False)
    gram = (Xf @ Xf.T).astype(np.float32, copy=False)
    norms = np.sum(Xf * Xf, axis=1, dtype=np.float32)
    d2 = norms[:, None] + norms[None, :] - 2.0 * np.abs(gram)
    np.fill_diagonal(d2, 0.0)
    return d2


def xy_o2_overlap_dist2_matrix(theta: np.ndarray) -> np.ndarray:
    """Distance for XY angles with global O(2) symmetry quotient.

    Implements
      d(θ^A, θ^B) = N - max( |Σ_j exp(i(θ^A_j - θ^B_j))|, |Σ_j exp(i(θ^A_j + θ^B_j))| ),
    and returns this as a nonnegative matrix suitable for use as a squared-distance scale in the kernel.
    """

    theta = np.asarray(theta, dtype=np.float32)
    if theta.ndim != 2:
        raise ValueError("Expected theta with shape (n_samples, n_sites)")
    n = int(theta.shape[0])
    if n == 0:
        return np.empty((0, 0), dtype=np.float32)

    spins = np.exp(1j * theta.astype(np.float32)).astype(np.complex64)
    a = spins @ spins.conj().T
    b = spins @ spins.T
    mags = np.maximum(np.abs(a), np.abs(b)).astype(np.float32)
    n_sites = float(theta.shape[1])
    d2 = (2.0 * (n_sites - mags)).astype(np.float32)
    np.fill_diagonal(d2, 0.0)
    return d2


def knn_from_dist2(d2: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    d2 = np.asarray(d2, dtype=np.float32)
    n = int(d2.shape[0])
    if n < 2:
        raise ValueError("Need at least 2 samples to build kNN graph")
    k = int(min(max(1, k), n - 1))
    idx = np.argpartition(d2, kth=k, axis=1)[:, : k + 1]
    knn = np.empty((n, k), dtype=np.int32)
    knn_d2 = np.empty((n, k), dtype=np.float32)
    for i in range(n):
        cand = idx[i]
        cand = cand[cand != i]
        cand = cand[:k]
        order = np.argsort(d2[i, cand])
        nn = cand[order]
        knn[i] = nn
        knn_d2[i] = d2[i, nn]
    return knn, knn_d2


def epsilon_median_heuristic(knn_d2: np.ndarray) -> float:
    vals = np.asarray(knn_d2, dtype=np.float32).ravel()
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return 1.0
    return float(np.median(vals))


def epsilon_median_allpairs(d2: np.ndarray) -> float:
    d2 = np.asarray(d2, dtype=np.float32)
    if d2.ndim != 2 or d2.shape[0] != d2.shape[1]:
        raise ValueError("Expected square distance matrix")
    if d2.shape[0] < 2:
        return 1.0
    tri = d2[np.triu_indices(d2.shape[0], k=1)]
    tri = tri[np.isfinite(tri) & (tri > 0)]
    if tri.size == 0:
        return 1.0
    return float(np.median(tri))

def diffusion_map_from_knn(
    knn_idx: np.ndarray,
    knn_d2: np.ndarray,
    *,
    epsilon: float,
    alpha: float = 1.0,
    n_eigs: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (evals, psi) where psi are right eigenvectors of the Markov matrix P."""

    epsilon = float(epsilon)
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError(f"epsilon must be finite and > 0 (got {epsilon})")

    knn_idx = np.asarray(knn_idx, dtype=np.int32)
    knn_d2 = np.asarray(knn_d2, dtype=np.float32)
    n = int(knn_idx.shape[0])
    if n == 0:
        return np.array([], dtype=np.float32), np.empty((0, 0), dtype=np.float32)

    W = np.zeros((n, n), dtype=np.float32)
    weights = np.exp(-knn_d2 / epsilon).astype(np.float32)
    for i in range(n):
        js = knn_idx[i]
        W[i, js] = weights[i]
    W = np.maximum(W, W.T)

    q = W.sum(axis=1)
    q = np.maximum(q, 1e-12)
    if alpha != 0.0:
        q_alpha = np.power(q, float(alpha))
        W = W / (q_alpha[:, None] * q_alpha[None, :])

    d = W.sum(axis=1)
    d = np.maximum(d, 1e-12)
    inv_sqrt_d = 1.0 / np.sqrt(d)
    S = (W * inv_sqrt_d[:, None]) * inv_sqrt_d[None, :]

    evals, evecs = np.linalg.eigh(S.astype(np.float64))
    order = np.argsort(evals)[::-1]
    evals = evals[order].astype(np.float32)
    evecs = evecs[:, order].astype(np.float32)

    n_eigs = int(min(max(1, n_eigs), n))
    evals = evals[:n_eigs]
    evecs = evecs[:, :n_eigs]

    psi = (evecs * inv_sqrt_d[:, None]).astype(np.float32)
    return evals, psi


def choose_k_by_eigengap(evals: np.ndarray, *, max_clusters: int = 8) -> int:
    """Choose cluster count K by the largest eigengap between λ_{K-1} and λ_K (excluding λ0=1)."""
    evals = np.asarray(evals, dtype=np.float32)
    if evals.size < 3:
        return 2
    max_clusters = int(min(max(2, max_clusters), evals.size - 1))
    gaps = evals[1:max_clusters] - evals[2 : max_clusters + 1]
    k = int(np.argmax(gaps) + 2)
    return int(max(2, k))


def kmeans_numpy(
    X: np.ndarray,
    k: int,
    rng: np.random.Generator,
    n_init: int = 10,
    max_iter: int = 200,
) -> tuple[np.ndarray, np.ndarray, float]:
    X = np.asarray(X, dtype=np.float32)
    n = int(X.shape[0])
    k = int(k)
    if k < 1 or k > n:
        raise ValueError("k must satisfy 1 <= k <= n")

    best_labels = None
    best_centers = None
    best_inertia = float("inf")

    for _ in range(int(n_init)):
        idx = rng.choice(n, size=k, replace=False)
        centers = X[idx].copy()

        labels = np.zeros(n, dtype=np.int32)
        for _it in range(int(max_iter)):
            d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = d2.argmin(axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                mask = labels == j
                if not np.any(mask):
                    centers[j] = X[rng.integers(0, n)]
                else:
                    centers[j] = X[mask].mean(axis=0)

        inertia = float(((X - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    assert best_labels is not None and best_centers is not None
    return best_labels, best_centers, float(best_inertia)


def tc_crossing(temps: np.ndarray, frac: np.ndarray, target: float = 0.5) -> float:
    temps = np.asarray(temps, dtype=float)
    frac = np.asarray(frac, dtype=float)
    order = np.argsort(temps)
    temps = temps[order]
    frac = frac[order]
    y = frac - float(target)
    for i in range(y.size - 1):
        if y[i] == 0.0:
            return float(temps[i])
        if y[i] * y[i + 1] < 0.0:
            t0, t1 = float(temps[i]), float(temps[i + 1])
            f0, f1 = float(frac[i]), float(frac[i + 1])
            if f1 == f0:
                return 0.5 * (t0 + t1)
            return t0 + (target - f0) * (t1 - t0) / (f1 - f0)
    return float(temps[int(np.argmin(np.abs(y)))])


def tc_max_slope(temps: np.ndarray, frac: np.ndarray) -> float:
    temps = np.asarray(temps, dtype=float)
    frac = np.asarray(frac, dtype=float)
    if temps.size < 3:
        return float("nan")
    slope = np.gradient(frac, temps)
    return float(temps[int(np.argmax(np.abs(slope)))])


@dataclasses.dataclass
class PipelineResult:
    epsilon: float
    evals: np.ndarray
    sample_temps: np.ndarray
    temps: np.ndarray
    tc: float
    embedding: np.ndarray


class UnsupervisedPhasePipeline:
    """Invariant diffusion-maps pipeline.

    The implementation matches the core steps in `XY_Diffusion_Map.ipynb`:
    1) compute a full pairwise distance matrix d^2 using the provided callable,
    2) set epsilon = median(d^2) (all pairs, excluding the diagonal),
    3) build a Gaussian kernel K_ij = exp(-d^2_ij / epsilon),
    4) row-normalize K to a Markov matrix (random walk),
    5) symmetrize for stable eigendecomposition and compute diffusion coordinates,
    6) delegate Tc extraction to `tc_eval`, which is the only model-specific step besides the distance.

    Only `dist2_matrix` and `tc_eval` are intended to change between Ising and XY usage.
    """
    def __init__(
        self,
        *,
        dist2_matrix: PairwiseDist2Matrix,
        tc_eval: Callable[[np.ndarray, np.ndarray], float],
    ) -> None:
        self.dist2_matrix = dist2_matrix
        self.tc_eval = tc_eval

    def fit(self, temps: np.ndarray, X: np.ndarray) -> PipelineResult:
        temps = np.asarray(temps, dtype=float)
        X = np.asarray(X)
        if temps.shape[0] != X.shape[0]:
            raise ValueError("temps and X must have the same number of rows")

        Ts = temps
        d2 = np.asarray(self.dist2_matrix(X), dtype=np.float32)
        epsilon = epsilon_median_allpairs(d2)
        epsilon = max(float(epsilon), 1e-12)
        kernel = np.exp(-d2 / epsilon).astype(np.float32)
        row_sums = kernel.sum(axis=1, keepdims=True).astype(np.float32)
        row_sums[row_sums == 0] = 1.0
        d_half = np.sqrt(row_sums)
        A = kernel / (d_half @ d_half.T)

        eigvals, eigvecs = np.linalg.eigh(A.astype(np.float64))
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order].astype(np.float32)
        eigvecs = eigvecs[:, order].astype(np.float32)

        n = int(min(5, eigvecs.shape[1] - 1))
        lambdas = eigvals[1 : 1 + n]
        Y = eigvecs[:, 1 : 1 + n] * lambdas
        tc = float(self.tc_eval(Ts, Y))

        return PipelineResult(
            epsilon=float(epsilon),
            evals=eigvals,
            sample_temps=Ts,
            temps=np.unique(Ts),
            tc=tc,
            embedding=Y,
        )


def tc_eval_u_max_slope(temps_per_sample: np.ndarray, embedding: np.ndarray) -> float:
    temps_per_sample = np.asarray(temps_per_sample, dtype=float)
    u = np.asarray(embedding[:, 0], dtype=float)
    temps_unique = np.unique(temps_per_sample)
    mean_u = np.array([u[np.isclose(temps_per_sample, float(t))].mean() for t in temps_unique], dtype=float)
    dmean = np.gradient(mean_u, temps_unique)
    return float(temps_unique[int(np.argmax(np.abs(dmean)))])


def tc_eval_ordered_fraction_crossing(
    temps_per_sample: np.ndarray,
    embedding: np.ndarray,
    *,
    seed: int = 0,
) -> float:
    temps_per_sample = np.asarray(temps_per_sample, dtype=float)
    temps_unique = np.unique(temps_per_sample)
    Y = np.asarray(embedding, dtype=np.float32)
    if Y.shape[1] >= 2:
        Z = Y[:, :2]
    else:
        Z = np.concatenate([Y[:, :1], np.zeros((Y.shape[0], 1), dtype=np.float32)], axis=1)

    rng = np.random.default_rng(int(seed))
    labels, _centers, _inertia = kmeans_numpy(Z, k=2, rng=rng, n_init=20, max_iter=300)
    lowT = float(np.min(temps_unique))
    mask_low = np.isclose(temps_per_sample, lowT)
    if np.any(mask_low):
        counts = np.bincount(labels[mask_low], minlength=2)
        ordered_cluster = int(np.argmax(counts))
    else:
        ordered_cluster = 0
    ordered = labels == ordered_cluster
    frac = np.array([float(np.mean(ordered[np.isclose(temps_per_sample, float(t))])) for t in temps_unique], dtype=float)
    return float(tc_crossing(temps_unique, frac, target=0.5))


def ordered_fraction_curve(
    temps_per_sample: np.ndarray,
    embedding: np.ndarray,
    *,
    seed: int = 0,
    dims: int = 2,
) -> tuple[np.ndarray, np.ndarray, float]:
    temps_per_sample = np.asarray(temps_per_sample, dtype=float)
    temps_unique = np.unique(temps_per_sample)
    Y = np.asarray(embedding, dtype=np.float32)
    d = int(max(1, dims))
    if Y.shape[1] >= d:
        Z = Y[:, :d]
    else:
        Z = np.concatenate([Y, np.zeros((Y.shape[0], d - Y.shape[1]), dtype=np.float32)], axis=1)

    rng = np.random.default_rng(int(seed))
    labels, _centers, _inertia = kmeans_numpy(Z, k=2, rng=rng, n_init=20, max_iter=300)
    lowT = float(np.min(temps_unique))
    mask_low = np.isclose(temps_per_sample, lowT)
    if np.any(mask_low):
        counts = np.bincount(labels[mask_low], minlength=2)
        ordered_cluster = int(np.argmax(counts))
    else:
        ordered_cluster = 0
    ordered = labels == ordered_cluster

    frac = np.array(
        [float(np.mean(ordered[np.isclose(temps_per_sample, float(t))])) for t in temps_unique],
        dtype=float,
    )
    tc = float(tc_crossing(temps_unique, frac, target=0.5))
    return temps_unique, frac, tc
