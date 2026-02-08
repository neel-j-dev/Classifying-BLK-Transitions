from __future__ import annotations
import numpy as np
from scipy.special import iv  # modified Bessel I_m
from typing import Dict, Tuple, Optional, List, Union
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


ArrayLike = Union[np.ndarray, "sp.spmatrix"]


def angles_to_spins(theta: np.ndarray) -> np.ndarray:
    """
    theta: (N, ...), angles in radians
    returns spins s = exp(i theta) as complex array of shape (N, S) with S=prod(...)
    """
    theta = np.asarray(theta)
    if theta.ndim < 2:
        raise ValueError("theta must have shape (N, ...) with N>=1 and at least one lattice dimension.")
    N = theta.shape[0]
    spins = np.exp(1j * theta.reshape(N, -1))
    return spins


def compute_overlaps_o2(spins: np.ndarray):
    """
    spins: (N, S) complex, with |spins|=1 typically.
    returns:
      z      = spins @ conj(spins).T  (rotation channel)
      z_tilde= spins @ spins.T        (reflection channel)
    """
    spins = np.asarray(spins)
    if spins.ndim != 2:
        raise ValueError("spins must be (N, S).")
    z = spins @ spins.conj().T
    z_tilde = spins @ spins.T
    return z, z_tilde


def lifted_kernel_so2_from_overlap(z: np.ndarray, epsilon: float, alpha: float, include_const: bool = False, S: int = None):
    """
    Compute W_ij(alpha) = exp( -||s_i - e^{i alpha} s_j||^2 / epsilon )
    using only z_ij = sum_a s_i(a) conj(s_j(a)).

    z: (N,N) complex overlap
    epsilon: kernel bandwidth (>0)
    alpha: rotation angle in radians
    include_const: if True, includes exp(-2S/epsilon) factor
                   (usually unnecessary because it cancels in normalization)
    S: number of lattice sites (required if include_const=True)

    Returns: W_alpha (N,N) real, nonnegative
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")

    # ||s_i - e^{i alpha} s_j||^2 = 2S - 2 Re(e^{-i alpha} z_ij)
    # so W = exp(-(2S - 2 Re(...))/eps) = exp(-2S/eps) * exp( (2/eps) Re(e^{-i alpha} z_ij) )
    phase = np.exp(-1j * alpha)
    kappa_term = (2.0 / epsilon) * np.real(phase * z)
    W = np.exp(kappa_term)

    if include_const:
        if S is None:
            raise ValueError("Provide S (number of lattice sites) if include_const=True.")
        W *= np.exp(-(2.0 * S) / epsilon)

    return W


def projected_kernel_so2(z: np.ndarray, epsilon: float, m: int, include_const: bool = False, S: int = None):
    """
    Compute the SO(2) Fourier/irrep-projected kernel:
      W_hat^{(m)}_{ij} ∝ I_m( 2|z_ij|/epsilon ) * exp(-i m arg(z_ij))

    z: (N,N) complex overlap (rotation channel)
    epsilon: bandwidth
    m: integer mode >=0 recommended (negative m is redundant: W_-m = conj(W_m))
    include_const: if True includes exp(-2S/epsilon)
    S: sites if include_const=True

    Returns: (N,N) complex (for m>0), real nonnegative (for m=0)
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if m < 0:
        raise ValueError("Use m>=0. (m<0 is redundant: W_-m = conj(W_m)).")

    absz = np.abs(z)
    argz = np.angle(z)

    kappa = (2.0 / epsilon) * absz
    Wm = iv(m, kappa) * np.exp(-1j * m * argz)

    if include_const:
        if S is None:
            raise ValueError("Provide S (number of lattice sites) if include_const=True.")
        Wm *= np.exp(-(2.0 * S) / epsilon)

    return Wm


def projected_kernel_o2(theta: np.ndarray, epsilon: float, m: int, include_const: bool = False):
    """
    Convenience wrapper: given angle configs theta (N, ...),
    compute the O(2) projected kernel by averaging rotation and reflection channels:

      W_hat^{(m)}_O2 = 0.5 * W_hat^{(m)}(z) + 0.5 * W_hat^{(m)}(z_tilde)

    where:
      z       = sum s_i conj(s_j)
      z_tilde = sum s_i s_j   (reflection channel)

    Returns: (N,N) complex (m>0) or real (m=0)
    """
    spins = angles_to_spins(theta)
    N, S = spins.shape
    z, z_tilde = compute_overlaps_o2(spins)

    Wm_rot = projected_kernel_so2(z, epsilon, m, include_const=include_const, S=S if include_const else None)
    # Wm_ref = projected_kernel_so2(z_tilde, epsilon, m, include_const=include_const, S=S if include_const else None)
    # return 0.5 * (Wm_rot + Wm_ref)
    return Wm_rot  # reflection channel is often very small and noisy, so omit for now


def degree_from_W0(W0: np.ndarray) -> np.ndarray:
    """Degree/row-sum for the m=0 (real, nonnegative) kernel."""
    d = np.asarray(W0).sum(axis=1)
    # avoid zeros
    return np.maximum(d, 1e-300)


def symmetric_normalize(Wm: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Symmetric normalization: S_m = D^{-1/2} W_m D^{-1/2}
    where d = row-sum of W0 (typically).
    """
    inv_sqrt = 1.0 / np.sqrt(d)
    return (inv_sqrt[:, None] * Wm) * inv_sqrt[None, :]


def _is_hermitian(A: np.ndarray, tol: float = 1e-10) -> bool:
    return np.linalg.norm(A - A.conj().T) / max(1.0, np.linalg.norm(A)) < tol


def eig_topk(
    A: ArrayLike,
    k: int,
    *,
    assume_hermitian: bool = True,
    which: str = "LA",
    tol: float = 1e-8,
    maxiter: int = 2000,
    dense_fallback: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute top-k eigenpairs of A.

    Returns:
        evals: (k,) eigenvalues sorted descending by real part
        evecs: (N,k) eigenvectors (columns)

    Notes:
      - If A is Hermitian, we use eigh/eigsh.
      - If A is not Hermitian, we use eig/eigs (less stable).
    """
    if k <= 0:
        raise ValueError("k must be positive")

    # Dense path
    if isinstance(A, np.ndarray):
        if assume_hermitian and _is_hermitian(A):
            evals, evecs = np.linalg.eigh(A)
            idx = np.argsort(evals.real)[::-1][:k]
            return evals[idx], evecs[:, idx]
        else:
            evals, evecs = np.linalg.eig(A)
            idx = np.argsort(evals.real)[::-1][:k]
            return evals[idx], evecs[:, idx]

    # Sparse path
    if sp is None or spla is None:
        if not dense_fallback:
            raise RuntimeError("scipy is required for sparse eigendecomposition.")
        A = A.toarray()
        return eig_topk(A, k, assume_hermitian=assume_hermitian)

    if not sp.isspmatrix(A):
        raise ValueError("Sparse path expects a scipy sparse matrix.")

    N = A.shape[0]
    if k >= N:
        # fall back to dense
        if not dense_fallback:
            raise ValueError("k must be < N for sparse eigensolvers.")
        return eig_topk(A.toarray(), k, assume_hermitian=assume_hermitian)

    # Hermitian solver if possible
    if assume_hermitian:
        # eigsh expects real-symmetric or complex-Hermitian
        evals, evecs = spla.eigsh(A, k=k, which=which, tol=tol, maxiter=maxiter)
        idx = np.argsort(evals.real)[::-1]
        return evals[idx], evecs[:, idx]

    # General non-Hermitian
    evals, evecs = spla.eigs(A, k=k, which=which, tol=tol, maxiter=maxiter)
    idx = np.argsort(evals.real)[::-1]
    return evals[idx], evecs[:, idx]


def flatten_hermitian_gram_to_real(G: np.ndarray) -> np.ndarray:
    """
    Flatten a batch of Hermitian Gram matrices into real feature vectors
    while preserving Frobenius norms.

    Input:
        G: (N,K,K) complex Hermitian-ish (numerical noise ok)

    Output:
        feats: (N, K + 2 * K*(K-1)/2) real
               = diag(G) + sqrt(2)*Re(upper offdiag) + sqrt(2)*Im(upper offdiag)
    """
    N, K, K2 = G.shape
    assert K == K2

    iu = np.triu_indices(K)
    di = iu[0] == iu[1]
    off = ~di

    tri = G[:, iu[0], iu[1]]  # (N, K*(K+1)/2) complex
    diag = tri[:, di].real    # (N, K)
    off_vals = tri[:, off]    # (N, K*(K-1)/2)

    # Frobenius-preserving real embedding for Hermitian matrices:
    # ||G||_F^2 = sum diag^2 + 2*sum |off|^2
    # so represent off parts as sqrt(2)*Re + sqrt(2)*Im
    s2 = np.sqrt(2.0)
    feats = np.concatenate([diag, s2 * off_vals.real, s2 * off_vals.imag], axis=1)
    return feats.astype(np.float64, copy=False)


def invariant_embedding_from_mode_eigs(
    evals: np.ndarray,
    evecs: np.ndarray,
    *,
    t: int = 1,
    diagonal_only: bool = False,
) -> np.ndarray:
    """
    Build an invariant embedding from one mode's eigenpairs.

    Given top-K eigenpairs (after skipping the trivial one, typically),
    define for each sample i:
      a_i[k] = (evals[k]**t) * evecs[i,k]

    Full (paper-faithful) invariant features: Gram matrix G_i = a_i a_i^*
    -> flatten to real features preserving Frobenius distances.

    If diagonal_only=True, uses only |a_i[k]|^2 (cheaper, less faithful).
    """
    if t < 0:
        raise ValueError("t must be >= 0")
    K = evals.shape[0]

    # diffusion-time weights; keep complex safely
    wt = (evals.astype(np.complex128) ** t)  # (K,)
    A = evecs.astype(np.complex128) * wt[None, :]  # (N,K)

    if diagonal_only:
        return (np.abs(A) ** 2).astype(np.float64)

    # G[i,:,:] = A[i,:] outer conj(A[i,:])
    G = np.einsum("ik,il->ikl", A, np.conjugate(A))  # (N,K,K)
    return flatten_hermitian_gram_to_real(G)


def build_invariant_embedding(
    S_by_m: Dict[int, ArrayLike],
    *,
    K: int = 30,
    t: int = 1,
    skip_first: bool = False,
    assume_hermitian: bool = True,
    diagonal_only: bool = False,
) -> Tuple[np.ndarray, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """
    Diagonalize each mode matrix S_m and return a concatenated invariant embedding.

    Args:
        S_by_m: dict mapping mode m -> S_m (NxN), dense or sparse
        K:      number of nontrivial eigenvectors per mode to use in embedding
        t:      diffusion time (weights eigenvectors by eval^t)
        skip_first: skip the top eigenpair (usually the stationary/trivial one)
        assume_hermitian: try Hermitian solvers; checks dense Hermitian automatically
        diagonal_only: if True, use only |a_k(i)|^2 instead of full Gram (faster, less faithful)

    Returns:
        Psi: (N, D) real invariant embedding
        eigs: dict m -> (evals_used, evecs_used)
    """
    if K <= 0:
        raise ValueError("K must be positive")

    eigs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    blocks: List[np.ndarray] = []

    for m, Sm in sorted(S_by_m.items(), key=lambda kv: kv[0]):
        # compute K + 1 if skipping first
        k_need = K + (1 if skip_first else 0)
        evals, evecs = eig_topk(Sm, k_need, assume_hermitian=assume_hermitian)

        if skip_first:
            evals_use = evals[1:K+1]
            evecs_use = evecs[:, 1:K+1]
        else:
            evals_use = evals[:K]
            evecs_use = evecs[:, :K]

        eigs[m] = (evals_use, evecs_use)

        Psi_m = invariant_embedding_from_mode_eigs(
            evals_use, evecs_use, t=t, diagonal_only=diagonal_only
        )
        blocks.append(Psi_m)

    Psi = np.concatenate(blocks, axis=1) if blocks else np.empty((0, 0), dtype=np.float64)
    return Psi, eigs


def phase_map_from_invariant_embedding(
    Psi: np.ndarray,
    temperatures: np.ndarray,
    deltas: np.ndarray,
    *,
    n_clusters: int = 3,
    reducer: str = "pca",          # "pca" or None
    n_components: int = 30,
    clusterer: str = "gmm",        # "gmm" or "kmeans"
    random_state: int = 0,
    round_decimals: int = 8,       # helps float-key stability
    make_plot: bool = True,
):
    """
    Build a phase map by clustering per-(T,delta) centroids of Psi.

    Returns:
        phase_map: (nT, nD) int array of cluster labels (or -1 where missing)
        T_unique: sorted unique temperatures (length nT)
        D_unique: sorted unique deltas (length nD)
        labels_param: (n_params,) labels for each unique (T,delta)
        centroids: (n_params, dim) centroid Psi for each unique (T,delta)
        uniq_pairs: (n_params,2) array of unique (T,delta)
    """
    Psi = np.asarray(Psi, dtype=np.float64)
    temperatures = np.asarray(temperatures, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)

    if Psi.shape[0] != temperatures.shape[0] or Psi.shape[0] != deltas.shape[0]:
        raise ValueError("Psi, temperatures, deltas must have the same length (N samples).")

    # Stabilize float keys (important if temps/deltas were computed rather than literal constants)
    Tq = np.round(temperatures, round_decimals)
    Dq = np.round(deltas, round_decimals)

    pairs = np.stack([Tq, Dq], axis=1)
    uniq_pairs, inv = np.unique(pairs, axis=0, return_inverse=True)  # inv maps sample -> param_index
    n_params = uniq_pairs.shape[0]

    # Centroids (and some optional diagnostics)
    centroids = np.zeros((n_params, Psi.shape[1]), dtype=np.float64)
    counts = np.bincount(inv)

    for k in range(n_params):
        centroids[k] = Psi[inv == k].mean(axis=0)

    # Standardize
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(centroids)

    # Optional dimensionality reduction (highly recommended for K=30 Gram features)
    if reducer == "pca":
        n_components_eff = min(n_components, X.shape[1], n_params - 1) if n_params > 1 else 1
        Xr = PCA(n_components=n_components_eff, random_state=random_state).fit_transform(X)
    elif reducer is None:
        Xr = X
    else:
        raise ValueError("reducer must be 'pca' or None")

    # Cluster
    if clusterer == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
        labels_param = model.fit_predict(Xr)
    elif clusterer == "gmm":
        model = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=random_state)
        labels_param = model.fit_predict(Xr)
    else:
        raise ValueError("clusterer must be 'kmeans' or 'gmm'")

    # Quick checks
    print(f"Unique (T,delta) points: {n_params}")
    print("Cluster sizes:", np.bincount(labels_param, minlength=n_clusters))

    if n_params > n_clusters:
        try:
            sil = silhouette_score(Xr, labels_param)
            print(f"Silhouette score (on centroids): {sil:.3f}")
        except Exception as e:
            print("Silhouette score unavailable:", e)

    # Build grid map
    T_unique = np.unique(uniq_pairs[:, 0])
    D_unique = np.unique(uniq_pairs[:, 1])
    T_unique.sort()
    D_unique.sort()

    phase_map = -np.ones((T_unique.size, D_unique.size), dtype=int)

    # Fill phase map
    for k in range(n_params):
        T, D = uniq_pairs[k]
        iT = np.searchsorted(T_unique, T)
        iD = np.searchsorted(D_unique, D)
        phase_map[iT, iD] = int(labels_param[k])

    # Optional plot
    if make_plot:
        plt.figure(figsize=(7, 5))
        im = plt.imshow(
            phase_map,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[D_unique.min(), D_unique.max(), T_unique.min(), T_unique.max()],
        )
        plt.xlabel(r"$\delta$")
        plt.ylabel(r"$T$")
        plt.title(f"Phase map from invariant embedding (clusters={n_clusters}, {clusterer}, {reducer})")
        plt.colorbar(im, label="cluster label")
        plt.tight_layout()
        plt.savefig("phase_map.png", dpi=300)

    return phase_map, T_unique, D_unique, labels_param, centroids, uniq_pairs



if __name__ == "__main__":
    # Load XY dataset and metadata produced by the simulator
    from pathlib import Path
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    # Path to the compressed dataset produced by build_xy_dataset
    DATA_PATH = Path('gxy_dataset.npz')
    METADATA_PATH = Path('gxy_dataset.metadata.json')

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset {DATA_PATH} not found. Update DATA_PATH to point to your file.")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file {METADATA_PATH} not found.")

    with np.load(DATA_PATH) as npz:
        configurations = npz['configurations']

    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    num_samples, num_sites = configurations.shape
    print(f"Loaded {num_samples} samples with {num_sites} lattice sites each.")

    W_0 = projected_kernel_o2(configurations, 100, 0)
    W_1 = projected_kernel_o2(configurations, 100, 1)
    W_2 = projected_kernel_o2(configurations, 100, 2)
    D = degree_from_W0(W_0)
    S_0 = symmetric_normalize(W_0, D)
    S_1 = symmetric_normalize(W_1, D)
    S_2 = symmetric_normalize(W_2, D)

    S_m = {0: S_0, 1: S_1, 2: S_2}
    Psi, eigs = build_invariant_embedding(S_m, K=3, t=1, skip_first=False, assume_hermitian=True, diagonal_only=False)

    temperatures = np.array([entry['temperature'] for entry in metadata], dtype=np.float64)
    deltas = np.array([entry['delta'] for entry in metadata], dtype=np.float64)

    phase_map, T_grid, D_grid, labels_param, centroids, uniq_pairs = phase_map_from_invariant_embedding(
    Psi,
    temperatures,
    deltas,
    n_clusters=3,
    reducer=None,
    n_components=30,
    clusterer="gmm",     # try "kmeans" too
    random_state=0,
    make_plot=True,
    )



# # Psi = np.real_if_close(Psi, tol=1e-7)  # convert to real if numerically close

# print("Eigenvalues by mode:")
# for m, (evals, evecs) in eigs.items():
#     eigs[m] = (np.real_if_close(evals, tol=1e-5), evecs)  # convert evals to real if close
#     print(f"Mode {m}: {eigs[m][0][:5]}")  # Show first 5 eigenvalues

# print("Invariant embedding shape:", Psi.shape)
    
    
    
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn.mixture import GaussianMixture

# # 1) (Optional but recommended) check Hermitian-ness if you assume it:
# print("S1 Hermitian?", np.allclose(S_1, S_1.conj().T, atol=1e-10))
# print("S2 Hermitian?", np.allclose(S_2, S_2.conj().T, atol=1e-10))

# # 2) Standardize features (important for k-means / GMM)
# X = StandardScaler(with_mean=True, with_std=True).fit_transform(Psi)

# # 3) (Optional) reduce dimension a bit for stability/speed
# #    If Psi is big (K=30 -> per mode ~900 dims), PCA helps a lot.
# # Xr = PCA(n_components=30, random_state=0).fit_transform(X)
# Xr = X

# # 4a) k-means into 3 phases
# labels_km = KMeans(n_clusters=3, n_init="auto", random_state=0).fit_predict(Xr)

# # 4b) or Gaussian mixture (often better if clusters are ellipsoids / unequal sizes)
# gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=0)
# labels_gmm = gmm.fit_predict(Xr)

# print("kmeans counts:", np.bincount(labels_km))
# print("gmm counts:", np.bincount(labels_gmm))

# temperatures = np.array([entry['temperature'] for entry in metadata], dtype=np.float64)
# deltas = np.array([entry['delta'] for entry in metadata], dtype=np.float64)

    
    

