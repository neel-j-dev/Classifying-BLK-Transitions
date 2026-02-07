import numpy as np
from scipy.special import iv  # modified Bessel I_m
from __future__ import annotations
from typing import Dict, Tuple, Optional, List, Union
import scipy.sparse as sp
import scipy.sparse.linalg as spla


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
    Wm_ref = projected_kernel_so2(z_tilde, epsilon, m, include_const=include_const, S=S if include_const else None)

    return 0.5 * (Wm_rot + Wm_ref)


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
    skip_first: bool = True,
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

