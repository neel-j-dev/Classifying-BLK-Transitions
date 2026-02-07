import numpy as np
from scipy.special import iv  # modified Bessel I_m


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
