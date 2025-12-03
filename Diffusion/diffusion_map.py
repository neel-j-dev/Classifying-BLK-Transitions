"""
Diffusion Map Implementation for Topological Order Analysis

This module implements the diffusion map algorithm for dimensionality reduction
and clustering of spin configurations from the 2D XY model and Ising gauge theory.

Based on the paper: "Identifying topological order through unsupervised machine learning"
by Rodriguez-Nieva and Scheurer (arXiv:1805.05961v2)

Author: Generated for reproducibility study
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigs
from typing import Tuple, Optional, Dict
import warnings

try:  # Optional GPU backend via PyTorch
    import torch
except ImportError:  # pragma: no cover
    torch = None


def compute_gaussian_kernel(
    X: np.ndarray,
    epsilon: float,
    N: int = 1024,
    use_torch: bool = False,
    device: Optional[str] = None,
    return_torch: bool = False,
) -> np.ndarray:
    """
    Compute the Gaussian kernel matrix K from input samples.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input data matrix where each row is a sample (spin configuration)
    epsilon : float
        Bandwidth hyperparameter for the Gaussian kernel
    N : int, default=1024
        Number of spins (used for normalization). Default is 1024 for 32x32 lattice

    Returns
    -------
    K : np.ndarray, shape (n_samples, n_samples)
        Gaussian kernel matrix

    Notes
    -----
    The kernel is computed as: K_ij = exp(-||x_i - x_j||^2 / (2 * N * epsilon))
    Following the paper's specification for the XY model diffusion map analysis.
    """
    if use_torch:
        if torch is None:  # pragma: no cover
            raise ImportError("PyTorch is not installed; cannot use GPU backend.")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        X_t = torch.as_tensor(X, device=device, dtype=torch.float64)
        pairwise_sq_dists = torch.cdist(X_t, X_t, p=2) ** 2
        K_t = torch.exp(-pairwise_sq_dists / (2.0 * N * epsilon))
        if return_torch:
            return K_t
        return K_t.cpu().numpy()

    pairwise_sq_dists = squareform(pdist(X, metric='sqeuclidean'))
    K = np.exp(-pairwise_sq_dists / (2.0 * N * epsilon))
    return K


def construct_transition_matrix(K):
    """
    Construct the row-stochastic transition matrix P from the kernel matrix K.

    Parameters
    ----------
    K : np.ndarray, shape (n_samples, n_samples)
        Gaussian kernel matrix

    Returns
    -------
    P : np.ndarray, shape (n_samples, n_samples)
        Row-stochastic transition matrix (each row sums to 1)

    Notes
    -----
    The transition matrix is constructed by normalizing each row of K:
    P_ij = K_ij / sum_j(K_ij)
    """
    if torch is not None and isinstance(K, torch.Tensor):
        row_sums = K.sum(dim=1, keepdim=True)
        if torch.any(row_sums == 0):
            warnings.warn("Warning: Some rows of kernel matrix sum to zero.")
            row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        P = K / row_sums
        return P

    row_sums = K.sum(axis=1, keepdims=True)
    if np.any(row_sums == 0):
        warnings.warn("Warning: Some rows of kernel matrix sum to zero.")
        row_sums[row_sums == 0] = 1.0

    P = K / row_sums

    return P


def compute_eigendecomposition(P, n_components: int = 10,
                                use_sparse: bool = False,
                                use_torch: bool = False,
                                device: Optional[str] = None,
                                return_torch: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues and right eigenvectors of the transition matrix P.

    Parameters
    ----------
    P : np.ndarray, shape (n_samples, n_samples)
        Row-stochastic transition matrix
    n_components : int, default=10
        Number of leading eigenvectors to compute
    use_sparse : bool, default=False
        If True, use sparse eigendecomposition (efficient for large matrices)

    Returns
    -------
    eigenvalues : np.ndarray, shape (n_components,) or (n_samples,)
        Eigenvalues sorted in descending order by magnitude
    eigenvectors : np.ndarray, shape (n_samples, n_components) or (n_samples, n_samples)
        Right eigenvectors corresponding to the eigenvalues (column vectors)
    """
    n_samples = P.shape[0]

    if use_torch or (torch is not None and isinstance(P, torch.Tensor)):
        if torch is None:  # pragma: no cover
            raise ImportError("PyTorch is not installed; cannot use GPU backend.")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        P_t = P if isinstance(P, torch.Tensor) else torch.as_tensor(P, device=device, dtype=torch.float64)
        if use_sparse:
            raise ValueError("Sparse eigendecomposition is not supported with the torch backend.")

        eigenvalues, eigenvectors = torch.linalg.eig(P_t.T)
        idx = torch.argsort(torch.abs(eigenvalues), descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        if n_components < n_samples:
            eigenvalues = eigenvalues[:n_components]
            eigenvectors = eigenvectors[:, :n_components]

        if not return_torch:
            eigenvalues_np = eigenvalues.detach().cpu().numpy()
            eigenvectors_np = eigenvectors.detach().cpu().numpy()
            if np.allclose(eigenvalues_np.imag, 0):
                eigenvalues_np = eigenvalues_np.real
            if np.allclose(eigenvectors_np.imag, 0):
                eigenvectors_np = eigenvectors_np.real
            return eigenvalues_np, eigenvectors_np

        # For torch return type, keep complex dtype; caller can convert if desired.
        return eigenvalues, eigenvectors

    if use_sparse and n_components < n_samples:
        eigenvalues, eigenvectors = eigs(P.T, k=n_components, which='LM')
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    else:
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        if n_components < n_samples:
            eigenvalues = eigenvalues[:n_components]
            eigenvectors = eigenvectors[:, :n_components]

    if np.allclose(eigenvalues.imag, 0):
        eigenvalues = eigenvalues.real
    if np.allclose(eigenvectors.imag, 0):
        eigenvectors = eigenvectors.real

    return eigenvalues, eigenvectors


def construct_diffusion_map_embedding(eigenvalues, eigenvectors,
                                       n_dimensions: int = 3, t: int = 1,
                                       skip_first: bool = True):
    """
    Construct diffusion map coordinates (embedding) from eigenvectors and eigenvalues.

    Parameters
    ----------
    eigenvalues : np.ndarray, shape (n_components,)
        Eigenvalues sorted in descending order
    eigenvectors : np.ndarray, shape (n_samples, n_components)
        Right eigenvectors corresponding to eigenvalues (column vectors)
    n_dimensions : int, default=3
        Number of diffusion map dimensions to compute
    t : int, default=1
        Diffusion time parameter (power to raise eigenvalues)
    skip_first : bool, default=True
        If True, skip the first eigenvector (trivial eigenvector with eigenvalue=1)

    Returns
    -------
    embedding : np.ndarray, shape (n_samples, n_dimensions)
        Diffusion map embedding coordinates

    Notes
    -----
    The diffusion map coordinates are constructed as:
    phi_l(i) = lambda_l^t * psi_l(i)
    """
    start_idx = 1 if skip_first else 0
    end_idx = start_idx + n_dimensions

    if end_idx > eigenvectors.shape[1]:
        raise ValueError(f"Requested {n_dimensions} dimensions, but only "
                        f"{eigenvectors.shape[1] - start_idx} are available after skipping.")

    selected_eigenvalues = eigenvalues[start_idx:end_idx]
    selected_eigenvectors = eigenvectors[:, start_idx:end_idx]

    if torch is not None and (isinstance(selected_eigenvalues, torch.Tensor) or isinstance(selected_eigenvectors, torch.Tensor)):
        embedding = selected_eigenvectors * (selected_eigenvalues ** t)
        return embedding

    embedding = selected_eigenvectors * (selected_eigenvalues ** t)
    return embedding


def diffusion_map(X: np.ndarray, epsilon: float, n_components: int = 10, 
                  n_dimensions: int = 3, t: int = 1, N: int = 1024,
                  use_sparse: bool = False, skip_first: bool = True,
                  verbose: bool = True, use_torch: bool = False,
                  device: Optional[str] = None, return_torch: bool = False) -> Dict[str, np.ndarray]:
    """
    Complete diffusion map pipeline from data to embedding.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input data matrix where each row is a sample
    epsilon : float
        Bandwidth hyperparameter for the Gaussian kernel
    n_components : int, default=10
        Number of eigenvectors to compute
    n_dimensions : int, default=3
        Number of diffusion map dimensions for the final embedding
    t : int, default=1
        Diffusion time parameter
    N : int, default=1024
        Number of features for kernel normalization (1024 for 32x32 lattice)
    use_sparse : bool, default=False
        Whether to use sparse eigendecomposition
    skip_first : bool, default=True
        Whether to skip the first (trivial) eigenvector in the embedding
    verbose : bool, default=True
        Whether to print progress messages
    use_torch : bool, default=False
        If True and a CUDA-enabled PyTorch is available, compute kernel and
        eigendecomposition on GPU.
    device : str or None, default=None
        PyTorch device to use when `use_torch` is True. Defaults to "cuda" when
        available, otherwise "cpu".
    return_torch : bool, default=False
        If True and `use_torch` is enabled, return torch tensors instead of numpy arrays.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'embedding': Diffusion map embedding coordinates
        - 'eigenvalues': Computed eigenvalues
        - 'eigenvectors': Computed eigenvectors
        - 'kernel': Gaussian kernel matrix
        - 'transition_matrix': Row-stochastic transition matrix
    """
    if verbose:
        print("Computing diffusion map...")

    K = compute_gaussian_kernel(
        X,
        epsilon=epsilon,
        N=N,
        use_torch=use_torch,
        device=device,
        return_torch=return_torch,
    )
    P = construct_transition_matrix(K)
    eigenvalues, eigenvectors = compute_eigendecomposition(
        P,
        n_components=n_components,
        use_sparse=use_sparse,
        use_torch=use_torch or isinstance(P, (torch.Tensor,)) if torch is not None else use_torch,
        device=device,
        return_torch=return_torch,
    )
    embedding = construct_diffusion_map_embedding(
        eigenvalues,
        eigenvectors,
        n_dimensions=n_dimensions,
        t=t,
        skip_first=skip_first,
    )

    if not return_torch and torch is not None:
        # Ensure numpy outputs for compatibility with scikit-learn.
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        if isinstance(eigenvalues, torch.Tensor):
            eigenvalues = eigenvalues.detach().cpu().numpy()
        if isinstance(eigenvectors, torch.Tensor):
            eigenvectors = eigenvectors.detach().cpu().numpy()
        if isinstance(K, torch.Tensor):
            K = K.detach().cpu().numpy()
        if isinstance(P, torch.Tensor):
            P = P.detach().cpu().numpy()

    if verbose:
        print(f"Diffusion map complete: embedding shape = {embedding.shape}")

    return {
        'embedding': embedding,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'kernel': K,
        'transition_matrix': P
    }
