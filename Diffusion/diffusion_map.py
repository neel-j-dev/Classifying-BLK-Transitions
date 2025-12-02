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


def compute_gaussian_kernel(X: np.ndarray, epsilon: float, N: int = 1024) -> np.ndarray:
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
    # Compute pairwise squared Euclidean distances
    pairwise_sq_dists = squareform(pdist(X, metric='sqeuclidean'))

    # Compute Gaussian kernel with specified normalization
    K = np.exp(-pairwise_sq_dists / (2.0 * N * epsilon))

    return K


def construct_transition_matrix(K: np.ndarray) -> np.ndarray:
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
    # Compute row sums of kernel matrix
    row_sums = K.sum(axis=1, keepdims=True)

    # Check for zero row sums
    if np.any(row_sums == 0):
        warnings.warn("Warning: Some rows of kernel matrix sum to zero.")
        row_sums[row_sums == 0] = 1.0

    # Normalize rows to create row-stochastic matrix
    P = K / row_sums

    return P


def compute_eigendecomposition(P: np.ndarray, n_components: int = 10, 
                                use_sparse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
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

    # Convert to real if eigenvalues are real (within numerical precision)
    if np.allclose(eigenvalues.imag, 0):
        eigenvalues = eigenvalues.real
    if np.allclose(eigenvectors.imag, 0):
        eigenvectors = eigenvectors.real

    return eigenvalues, eigenvectors


def construct_diffusion_map_embedding(eigenvalues: np.ndarray, eigenvectors: np.ndarray, 
                                       n_dimensions: int = 3, t: int = 1,
                                       skip_first: bool = True) -> np.ndarray:
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

    # Construct diffusion map embedding: phi_l = lambda_l^t * psi_l
    embedding = selected_eigenvectors * (selected_eigenvalues ** t)

    return embedding


def diffusion_map(X: np.ndarray, epsilon: float, n_components: int = 10, 
                  n_dimensions: int = 3, t: int = 1, N: int = 1024,
                  use_sparse: bool = False, skip_first: bool = True,
                  verbose: bool = True) -> Dict[str, np.ndarray]:
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

    K = compute_gaussian_kernel(X, epsilon=epsilon, N=N)
    P = construct_transition_matrix(K)
    eigenvalues, eigenvectors = compute_eigendecomposition(P, n_components=n_components, 
                                                           use_sparse=use_sparse)
    embedding = construct_diffusion_map_embedding(eigenvalues, eigenvectors, 
                                                   n_dimensions=n_dimensions, t=t,
                                                   skip_first=skip_first)

    if verbose:
        print(f"Diffusion map complete: embedding shape = {embedding.shape}")

    return {
        'embedding': embedding,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'kernel': K,
        'transition_matrix': P
    }
