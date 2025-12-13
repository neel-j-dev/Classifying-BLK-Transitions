import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
matplotlib.use("Agg")

# ---------- Optional Numba, with safe fallback ----------

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # Fallback that behaves like numba.njit but does nothing
    def njit(func=None, **kwargs):
        # supports both @njit and @njit(...)
        if func is None:
            def wrapper(f):
                return f
            return wrapper
        return func


# ---------- LATTICE LAYER (same idea as Ising) ----------

def make_square_lattice(Lx, Ly=None):
    """
    Build a 2D square lattice with periodic boundary conditions.

    Returns:
        neighbors: int array of shape (N, 4) listing 4 nearest neighbours for each site
        N:         number of sites
        Lx, Ly:    lattice dimensions

    Sites are indexed in row-major: i = x + Lx * y
    """
    if Ly is None:
        Ly = Lx
    N = Lx * Ly
    neighbors = np.empty((N, 4), dtype=np.int64)

    for y in range(Ly):
        for x in range(Lx):
            i = x + Lx * y
            xp = (x + 1) % Lx
            xm = (x - 1) % Lx
            yp = (y + 1) % Ly
            ym = (y - 1) % Ly

            # order: right, left, up, down
            neighbors[i, 0] = xp + Lx * y     # right
            neighbors[i, 1] = xm + Lx * y     # left
            neighbors[i, 2] = x + Lx * yp     # up
            neighbors[i, 3] = x + Lx * ym     # down

    return neighbors, N, Lx, Ly


# ---------- XY MODEL DYNAMICS (local Metropolis) ----------

@njit
def xy_metropolis_sweep(theta, neighbors, beta, J, h, proposal_width):
    """
    One Metropolis sweep for the classical XY model.

    Args:
        theta:          1D float array of shape (N,), angles in [0, 2π)
        neighbors:      int64 array (N, z) of neighbor indices
        beta:           inverse temperature 1/T
        J:              coupling constant for -J cos(θ_i - θ_j)
        h:              external field along x: -h Σ cos θ_i
        proposal_width: max change in angle per proposal, in radians
                        (propose θ_i' = θ_i + δ, δ ∈ [-proposal_width, proposal_width])
    """
    N = theta.shape[0]
    z = neighbors.shape[1]
    two_pi = 2.0 * np.pi

    for i in range(N):
        theta_i = theta[i]

        # propose a new angle for site i
        delta = (np.random.rand() - 0.5) * 2.0 * proposal_width
        theta_new = theta_i + delta

        # wrap into [0, 2π)
        theta_new = theta_new % two_pi

        # --- compute local energy difference ---

        # old local energy from bonds attached to i and field term:
        # E_i_old = -J Σ_j cos(θ_i - θ_j) - h cos θ_i
        e_old = 0.0
        for k in range(z):
            j = neighbors[i, k]
            e_old -= J * np.cos(theta_i - theta[j])
        e_old -= h * np.cos(theta_i)

        # new local energy if we replace θ_i -> θ_new
        e_new = 0.0
        for k in range(z):
            j = neighbors[i, k]
            e_new -= J * np.cos(theta_new - theta[j])
        e_new -= h * np.cos(theta_new)

        dE = e_new - e_old

        # Metropolis accept/reject
        if dE <= 0.0 or np.random.rand() < np.exp(-beta * dE):
            theta[i] = theta_new


# ---------- XY MODEL DYNAMICS (Wolff cluster) ----------

@njit
def xy_wolff_step(theta, neighbors, beta, J):
    """
    One Wolff cluster update for the XY model (h = 0, ferromagnetic J > 0).

    Uses the embedded-cluster algorithm:
      - pick random direction r (angle phi),
      - define projections p_i = S_i · r = cos(theta_i - phi),
      - grow a cluster with bond prob
          p_add = 1 - exp(-2 beta J p_i p_j)  if p_i p_j > 0,
      - reflect spins in the cluster across the hyperplane perpendicular to r.

    Args:
        theta:     1D float array (N,) of angles in [0, 2π)
        neighbors: 2D int array (N, z) neighbor indices
        beta:      inverse temperature
        J:         XY coupling

    Returns:
        cluster_size: number of spins in the flipped cluster
    """
    N = theta.shape[0]
    z = neighbors.shape[1]
    two_pi = 2.0 * np.pi

    # random direction r
    phi = two_pi * np.random.rand()
    r_x = np.cos(phi)
    r_y = np.sin(phi)

    # projections p_i = S_i · r
    projs = np.empty(N, dtype=np.float64)
    for i in range(N):
        projs[i] = np.cos(theta[i] - phi)

    # pick random seed
    seed = int(np.random.rand() * N)

    in_cluster = np.zeros(N, dtype=np.uint8)
    stack = np.empty(N, dtype=np.int64)

    in_cluster[seed] = 1
    stack[0] = seed
    stack_size = 1
    cluster_size = 1

    # grow cluster
    while stack_size > 0:
        stack_size -= 1
        i = stack[stack_size]
        pi = projs[i]
        for k in range(z):
            j = neighbors[i, k]
            if in_cluster[j] == 0:
                pj = projs[j]
                prod = pi * pj
                if prod > 0.0:
                    p_add = 1.0 - np.exp(-2.0 * beta * J * prod)
                    if np.random.rand() < p_add:
                        in_cluster[j] = 1
                        stack[stack_size] = j
                        stack_size += 1
                        cluster_size += 1

    # reflect cluster across hyperplane perpendicular to r:
    # S' = S - 2 (S·r) r
    for i in range(N):
        if in_cluster[i] == 1:
            th = theta[i]
            s_x = np.cos(th)
            s_y = np.sin(th)
            dot = s_x * r_x + s_y * r_y
            s_xp = s_x - 2.0 * dot * r_x
            s_yp = s_y - 2.0 * dot * r_y
            theta[i] = np.arctan2(s_yp, s_xp) % two_pi

    return cluster_size



# ---------- XY CHAIN DRIVER (supports Metropolis or Wolff) ----------

def run_xy_chain(Lx, Ly, T, J=1.0, h=0.0,
                 n_therm=1000, n_sweeps=10000,
                 sample_interval=10,
                 proposal_width=np.pi / 2,
                 seed=1234,
                 update: str = "metropolis"):
    """
    Run a single-chain 2D XY simulation.

    Args:
        Lx, Ly:          lattice size
        T:               temperature
        J:               XY coupling
        h:               external field along x: -h Σ cos θ_i
                         (for Wolff updates, h MUST be 0)
        n_therm:         number of thermalization sweeps/steps
        n_sweeps:        number of production sweeps/steps
        sample_interval: store one configuration every 'sample_interval' production sweeps
        proposal_width:  Metropolis proposal width (ignored for Wolff)
        seed:            RNG seed
        update:          "metropolis" or "wolff"

    Returns:
        configs:        (n_samples, Ly, Lx) angles
        mags_trace:     (n_therm + n_sweeps,) |M| per step (full history)
        energies_trace: (n_therm + n_sweeps,) energy per site per step
    """
    if update not in ("metropolis", "wolff"):
        raise ValueError(f"Unknown update scheme '{update}', expected 'metropolis' or 'wolff'.")

    if update == "wolff" and h != 0.0:
        raise ValueError("Wolff cluster algorithm for XY is implemented only for h = 0.")

    np.random.seed(seed)
    neighbors, N, Lx, Ly = make_square_lattice(Lx, Ly)
    beta = 1.0 / T

    # random initial angles θ ∈ [0, 2π)
    theta = np.random.rand(N) * 2.0 * np.pi

    total_sweeps = n_therm + n_sweeps
    mags_trace = np.empty(total_sweeps, dtype=np.float64)
    energies_trace = np.empty(total_sweeps, dtype=np.float64)

    # number of samples in production region
    n_samples = n_sweeps // sample_interval
    configs = np.empty((n_samples, Ly, Lx), dtype=np.float64)
    sample_idx = 0

    for sweep in tqdm(range(total_sweeps), desc=f"XY ({update})"):
        if update == "metropolis":
            xy_metropolis_sweep(theta, neighbors, beta, J, h, proposal_width)
        else:  # "wolff"
            xy_wolff_step(theta, neighbors, beta, J)

        # compute observables THIS step
        E_per_site, M_abs = xy_observables(theta, neighbors, J, h)
        mags_trace[sweep] = M_abs
        energies_trace[sweep] = E_per_site

        # if we're in the production region, store configs every sample_interval
        if sweep >= n_therm:
            prod_step = sweep - n_therm  # 0-based index in production
            if (prod_step + 1) % sample_interval == 0:
                configs[sample_idx] = theta.reshape((Ly, Lx))
                sample_idx += 1

    return configs, mags_trace, energies_trace


def plot_thermalization(mags_trace, energies_trace=None, n_therm=0):
    """
    Plot |M| (and optionally energy) vs Monte Carlo sweep,
    with a vertical line at the end of thermalization.

    Args:
        mags_trace:     1D array of |M| for all sweeps
        energies_trace: 1D array of energies (same length) or None
        n_therm:        number of initial sweeps considered thermalization
    """
    sweeps = np.arange(len(mags_trace))

    if energies_trace is None:
        fig, ax = plt.subplots()
        ax.plot(sweeps, mags_trace, label='|M|')
        ax.axvline(n_therm, linestyle='--', label='end of thermalization')
        ax.set_xlabel('Sweep')
        ax.set_ylabel('|M|')
        ax.legend()
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

        ax1.plot(sweeps, mags_trace, label='|M|')
        ax1.axvline(n_therm, linestyle='--', label='end of thermalization')
        ax1.set_ylabel('|M|')
        ax1.legend()

        ax2.plot(sweeps, energies_trace, label='Energy per site')
        ax2.axvline(n_therm, linestyle='--', label='end of thermalization')
        ax2.set_xlabel('Sweep')
        ax2.set_ylabel('Energy')
        ax2.legend()

    plt.tight_layout()
    plt.savefig("xy_thermalization.png", dpi=200)
    print("Saved plot to xy_thermalization.png")


# ---------- AUTOCORRELATION ----------

def autocorrelation(series, max_lag=None):
    """
    Compute the normalized autocorrelation function C(tau)
    for a 1D time series using a simple O(T^2) estimator.

    Args:
        series:  1D array-like
        max_lag: maximum lag to compute (default T-1)

    Returns:
        lags: array [0, 1, ..., max_lag]
        C:    array of same length with C[0] = 1
    """
    x = np.asarray(series, dtype=np.float64)
    T = x.shape[0]

    if max_lag is None or max_lag >= T:
        max_lag = T - 1

    x_mean = x.mean()
    var = np.mean((x - x_mean) ** 2)
    if var == 0.0:
        # series is constant -> trivially fully correlated
        return np.arange(max_lag + 1), np.ones(max_lag + 1)

    C = np.empty(max_lag + 1, dtype=np.float64)
    C[0] = 1.0

    for tau in range(1, max_lag + 1):
        cov = 0.0
        for t in range(T - tau):
            cov += (x[t] - x_mean) * (x[t + tau] - x_mean)
        cov /= (T - tau)
        C[tau] = cov / var

    lags = np.arange(max_lag + 1)
    return lags, C


def estimate_autocorr_time(series, threshold=np.exp(-1), max_lag=None):
    """
    Estimate an "autocorrelation time" as the smallest lag tau
    such that C(tau) < threshold.

    Args:
        series:    1D time series (e.g. |M| history)
        threshold: value where we declare "correlation has died"
                   default = 1/e (≈ 0.367)
                   you might also use 0.1 or 0.01
        max_lag:   maximum lag to consider (default T-1)

    Returns:
        tau_hit: integer lag where C crosses below threshold.
                 If it never does within max_lag, returns max_lag.
        lags, C: the full autocorrelation function (for plotting / inspection).
    """
    lags, C = autocorrelation(series, max_lag=max_lag)
    for tau, c in zip(lags[1:], C[1:]):  # skip tau = 0
        if c < threshold:
            return tau, lags, C

    # did not cross the threshold
    return lags[-1], lags, C


def xy_observables(theta, neighbors, J, h):
    """
    Compute energy per site and |magnetization| for the XY model.

    Args:
        theta:     1D array of angles (radians)
        neighbors: (N, z) neighbor indices
        J:         coupling
        h:         external field along x: -h Σ cos θ_i

    Returns:
        E_per_site, |M|
    """
    N = theta.shape[0]
    z = neighbors.shape[1]

    # --- energy ---
    # E = -J sum_<ij> cos(θ_i - θ_j) - h sum_i cos θ_i
    # count each bond once: only include j > i
    E = 0.0
    for i in range(N):
        for k in range(z):
            j = neighbors[i, k]
            if j > i:
                E -= J * np.cos(theta[i] - theta[j])
    E -= h * np.cos(theta).sum()
    E_per_site = E / N

    # --- magnetization ---
    mx = np.cos(theta).mean()
    my = np.sin(theta).mean()
    M_abs = np.sqrt(mx * mx + my * my)

    return E_per_site, M_abs


def xy_snapshot_magnetization(snapshot: np.ndarray) -> float:
    """Return |M| using cos/sin averages for a 2D array of angles."""
    cos_vals = np.cos(snapshot)
    sin_vals = np.sin(snapshot)
    mx = cos_vals.mean()
    my = sin_vals.mean()
    return float(np.hypot(mx, my))


def xy_snapshot_energy_density(snapshot: np.ndarray, J: float, h: float) -> float:
    """Energy per site with periodic boundary conditions for a 2D configuration."""
    energy = 0.0
    # count bonds along +x and +y once each
    energy -= J * np.sum(np.cos(snapshot - np.roll(snapshot, shift=-1, axis=0)))
    energy -= J * np.sum(np.cos(snapshot - np.roll(snapshot, shift=-1, axis=1)))
    energy -= h * np.cos(snapshot).sum()
    return float(energy) / snapshot.size


def build_xy_dataset(
    temperatures: Sequence[float],
    samples_per_temp: int,
    lattice_shape: Tuple[int, int],
    burn_in_sweeps: int,
    sweeps_per_sample: int,
    J: float = 1.0,
    h: float = 0.0,
    proposal_width: float = np.pi / 2,
    seed: Optional[int] = None,
    show_progress: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    update: str = "metropolis",
) -> Tuple[np.ndarray, List[dict]]:
    """Generate flattened XY configurations plus metadata across temperatures."""

    temps = list(temperatures)
    if not temps:
        raise ValueError("`temperatures` must contain at least one entry.")
    if samples_per_temp <= 0 or sweeps_per_sample <= 0:
        raise ValueError("`samples_per_temp` and `sweeps_per_sample` must be positive.")

    if update not in ("metropolis", "wolff"):
        raise ValueError(f"Unknown update scheme '{update}', expected 'metropolis' or 'wolff'.")

    rng = np.random.default_rng(seed)
    flat_configs: List[np.ndarray] = []
    metadata: List[dict] = []
    total_tasks = len(temps)

    progress = tqdm(total=total_tasks, desc=f"XY dataset ({update})", disable=not show_progress)
    for temp_idx, temp in enumerate(temps):
        chain_seed = int(rng.integers(0, 1_000_000_000))
        n_production_sweeps = samples_per_temp * sweeps_per_sample
        configs, _, _ = run_xy_chain(
            Lx=lattice_shape[0],
            Ly=lattice_shape[1],
            T=temp,
            J=J,
            h=h,
            n_therm=burn_in_sweeps,
            n_sweeps=n_production_sweeps,
            sample_interval=sweeps_per_sample,
            proposal_width=proposal_width,
            seed=chain_seed,
            update=update,
        )

        if configs.shape[0] < samples_per_temp:
            raise RuntimeError(
                f"Requested {samples_per_temp} samples at T={temp} but only "
                f"received {configs.shape[0]} from the simulator."
            )

        for sample_idx in range(samples_per_temp):
            snapshot = np.asarray(configs[sample_idx], dtype=np.float32)
            flat_configs.append(snapshot.reshape(-1))
            metadata.append(
                {
                    "temperature": float(temp),
                    "beta": 1.0 / float(temp),
                    "sample_idx": sample_idx,
                    "magnetization": xy_snapshot_magnetization(snapshot),
                    "energy_density": xy_snapshot_energy_density(snapshot, J=J, h=h),
                    "lattice_shape": f"{lattice_shape[0]}x{lattice_shape[1]}",
                    "J": float(J),
                    "h": float(h),
                    "proposal_width": float(proposal_width),
                    "update": update,
                }
            )

        if progress is not None:
            progress.update(1)

    if progress is not None:
        progress.close()

    stacked = np.stack(flat_configs)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, configurations=stacked)
        metadata_path = save_path.parent / f"{save_path.name}.metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

    return stacked, metadata


# ---------- QUICK TEST ----------

if __name__ == "__main__":
    L = 32
    T = 0.88
    n_therm = 500
    n_sweeps = 100

    # switch update="wolff" to test clusters
    configs, mags_trace, energies_trace = run_xy_chain(
        Lx=L, Ly=L, T=T,
        J=1.0, h=0.0,
        n_therm=n_therm,
        n_sweeps=n_sweeps,
        sample_interval=100,
        proposal_width=np.pi,   # ignored for Wolff
        seed=42,
        update="wolff",
    )

    print("Configs shape:", configs.shape)

    # Plot observables and thermalization
    plot_thermalization(mags_trace, energies_trace, n_therm=n_therm)

    # Autocorrelation estimate on production part
    prod_mags = mags_trace[n_therm:]
    tau, lags, C = estimate_autocorr_time(prod_mags, threshold=0.1, max_lag=None)
    print("Estimated autocorrelation time in production region (C<0.1):", tau)

    # Build dataset with Wolff updates
    # build_xy_dataset(
    #     temperatures=np.arange(0.7, 0.9, 0.1),
    #     lattice_shape=(64, 64),
    #     burn_in_sweeps=20000,
    #     samples_per_temp=1,
    #     sweeps_per_sample=300,
    #     proposal_width=np.pi,
    #     save_path="xy_dataset_wolff",
    #     update="wolff",
    # )
