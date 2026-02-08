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

@njit
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
            i = x + Lx * y  # linear index for site (x, y)
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
def gxy_metropolis_sweep(theta, neighbors, beta, J, delta, proposal_width):
    """
    One Metropolis sweep for the generalized XY (gXY) model:

        H = -J sum_<ij> [ delta*cos(dtheta) + (1-delta)*cos(2*dtheta) ].

    Args:
        theta:          (N,) angles in [0, 2π)
        neighbors:      (N, z) neighbor indices
        beta:           1/T
        J:              overall coupling
        delta:          mixing parameter in [0,1]
        proposal_width: propose theta' = theta + u, u ~ Uniform[-proposal_width, proposal_width]
    """
    N = theta.shape[0]
    z = neighbors.shape[1]
    two_pi = 2.0 * np.pi

    for i in range(N):
        theta_i = theta[i]

        d = (np.random.rand() - 0.5) * 2.0 * proposal_width
        theta_new = (theta_i + d) % two_pi

        e_old = 0.0
        e_new = 0.0
        for k in range(z):
            j = neighbors[i, k]
            dth_old = theta_i - theta[j]
            dth_new = theta_new - theta[j]

            # bond energy contributions attached to i
            e_old -= J * (delta * np.cos(dth_old) + (1.0 - delta) * np.cos(2.0 * dth_old))
            e_new -= J * (delta * np.cos(dth_new) + (1.0 - delta) * np.cos(2.0 * dth_new))

        dE = e_new - e_old
        if dE <= 0.0 or np.random.rand() < np.exp(-beta * dE):
            theta[i] = theta_new

@njit
def gxy_overrelaxation_sweep(theta, neighbors, beta, J, delta, n_newton=3):
    """
    Overrelaxation-like sweep for gXY:
      - For delta == 1: exact XY microcanonical overrelaxation (always accepted).
      - For delta != 1: propose reflection about a local-energy minimizer found by a few Newton steps,
        then Metropolis accept/reject (keeps correct Boltzmann distribution).

    Args:
        theta:     (N,) angles in [0, 2π)
        neighbors: (N, z) neighbor indices
        beta:      1/T
        J:         coupling
        delta:     mixing parameter
        n_newton:  Newton iterations to approximate local minimizer

    Notes:
        This is not a cluster update; it’s a cheap local accelerator.
    """
    N = theta.shape[0]
    z = neighbors.shape[1]
    two_pi = 2.0 * np.pi
    eps_denom = 1e-12

    for i in range(N):
        th_old = theta[i]

        # --- compute local "fields" from neighbors ---
        H1x = 0.0
        H1y = 0.0
        H2x = 0.0
        H2y = 0.0
        for k in range(z):
            j = neighbors[i, k]
            thj = theta[j]
            c = np.cos(thj)
            s = np.sin(thj)
            H1x += c
            H1y += s
            c2 = np.cos(2.0 * thj)
            s2 = np.sin(2.0 * thj)
            H2x += c2
            H2y += s2

        # --- exact microcanonical overrelaxation for pure XY (delta=1) ---
        if np.abs(delta - 1.0) < 1e-15:
            # local field angle
            phi = np.arctan2(H1y, H1x)
            th_prop = (2.0 * phi - th_old) % two_pi
            theta[i] = th_prop
            continue

        # --- otherwise: find a local minimizer theta_star of the gXY local energy ---
        # local energy (up to constant): -J[ delta * Re(e^{i th} conj(H1)) + (1-delta) * Re(e^{i2th} conj(H2)) ]
        # Use a reasonable initial guess:
        if delta >= 0.5:
            th = np.arctan2(H1y, H1x)
        else:
            th = 0.5 * np.arctan2(H2y, H2x)

        th = th % two_pi

        # Newton iterations on derivative of local energy wrt th
        for _ in range(n_newton):
            c = np.cos(th)
            s = np.sin(th)
            c2 = np.cos(2.0 * th)
            s2 = np.sin(2.0 * th)

            # Re(e^{i th} conj(H1)) = c*H1x + s*H1y
            Re1 = c * H1x + s * H1y
            # Im(e^{i th} conj(H1)) = s*H1x - c*H1y
            Im1 = s * H1x - c * H1y

            # Re(e^{i 2th} conj(H2)) = c2*H2x + s2*H2y
            Re2 = c2 * H2x + s2 * H2y
            # Im(e^{i 2th} conj(H2)) = s2*H2x - c2*H2y
            Im2 = s2 * H2x - c2 * H2y

            # d/dth of local energy:
            # f'(th) = J[ delta*Im1 + 2(1-delta)*Im2 ]
            fprime = J * (delta * Im1 + 2.0 * (1.0 - delta) * Im2)

            # f''(th) = J[ delta*Re1 + 4(1-delta)*Re2 ]
            fsecond = J * (delta * Re1 + 4.0 * (1.0 - delta) * Re2)

            step = fprime / (fsecond + np.sign(fsecond) * eps_denom + eps_denom)
            th = (th - step) % two_pi

        theta_star = th

        # reflect old angle about theta_star (overrelaxation-style move)
        th_prop = (2.0 * theta_star - th_old) % two_pi

        # --- Metropolis accept/reject using local energy difference ---
        e_old = 0.0
        e_new = 0.0
        for k in range(z):
            j = neighbors[i, k]
            dth_old = th_old - theta[j]
            dth_new = th_prop - theta[j]
            e_old -= J * (delta * np.cos(dth_old) + (1.0 - delta) * np.cos(2.0 * dth_old))
            e_new -= J * (delta * np.cos(dth_new) + (1.0 - delta) * np.cos(2.0 * dth_new))

        dE = e_new - e_old
        if dE <= 0.0 or np.random.rand() < np.exp(-beta * dE):
            theta[i] = th_prop

from typing import Dict

def run_gxy_chain_auto_therm(
    Lx, Ly, T, J=1.0, delta=1.0,
    n_sweeps=100,
    sample_interval=100,
    proposal_width=np.pi / 2,
    seed=1234,
    update: str = "metropolis",
    n_overrelax: int = 1,
    initial_state: Optional[np.ndarray] = None,
    # --- adaptive therm params ---
    auto_therm: bool = True,
    max_therm: int = 1500,
    window: int = 200,
    check_every: int = 50,
    patience: int = 4,
    rel_tol_E: float = 5e-1,
    abs_tol_M: float = 2e-2,
    abs_tol_Q: float = 2e-2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    gXY simulation with optional adaptive thermalization.

    Returns:
        configs, mags_trace, nem_trace, energies_trace, info
    where info includes the chosen n_therm, etc.

    Notes:
      - At very low T, you may need large max_therm/window, and/or smaller proposal_width,
        and/or larger n_overrelax.
      - This heuristic detects stationarity of window-means, not true mixing. It's better than
        a fixed guess, but still not a proof of equilibration.
    """
    print("HAS_NUMBA: ", HAS_NUMBA)
    if update not in ("metropolis",):
        raise ValueError("This version currently supports only 'metropolis' for gXY.")

    np.random.seed(seed)
    beta = 1.0 / T
    neighbors, N, Lx, Ly = make_square_lattice(Lx, Ly)

    if initial_state is not None:
        theta = initial_state.flatten()
        if theta.shape[0] != Lx * Ly:
            raise ValueError("Initial state shape does not match lattice size.")
    else:
        theta = np.random.rand(N) * 2.0 * np.pi

    # ---- buffers for adaptive therm ----
    # We only need rolling windows of observables to decide thermalization.
    E_buf = np.empty(window, dtype=np.float64)
    M_buf = np.empty(window, dtype=np.float64)
    Q_buf = np.empty(window, dtype=np.float64)
    buf_fill = 0
    buf_pos = 0

    prev_mean_E = np.nan
    prev_mean_M = np.nan
    prev_mean_Q = np.nan
    stable_hits = 0

    mags_trace_list: List[float] = []
    nem_trace_list: List[float] = []
    energies_trace_list: List[float] = []

    sweep = 0
    n_therm_chosen = 0

    # --- adaptive thermalization phase ---
    if auto_therm:
        pbar = tqdm(total=max_therm, desc=f"gXY therm (auto)", leave=False)
        while sweep < max_therm:
            # dynamics
            gxy_metropolis_sweep(theta, neighbors, beta, J, delta, proposal_width)
            for _ in range(n_overrelax):
                gxy_overrelaxation_sweep(theta, neighbors, beta, J, delta, n_newton=3)

            E, M, Q = gxy_observables(theta, neighbors, J, delta)

            # store to rolling buffer
            E_buf[buf_pos] = E
            M_buf[buf_pos] = M
            Q_buf[buf_pos] = Q
            buf_pos = (buf_pos + 1) % window
            buf_fill = min(window, buf_fill + 1)

            # optionally store full trace (useful for plots)
            energies_trace_list.append(E)
            mags_trace_list.append(M)
            nem_trace_list.append(Q)

            sweep += 1
            pbar.update(1)

            # only check when buffer is full and at check cadence
            if buf_fill == window and (sweep % check_every == 0):
                mean_E = E_buf.mean()
                mean_M = M_buf.mean()
                mean_Q = Q_buf.mean()

                # change in window means
                dE = abs(mean_E - prev_mean_E)
                dM = abs(mean_M - prev_mean_M)
                dQ = abs(mean_Q - prev_mean_Q)

                # relative tol for E, absolute for order params (more stable)
                E_ok = (np.isnan(prev_mean_E) or dE <= rel_tol_E * max(1.0, abs(mean_E)))
                M_ok = (np.isnan(prev_mean_M) or dM <= abs_tol_M)
                Q_ok = (np.isnan(prev_mean_Q) or dQ <= abs_tol_Q)

                if E_ok and M_ok and Q_ok and not np.isnan(prev_mean_E):
                    stable_hits += 1
                else:
                    stable_hits = 0

                prev_mean_E, prev_mean_M, prev_mean_Q = mean_E, mean_M, mean_Q

                if stable_hits >= patience:
                    n_therm_chosen = sweep
                    break

        pbar.close()

        if n_therm_chosen == 0:
            # didn't stabilize within max_therm
            n_therm_chosen = sweep

    else:
        # fixed thermalization (compat mode)
        n_therm_chosen = 1000  # default-ish if auto_therm is False; override as you like

    # --- production phase ---
    total_prod = n_sweeps
    n_samples = total_prod // sample_interval
    configs = np.empty((n_samples, Ly, Lx), dtype=np.float64)
    sample_idx = 0

    pbar2 = tqdm(total=total_prod, desc=f"gXY prod", leave=True)
    for prod_sweep in range(total_prod):
        gxy_metropolis_sweep(theta, neighbors, beta, J, delta, proposal_width)
        for _ in range(n_overrelax):
            gxy_overrelaxation_sweep(theta, neighbors, beta, J, delta, n_newton=3)

        E, M, Q = gxy_observables(theta, neighbors, J, delta)

        energies_trace_list.append(E)
        mags_trace_list.append(M)
        nem_trace_list.append(Q)

        # store config every sample_interval
        if (prod_sweep + 1) % sample_interval == 0:
            configs[sample_idx] = theta.reshape((Ly, Lx))
            sample_idx += 1

        pbar2.update(1)
    pbar2.close()

    mags_trace = np.asarray(mags_trace_list, dtype=np.float64)
    nem_trace = np.asarray(nem_trace_list, dtype=np.float64)
    energies_trace = np.asarray(energies_trace_list, dtype=np.float64)

    info = {
        "n_therm": int(n_therm_chosen),
        "max_therm": int(max_therm),
        "window": int(window),
        "check_every": int(check_every),
        "patience": int(patience),
        "rel_tol_E": float(rel_tol_E),
        "abs_tol_M": float(abs_tol_M),
        "abs_tol_Q": float(abs_tol_Q),
    }
    return configs, mags_trace, nem_trace, energies_trace, info


def plot_thermalization(mags_trace, nem_trace, energies_trace=None, n_therm=0):
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
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 6))

        ax1.plot(sweeps, mags_trace, label='|M|')
        ax1.axvline(n_therm, linestyle='--', label='end of thermalization')
        ax1.set_ylabel('|M|')
        ax1.legend()

        ax2.plot(sweeps, energies_trace, label='Energy per site')
        ax2.axvline(n_therm, linestyle='--', label='end of thermalization')
        ax2.set_xlabel('Sweep')
        ax2.set_ylabel('Energy')
        ax2.legend()

        ax3.plot(sweeps, nem_trace, label='|Q|')
        ax3.axvline(n_therm, linestyle='--', label='end of thermalization')
        ax3.set_ylabel('|Q|')
        ax3.legend()

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

    x_mean = x.mean()  # subtract mean to get fluctuations
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


def gxy_observables(theta, neighbors, J, delta):
    """
    Return:
      E_per_site, |M|, |Q|
    where
      |M| = | <e^{i theta}> |
      |Q| = | <e^{i 2 theta}> |   (nematic order)
    """
    N = theta.shape[0]
    z = neighbors.shape[1]

    # Energy: count each bond once via j>i
    E = 0.0
    for i in range(N):
        for k in range(z):
            j = neighbors[i, k]
            if j > i:
                dth = theta[i] - theta[j]
                E -= J * (delta * np.cos(dth) + (1.0 - delta) * np.cos(2.0 * dth))
    E_per_site = E / N

    # Ferromagnetic order
    mx = np.cos(theta).mean()
    my = np.sin(theta).mean()
    M_abs = np.sqrt(mx * mx + my * my)

    # Nematic order
    qx = np.cos(2.0 * theta).mean()
    qy = np.sin(2.0 * theta).mean()
    Q_abs = np.sqrt(qx * qx + qy * qy)

    return E_per_site, M_abs, Q_abs



def gxy_snapshot_magnetization(snapshot: np.ndarray) -> float:
    mx = np.cos(snapshot).mean()
    my = np.sin(snapshot).mean()
    return float(np.hypot(mx, my))

def gxy_snapshot_nematic(snapshot: np.ndarray) -> float:
    qx = np.cos(2.0 * snapshot).mean()
    qy = np.sin(2.0 * snapshot).mean()
    return float(np.hypot(qx, qy))

def gxy_snapshot_energy_density(snapshot: np.ndarray, J: float, delta: float) -> float:
    # count bonds along +x and +y once each
    dth_x = snapshot - np.roll(snapshot, shift=-1, axis=0)
    dth_y = snapshot - np.roll(snapshot, shift=-1, axis=1)

    energy = 0.0
    energy -= J * np.sum(delta * np.cos(dth_x) + (1.0 - delta) * np.cos(2.0 * dth_x))
    energy -= J * np.sum(delta * np.cos(dth_y) + (1.0 - delta) * np.cos(2.0 * dth_y))
    return float(energy) / snapshot.size



def _serpentine_points(temperatures: Sequence[float], deltas: Sequence[float]) -> List[Tuple[float, float]]:
    """
    Return a path that walks the (T, delta) grid in a serpentine pattern
    to keep successive points close.
    """
    temps_sorted = sorted(set(float(t) for t in temperatures))
    deltas_sorted = sorted(set(float(d) for d in deltas))

    path: List[Tuple[float, float]] = []
    for idx, delta in enumerate(deltas_sorted):
        row = temps_sorted if idx % 2 == 0 else list(reversed(temps_sorted))
        for temp in row:
            path.append((temp, delta))
    return path


def _nearest_cached_state(
    T: float,
    delta: float,
    cache: dict,
    max_distance: Optional[float],
) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]], float]:
    """
    Pick the cached configuration whose parameters are closest in the
    T/delta plane. If max_distance is not None, ignore candidates farther
    than that threshold.
    """
    best_key: Optional[Tuple[float, float]] = None
    best_dist = float("inf")
    for key in cache.keys():
        dt = T - key[0]
        dd = delta - key[1]
        dist = float(np.hypot(dt, dd))
        if dist < best_dist:
            best_dist = dist
            best_key = key

    if best_key is None:
        return None, None, float("inf")

    if max_distance is not None and best_dist > max_distance:
        return None, best_key, best_dist

    return cache[best_key], best_key, best_dist


def build_xy_dataset(
    temperatures: Sequence[float],
    deltas: Sequence[float],
    samples_per_temp: int,
    lattice_shape: Tuple[int, int],
    burn_in_sweeps: int,
    sweeps_per_sample: int,
    J: float = 1.0,
    proposal_width: float = np.pi / 2,
    seed: Optional[int] = None,
    show_progress: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    update: str = "metropolis",
) -> Tuple[np.ndarray, List[dict]]:
    """Generate flattened XY configurations plus metadata across temperatures."""

    temps = list(temperatures)
    deltas = list(deltas)
    if not temps:
        raise ValueError("`temperatures` must contain at least one entry.")
    if not deltas:
        raise ValueError("`deltas` must contain at least one entry.")
    if samples_per_temp <= 0 or sweeps_per_sample <= 0:
        raise ValueError("`samples_per_temp` and `sweeps_per_sample` must be positive.")

    if update not in ("metropolis"):
        raise ValueError(f"Unknown update scheme '{update}', expected 'metropolis'.")

    rng = np.random.default_rng(seed)
    flat_configs: List[np.ndarray] = []
    metadata: List[dict] = []
    total_tasks = len(temps) * len(deltas)

    # track loop over temperatures
    progress = tqdm(total=total_tasks, desc=f"XY dataset ({update})", disable=not show_progress)
    for temp_idx, temp in enumerate(temps):
        for delta_idx, delta in enumerate(deltas): 
            chain_seed = int(rng.integers(0, 1_000_000_000))
            n_production_sweeps = samples_per_temp * sweeps_per_sample
            configs, _, _, _, _ = run_gxy_chain_auto_therm(
                Lx=lattice_shape[0],
                Ly=lattice_shape[1],
                T=temp,
                J=J,
                delta = delta,
                n_sweeps=n_production_sweeps,
                sample_interval=sweeps_per_sample,
                proposal_width=proposal_width,
                seed=chain_seed,
                update=update,
                max_therm=burn_in_sweeps,
                initial_state=flat_configs[-1].reshape(lattice_shape) if flat_configs else None,
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
                        "magnetization": gxy_snapshot_magnetization(snapshot),
                        "nematic": gxy_snapshot_nematic(snapshot),
                        "energy_density": gxy_snapshot_energy_density(snapshot, J=J, delta=delta),
                        "lattice_shape": f"{lattice_shape[0]}x{lattice_shape[1]}",
                        "J": float(J),
                        "delta": float(delta),
                        "proposal_width": float(proposal_width),
                        "update": update,
                    }
                )


            if progress is not None:
                progress.update(1)  # one temperature completed

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


def build_gxy_plane_dataset(
    temperatures: Sequence[float],
    deltas: Sequence[float],
    samples_per_point: int,
    lattice_shape: Tuple[int, int],
    sweeps_per_sample: int,
    *,
    J: float = 1.0,
    proposal_width: float = np.pi / 2,
    n_overrelax: int = 1,
    warm_start_max_distance: Optional[float] = 0.25,
    serpentine: bool = True,
    update: str = "metropolis",
    seed: Optional[int] = None,
    show_progress: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    max_therm: int = 1500,
    window: int = 200,
    check_every: int = 50,
    patience: int = 4,
    rel_tol_E: float = 5e-1,
    abs_tol_M: float = 2e-2,
    abs_tol_Q: float = 2e-2,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Build a dataset over a (T, delta) plane, traversing nearby points and
    reusing the last configuration from the closest simulated point as a
    warm start.
    """
    temps = list(temperatures)
    dels = list(deltas)
    if not temps:
        raise ValueError("`temperatures` must contain at least one entry.")
    if not dels:
        raise ValueError("`deltas` must contain at least one entry.")
    if samples_per_point <= 0 or sweeps_per_sample <= 0:
        raise ValueError("`samples_per_point` and `sweeps_per_sample` must be positive.")
    if update != "metropolis":
        raise ValueError("Only the local Metropolis update is supported here.")

    path = _serpentine_points(temps, dels) if serpentine else [
        (float(t), float(d)) for d in sorted(set(dels)) for t in sorted(set(temps))
    ]

    rng = np.random.default_rng(seed)
    flat_configs: List[np.ndarray] = []
    metadata: List[dict] = []
    cache: dict = {}
    last_state: Optional[np.ndarray] = None
    last_key: Optional[Tuple[float, float]] = None

    total_tasks = len(path)
    progress = tqdm(total=total_tasks, desc="gXY plane dataset", disable=not show_progress)

    for point_idx, (temp, delta) in enumerate(path):
        warm_state, warm_key, warm_dist = _nearest_cached_state(
            temp, delta, cache=cache, max_distance=warm_start_max_distance
        )
        if warm_state is None and last_state is not None and last_key is not None:
            warm_state = last_state
            warm_key = last_key
            warm_dist = float(np.hypot(temp - last_key[0], delta - last_key[1]))

        chain_seed = int(rng.integers(0, 1_000_000_000))
        n_production_sweeps = samples_per_point * sweeps_per_sample

        configs, mags_trace, nem_trace, energies_trace, info = run_gxy_chain_auto_therm(
            Lx=lattice_shape[0],
            Ly=lattice_shape[1],
            T=temp,
            J=J,
            delta=delta,
            n_sweeps=n_production_sweeps,
            sample_interval=sweeps_per_sample,
            proposal_width=proposal_width,
            seed=chain_seed,
            update=update,
            n_overrelax=n_overrelax,
            initial_state=warm_state,
            max_therm=max_therm,
            window=window,
            check_every=check_every,
            patience=patience,
            rel_tol_E=rel_tol_E,
            abs_tol_M=abs_tol_M,
            abs_tol_Q=abs_tol_Q,
        )

        final_state = np.asarray(configs[-1], dtype=np.float64)
        cache[(temp, delta)] = final_state
        last_state = final_state
        last_key = (temp, delta)

        warm_source_T = warm_key[0] if warm_key is not None else None
        warm_source_delta = warm_key[1] if warm_key is not None else None
        warm_dist_clean = None if not np.isfinite(warm_dist) else float(warm_dist)

        for sample_idx in range(samples_per_point):
            snapshot = np.asarray(configs[sample_idx], dtype=np.float32)
            flat_configs.append(snapshot.reshape(-1))
            metadata.append(
                {
                    "temperature": float(temp),
                    "beta": 1.0 / float(temp),
                    "sample_idx": sample_idx,
                    "magnetization": gxy_snapshot_magnetization(snapshot),
                    "nematic": gxy_snapshot_nematic(snapshot),
                    "energy_density": gxy_snapshot_energy_density(snapshot, J=J, delta=delta),
                    "lattice_shape": f"{lattice_shape[0]}x{lattice_shape[1]}",
                    "J": float(J),
                    "delta": float(delta),
                    "proposal_width": float(proposal_width),
                    "update": update,
                    "n_overrelax": int(n_overrelax),
                    "n_therm": int(info.get("n_therm", 0)),
                    "warm_start_source_T": warm_source_T,
                    "warm_start_source_delta": warm_source_delta,
                    "warm_start_distance": warm_dist_clean,
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

def analyze_gxy_run(
    mags_trace: np.ndarray,
    nem_trace: np.ndarray,
    energies_trace: np.ndarray,
    T: float,
    n_therm: int,
    N: int,
) -> dict:
    """
    Same spirit as analyze_xy_run, but for gXY with BOTH ferromagnetic |M|
    and nematic |Q| order parameters.

    Args:
        mags_trace:     |M| per step (length = n_therm + n_sweeps)
        nem_trace:      |Q| per step (length = n_therm + n_sweeps)
        energies_trace: energy per site per step
        T:              temperature
        n_therm:        number of initial steps to discard
        N:              number of sites

    Returns:
        dict with keys:
            "T", "beta", "E_mean", "E_var",
            "|M|_mean", "|M|_var", "|Q|_mean", "|Q|_var",
            "C" (specific heat per site),
            "chi_M" (susceptibility-like from |M| fluctuations),
            "chi_Q" (susceptibility-like from |Q| fluctuations).
    """
    beta = 1.0 / T
    prod_E = np.asarray(energies_trace[n_therm:], dtype=np.float64)
    prod_M = np.asarray(mags_trace[n_therm:], dtype=np.float64)
    prod_Q = np.asarray(nem_trace[n_therm:], dtype=np.float64)

    E_mean = prod_E.mean()
    E2_mean = np.mean(prod_E**2)
    M_mean = prod_M.mean()
    M2_mean = np.mean(prod_M**2)
    Q_mean = prod_Q.mean()
    Q2_mean = np.mean(prod_Q**2)

    E_var = E2_mean - E_mean**2
    M_var = M2_mean - M_mean**2
    Q_var = Q2_mean - Q_mean**2

    # specific heat per site
    C = beta**2 * E_var * N

    # susceptibilities (rough; using magnitudes)
    chi_M = beta * N * M_var
    chi_Q = beta * N * Q_var

    return {
        "T": float(T),
        "beta": float(beta),
        "E_mean": float(E_mean),
        "E_var": float(E_var),
        "|M|_mean": float(M_mean),
        "|M|_var": float(M_var),
        "|Q|_mean": float(Q_mean),
        "|Q|_var": float(Q_var),
        "C": float(C),
        "chi_M": float(chi_M),
        "chi_Q": float(chi_Q),
    }


def scan_gxy_temperatures(
    temperatures: Sequence[float],
    L: int,
    n_therm: int,
    n_sweeps: int,
    J: float = 1.0,
    delta: float = 1.0,
    sample_interval: int = 10,
    proposal_width: float = np.pi / 2,
    update: str = "metropolis",
    seed: Optional[int] = None,
    save_prefix: str = "gxy_Tscan",
) -> List[dict]:
    """
    Scan over temperatures for the generalized XY (gXY) model:

        H = -J sum_<ij> [ delta*cos(dtheta) + (1-delta)*cos(2*dtheta) ].

    Uses run_gxy_chain (Metropolis always OK; Wolff only valid for delta=1).

    Args:
        temperatures:   list/array of T values to scan
        L:              linear lattice size (Lx = Ly = L)
        n_therm:        thermalization steps
        n_sweeps:       production steps per temperature
        J:              overall coupling
        delta:          mixing parameter in [0,1]
        sample_interval: passed to run_gxy_chain (for snapshot saving)
        proposal_width: used only for Metropolis
        update:         "metropolis" or "wolff" (wolff requires delta == 1)
        seed:           base RNG seed
        save_prefix:    prefix for output figure filenames

    Returns:
        results: list of dicts (one per T) as from analyze_gxy_run
    """
    temps = np.array(list(temperatures), dtype=float)
    if temps.size == 0:
        raise ValueError("Need at least one temperature.")

    if update not in ("metropolis", "wolff"):
        raise ValueError(f"Unknown update scheme '{update}', expected 'metropolis' or 'wolff'.")

    if update == "wolff" and abs(delta - 1.0) > 1e-12:
        raise ValueError("Wolff update here is only valid for delta = 1 (pure XY).")

    rng = np.random.default_rng(seed)
    neighbors, N, _, _ = make_square_lattice(L, L)  # just to know N

    results: List[dict] = []

    for T in tqdm(temps, desc=f"gXY T-scan ({update}, delta={delta:g})"):
        chain_seed = int(rng.integers(0, 1_000_000_000))
        configs, mags_trace, nem_trace, energies_trace = run_gxy_chain(
            Lx=L,
            Ly=L,
            T=float(T),
            J=J,
            delta=delta,
            n_therm=n_therm,
            n_sweeps=n_sweeps,
            sample_interval=sample_interval,
            proposal_width=proposal_width,
            seed=chain_seed,
            update=update,
        )

        res = analyze_gxy_run(
            mags_trace=mags_trace,
            nem_trace=nem_trace,
            energies_trace=energies_trace,
            T=float(T),
            n_therm=n_therm,
            N=N,
        )
        # record parameters
        res["J"] = float(J)
        res["delta"] = float(delta)
        res["update"] = update
        results.append(res)

    # Convert to arrays for plotting
    Ts = np.array([r["T"] for r in results])
    E_mean = np.array([r["E_mean"] for r in results])
    M_mean = np.array([r["|M|_mean"] for r in results])
    Q_mean = np.array([r["|Q|_mean"] for r in results])
    C_vals = np.array([r["C"] for r in results])
    chi_M = np.array([r["chi_M"] for r in results])
    chi_Q = np.array([r["chi_Q"] for r in results])

    # --- Plot thermodynamics vs T ---
    fig, axs = plt.subplots(2, 3, figsize=(11, 6))

    ax = axs[0, 0]
    ax.plot(Ts, E_mean, "o-")
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\langle E \rangle$")

    ax = axs[0, 1]
    ax.plot(Ts, M_mean, "o-")
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\langle |M| \rangle$")

    ax = axs[0, 2]
    ax.plot(Ts, Q_mean, "o-")
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\langle |Q| \rangle$")

    ax = axs[1, 0]
    ax.plot(Ts, C_vals, "o-")
    ax.set_xlabel("T")
    ax.set_ylabel("C (per site)")

    ax = axs[1, 1]
    ax.plot(Ts, chi_M, "o-")
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\chi_M$ (rough)")

    ax = axs[1, 2]
    ax.plot(Ts, chi_Q, "o-")
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\chi_Q$ (rough)")

    fig.suptitle(f"gXY model, L={L}, J={J:g}, delta={delta:g}, update={update}")
    plt.tight_layout()
    fig.savefig(f"{save_prefix}_L{L}_J{J:g}_delta{delta:g}_{update}.png", dpi=200)
    print(f"Saved T-scan plot to {save_prefix}_L{L}_J{J:g}_delta{delta:g}_{update}.png")

    # crude indicators: peaks of specific heat / susceptibilities
    idx_C = int(np.argmax(C_vals))
    idx_chiM = int(np.argmax(chi_M))
    idx_chiQ = int(np.argmax(chi_Q))

    print(f"Peak C at T ≈ {Ts[idx_C]:.3f}")
    print(f"Peak chi_M at T ≈ {Ts[idx_chiM]:.3f}")
    print(f"Peak chi_Q at T ≈ {Ts[idx_chiQ]:.3f}")

    return results




# ---------- QUICK TEST ----------

if __name__ == "__main__":

    # L = 64
    # n_therm = 1000
    # n_sweeps = 20

    # temps = np.linspace(0.3, 1.0, 12)  # coarse scan around BKT

    # # Use Wolff to get good statistics near criticality
    # results = scan_xy_temperatures(
    #     temperatures=temps,
    #     L=L,
    #     n_therm=n_therm,
    #     n_sweeps=n_sweeps,
    #     J=1.0,
    #     h=0.0,
    #     sample_interval=30,
    #     proposal_width=np.pi,   # ignored for Wolff
    #     update="wolff",
    #     seed=123,
    #     save_prefix="xy_Tscan",
    # )

    # L = 128
    # T = 0.01
    # delta = 0.15
    # n_sweeps=100

    # switch update="wolff" to test clusters
    # configs, mags_trace, nem_trace, energies_trace, _ = run_gxy_chain_auto_therm(
    #     Lx=L, Ly=L, T=T, delta=delta,
    #     J=1.0,
    #     n_sweeps=n_sweeps,
    #     sample_interval=100,
    #     proposal_width=0.05,  
    #     seed=42,
    #     update="metropolis",
    #     max_therm=900
    # )

    # for i in tqdm(range(1, 200)): 
    #     configs, mags_trace, energies_trace = run_xy_chain(
    #         Lx=L, Ly=L, T=T+i*0.01,
    #         J=1.0, h=0.0,
    #         n_therm=1000,
    #         n_sweeps=n_sweeps,
    #         sample_interval=100,
    #         proposal_width=np.pi,   # ignored for Wolff
    #         seed=42,
    #         update="wolff",
    #         initial_state=configs[-1],  # warm start from previous run
    #     )

    # configs, mags_trace, energies_trace = run_xy_chain(
    #         Lx=L, Ly=L, T=2.01,
    #         J=1.0, h=0.0,
    #         n_therm=6000,
    #         n_sweeps=n_sweeps,
    #         sample_interval=100,
    #         proposal_width=np.pi,   # ignored for Wolff
    #         seed=42,
    #         update="wolff",
    #         initial_state=configs[-1],  # warm start from previous run
    #     )

    # print("Configs shape:", configs.shape)

    # Plot observables and thermalization
    # plot_thermalization(mags_trace, nem_trace, energies_trace, n_therm=1000)

    # Autocorrelation estimate on production part
    # prod_mags = mags_trace[n_therm:]
    # tau, lags, C = estimate_autocorr_time(prod_mags, threshold=0.1, max_lag=None)
    # print("Estimated autocorrelation time in production region (C<0.1):", tau)


    configs, meta = build_gxy_plane_dataset(
        temperatures=np.linspace(0.1, 1.0, 20),
        deltas=np.linspace(0.0, 1.0, 20),
        samples_per_point=1,
        lattice_shape=(16, 16),
        sweeps_per_sample=100,
        warm_start_max_distance=0.1,  # None to always reuse nearest
        proposal_width=0.3,
        n_overrelax=2,
        max_therm=1200,
        save_path="gxy_dataset"
    )
