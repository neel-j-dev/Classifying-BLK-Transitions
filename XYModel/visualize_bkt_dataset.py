"""Visualize Metropolis evolution and winding numbers using the BLT dataset utilities."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from . import xy as xy_module

# The dataset generator imports `xy` as a top-level module; register it so the import works
# when running this file as a module.
sys.modules.setdefault("xy", xy_module)
from . import generate_bkt_dataset as bkt

def _initialize_with_winding(lattice_shape: Tuple[int, int], nu_x: int, nu_y: int) -> np.ndarray:
    """Create a normalized lattice with a linear phase twist that sets the winding numbers."""

    h, w = lattice_shape
    x = np.arange(w)[None, :]
    y = np.arange(h)[:, None]
    phase = (nu_x * x / w) + (nu_y * y / h)
    return np.mod(phase, 1.0).astype(np.float64)

def _initialize_constant(lattice_shape: Tuple[int, int]) -> np.ndarray:
    """Create a normalized lattice with constant phase (zero winding)."""

    return np.zeros(lattice_shape, dtype=np.float64)


def _wrap_to_pi(delta: np.ndarray) -> np.ndarray:
    """Wrap angular differences to (-pi, pi]."""

    return (delta + np.pi) % (2 * np.pi) - np.pi


def compute_winding_numbers(theta: np.ndarray) -> Tuple[float, float]:
    """Compute (nu_x, nu_y) winding numbers from angular lattice `theta` in radians."""

    delta_x = _wrap_to_pi(np.diff(theta[0], append=theta[0, 0]))
    delta_y = _wrap_to_pi(np.diff(theta[:, 0], append=theta[0, 0]))
    return float(delta_x.sum() / (2 * np.pi)), float(delta_y.sum() / (2 * np.pi))


def compute_energy(theta: np.ndarray, coupling: float) -> float:
    """Compute the XY Hamiltonian for the given angular configuration."""

    term_x = np.cos(theta - np.roll(theta, -1, axis=1))
    term_y = np.cos(theta - np.roll(theta, -1, axis=0))
    return float(-coupling * (term_x.sum() + term_y.sum()))


def _angle_fields(sim: bkt.XYModelMetropolisSimulation) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return angular field (radians) and vector components (cos, sin) for plotting."""

    lattice_vectors = bkt.extract_lattice_vectors(sim).reshape(*sim.lattice_shape, 2)
    theta = np.mod(np.arctan2(lattice_vectors[..., 1], lattice_vectors[..., 0]), 2 * np.pi)
    return theta, lattice_vectors[..., 0], lattice_vectors[..., 1]


def run_local_sweeps(sim: bkt.XYModelMetropolisSimulation, sweeps: int, updates_per_sweep: int) -> None:
    """Advance only with local Metropolis updates (no global twist)."""

    for _ in range(sweeps):
        for _ in range(updates_per_sweep):
            sim.make_step()


def metropolis_evolution_panel(
    temperature: float,
    *,
    lattice_size: int = 32,
    coupling: float = 1.0,
    n_snapshots: int = 6,
    sweeps_per_snapshot: int = 200,
    burn_in_sweeps: int = 0,
    nu_x_init: int = 0,
    nu_y_init: int = 0,
    seed: int | None = None,
) -> plt.Figure:
    """Create a panel showing heatmaps and quivers over Metropolis evolution."""

    lattice_shape = (lattice_size, lattice_size)
    sim = bkt.XYModelMetropolisSimulation(
        lattice_shape=lattice_shape,
        beta=1.0 / temperature,
        J=coupling,
        random_state=seed,
        use_gpu=False,
    )

    # Seed the lattice with a prescribed winding, then let Metropolis evolve freely.
    sim.L[...] = _initialize_constant(lattice_shape)

    # Infer the initial winding from the initialized lattice.
    theta0, _, _ = _angle_fields(sim)
    initial_nu_x, initial_nu_y = compute_winding_numbers(theta0)
    sim.H = sim.compute_H()

    fig, axes = plt.subplots(4, n_snapshots, figsize=(4 * n_snapshots, 12))
    updates_per_sweep = lattice_shape[0] * lattice_shape[1]

    # Let the system thermalize before taking any snapshots; important for low T.
    if burn_in_sweeps > 0:
        run_local_sweeps(sim, burn_in_sweeps, updates_per_sweep)

    for snap_idx in range(n_snapshots):
        theta_before, U_before, V_before = _angle_fields(sim)
        energy_before = compute_energy(theta_before, coupling)
        nu_x_before, nu_y_before = compute_winding_numbers(theta_before)

        ax_top_before = axes[0, snap_idx]
        im_before = ax_top_before.imshow(theta_before, cmap="hsv", vmin=0, vmax=2 * np.pi, interpolation="nearest")
        sweep_num = burn_in_sweeps + snap_idx * sweeps_per_snapshot
        ax_top_before.set_title(
            f"Sweep {sweep_num} (before twist)\nE = {energy_before:.1f}\nν = ({nu_x_before:.2f}, {nu_y_before:.2f})",
            fontsize=10,
        )
        ax_top_before.set_xticks([])
        ax_top_before.set_yticks([])

        ax_quiver_before = axes[1, snap_idx]
        step = max(1, lattice_size // 16)
        x = np.arange(0, lattice_size, step)
        y = np.arange(0, lattice_size, step)
        X, Y = np.meshgrid(x, y)
        ax_quiver_before.quiver(
            X.flatten(),
            Y.flatten(),
            U_before[::step, ::step].flatten(),
            V_before[::step, ::step].flatten(),
            theta_before[::step, ::step].flatten(),
            cmap="hsv",
            clim=(0, 2 * np.pi),
            scale=25,
            width=0.008,
        )
        ax_quiver_before.set_xlim(-1, lattice_size)
        ax_quiver_before.set_ylim(-1, lattice_size)
        ax_quiver_before.set_aspect("equal")
        ax_quiver_before.set_xticks([])
        ax_quiver_before.set_yticks([])

        # Attempt one global twist move and visualize the result.
        sim.make_unwinding_step()

        theta_after, U_after, V_after = _angle_fields(sim)
        energy_after = compute_energy(theta_after, coupling)
        nu_x_after, nu_y_after = compute_winding_numbers(theta_after)

        ax_top_after = axes[2, snap_idx]
        im_after = ax_top_after.imshow(theta_after, cmap="hsv", vmin=0, vmax=2 * np.pi, interpolation="nearest")
        ax_top_after.set_title(
            f"After twist\nE = {energy_after:.1f}\nν = ({nu_x_after:.2f}, {nu_y_after:.2f})",
            fontsize=10,
        )
        ax_top_after.set_xticks([])
        ax_top_after.set_yticks([])

        ax_quiver_after = axes[3, snap_idx]
        ax_quiver_after.quiver(
            X.flatten(),
            Y.flatten(),
            U_after[::step, ::step].flatten(),
            V_after[::step, ::step].flatten(),
            theta_after[::step, ::step].flatten(),
            cmap="hsv",
            clim=(0, 2 * np.pi),
            scale=25,
            width=0.008,
        )
        ax_quiver_after.set_xlim(-1, lattice_size)
        ax_quiver_after.set_ylim(-1, lattice_size)
        ax_quiver_after.set_aspect("equal")
        ax_quiver_after.set_xticks([])
        ax_quiver_after.set_yticks([])

        # Advance locally to decorrelate before the next snapshot.
        run_local_sweeps(sim, sweeps_per_snapshot, updates_per_sweep)

    fig.colorbar(im_before, ax=axes[0, :], label="θ (rad)", shrink=0.7, pad=0.02)
    plt.suptitle(
        f"Metropolis Evolution: T/J = {temperature}, initial ν ≈ ({initial_nu_x:.2f}, {initial_nu_y:.2f})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize XY Metropolis evolution and windings using the BLT dataset utilities."
    )
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.3, 1.0])
    parser.add_argument("--lattice-size", type=int, default=32)
    parser.add_argument("--nu-x", type=int, default=1, dest="nu_x")
    parser.add_argument("--nu-y", type=int, default=0, dest="nu_y")
    parser.add_argument("--snapshots", type=int, default=6)
    parser.add_argument("--sweeps-per-snapshot", type=int, default=200)
    parser.add_argument(
        "--burn-in-sweeps",
        type=int,
        default=5000,
        help="Thermalization sweeps before the first snapshot (each sweep = lattice_size^2 updates).",
    )
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("XYModel"),
        help="Directory where images will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for temp in args.temperatures:
        fig = metropolis_evolution_panel(
            temperature=temp,
            lattice_size=args.lattice_size,
            coupling=args.coupling,
            n_snapshots=args.snapshots,
            sweeps_per_snapshot=args.sweeps_per_snapshot,
            burn_in_sweeps=args.burn_in_sweeps,
            nu_x_init=args.nu_x,
            nu_y_init=args.nu_y,
            seed=args.seed,
        )
        output_path = args.output_dir / f"metropolis_evolution_T{temp:.2f}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Saved: {output_path}")


if __name__ == "__main__":
    main()
