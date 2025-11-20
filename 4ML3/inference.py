"""Probability-focused inference visualization for BLT GNN models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from data_utils import load_split
from metrics import estimate_critical_temperature


def smooth_probabilities(temperatures: np.ndarray, probs: np.ndarray, window: int = 72) -> np.ndarray:
    if window <= 1 or len(probs) < 3:
        return probs
    order = np.argsort(temperatures)
    probs_sorted = probs[order]
    kernel = np.ones(window, dtype=float) / window
    smoothed_sorted = np.convolve(probs_sorted, kernel, mode="same")
    smoothed = np.empty_like(probs_sorted)
    smoothed[:] = smoothed_sorted
    result = np.empty_like(probs)
    result[order] = smoothed
    return result


def plot_probability_scatter(ax, temps: np.ndarray, probs: np.ndarray, labels: np.ndarray, t_c: float, true_tc: float | None, method: str):
    scatter = ax.scatter(
        temps,
        probs,
        c=probs,
        cmap="plasma",
        s=45,
        edgecolor="black",
        linewidth=0.25,
        alpha=0.9,
        label="Nodes",
    )
    cb = plt.colorbar(scatter, ax=ax)
    cb.set_label("Predicted probability (phase=1)")

    predicted = probs >= 0.5
    misclassified = predicted != labels
    if np.any(misclassified):
        ax.scatter(
            temps[misclassified],
            probs[misclassified],
            facecolors="none",
            edgecolor="#222222",
            marker="x",
            linewidth=1.2,
            s=60,
            label="Misclassified",
        )

    ax.axvline(t_c, color="#222222", linestyle="--", label=f"Estimated T_c ({method})={t_c:.3f}")
    if true_tc is not None:
        ax.axvline(true_tc, color="#0b4f6c", linestyle="-.", label=f"Reference T_c={true_tc:.3f}")
    ax.axhline(0.5, color="#666666", linestyle=":")
    order = np.argsort(temps)
    temps_sorted = temps[order]
    probs_smoothed = smooth_probabilities(temps, probs)
    ax.plot(temps_sorted, probs_smoothed[order], color="#444444", linewidth=1.2, label="Moving average")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("P(phase = vortex)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=8)


def plot_probability_heatmap(ax, temps: np.ndarray, probs: np.ndarray, labels: np.ndarray):
    order = np.argsort(temps)
    ordered_temps = temps[order]
    ordered_probs = probs[order]
    ordered_labels = labels[order]
    heat_data = np.vstack([ordered_probs, ordered_labels])
    im = ax.imshow(
        heat_data,
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Value")
    ax.set_yticks([0, 1], ["Predicted probability", "True label"])
    ax.set_yticks([0.5, 1.5], minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    tick_positions = np.linspace(0, len(order) - 1, 6, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{ordered_temps[idx]:.2f}" for idx in tick_positions], rotation=45, ha="right")
    ax.set_xlabel("Temperature (sorted)")
    ax.set_title("Node-wise heatmap")

def plot_temperature_windows(axes, temps: np.ndarray, probs: np.ndarray, labels: np.ndarray, t_c: float):
    order = np.argsort(temps)
    temps_sorted = temps[order]
    probs_sorted = probs[order]
    labels_sorted = labels[order]
    splits = np.array_split(np.arange(len(temps_sorted)), len(axes))

    for ax, idxs in zip(axes, splits):
        if len(idxs) == 0:
            continue
        t_chunk = temps_sorted[idxs]
        p_chunk = probs_sorted[idxs]
        l_chunk = labels_sorted[idxs]
        scatter = ax.scatter(
            t_chunk,
            p_chunk,
            c=p_chunk,
            cmap="plasma",
            s=35,
            edgecolor="black",
            linewidth=0.2,
            alpha=0.9,
        )
        ax.axvline(t_c, color="#555555", linestyle="--", linewidth=1)
        ax.axhline(0.5, color="#777777", linestyle=":", linewidth=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(t_chunk.min() - 0.02, t_chunk.max() + 0.02)
        ax.set_title(f"T ∈ [{t_chunk.min():.2f}, {t_chunk.max():.2f}]")
        ax.set_xlabel("Temperature")
        if ax is axes[0]:
            ax.set_ylabel("P(phase=1)")
        predicted = p_chunk >= 0.5
        misclassified = predicted != l_chunk
        if np.any(misclassified):
            ax.scatter(
                t_chunk[misclassified],
                p_chunk[misclassified],
                facecolors="none",
                edgecolor="#222222",
                marker="x",
                linewidth=1.0,
                s=50,
            )


def load_true_tc(dataset_path: Path, cli_tc: float | None) -> float | None:
    if cli_tc is not None:
        return cli_tc
    metadata_path = dataset_path.parent / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open() as f:
            metadata = json.load(f)
        return metadata.get("critical_temp")
    return None


def main():
    parser = argparse.ArgumentParser(description="Visualize GNN phase probabilities without vortex detection.")
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/pyg_model.joblib"))
    parser.add_argument("--dataset", type=Path, default=Path("../XYModel/blt_dataset/train.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/inference_plot.png"))
    parser.add_argument("--true-tc", type=float, default=None, help="Optional reference critical temperature.")
    args = parser.parse_args()

    artifact = joblib.load(args.artifact)
    probs = np.asarray(artifact["train_probabilities"])
    temps = np.asarray(artifact["train_temperatures"])
    _, _, labels = load_split(args.dataset)
    if len(labels) != len(probs):
        print("Warning: dataset length mismatch. Using artifact ordering only.")
        labels = np.zeros_like(probs, dtype=int)

    true_tc = load_true_tc(args.dataset, args.true_tc)
    probs_smoothed = smooth_probabilities(temps, probs)
    t_c = estimate_critical_temperature(temps, probs_smoothed)
    print(f"Estimated critical temperature (smoothed crossing): {t_c:.3f}")
    if true_tc is not None:
        print(f"Reference critical temperature: {true_tc:.3f}")

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9])
    ax_scatter = fig.add_subplot(gs[0, :2])
    ax_heatmap = fig.add_subplot(gs[0, 2])
    window_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]

    plot_probability_scatter(ax_scatter, temps, probs, labels, t_c, true_tc, method="smoothed crossing")
    plot_probability_heatmap(ax_heatmap, temps, probs, labels)
    plot_temperature_windows(window_axes, temps, probs, labels, t_c)

    fig.suptitle("GNN inference: probability view", fontsize=14)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Saved visualization to {args.output}")

if __name__ == "__main__":
    main()
