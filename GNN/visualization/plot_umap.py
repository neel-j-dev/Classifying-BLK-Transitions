"""UMAP visualization of PyG lattice graph embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader

import umap

from GNN.utils.data_utils import load_split
from GNN.utils.lattice_utils import build_lattice_edge_index, infer_lattice_shape, load_metadata, make_lattice_graphs
from GNN.models.models import build_pyg_lattice_model
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.svm import SVC


def main():
    parser = argparse.ArgumentParser(description="UMAP visualization of PyG lattice graph embeddings.")
    parser.add_argument("--artifact", type=Path, default=Path("GNN/artifacts/pyg_lattice_model.joblib"))
    parser.add_argument("--dataset", type=Path, default=Path("XYModel/blt_dataset/train.npz"))
    parser.add_argument("--output", type=Path, default=Path("GNN/artifacts/umap_embeddings.png"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-neighbors", type=int, default=50)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", type=str, default="euclidean")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = joblib.load(args.artifact)
    model_cfg = artifact.get("model_config", {"input_dim": 2, "hidden_dim": 64, "num_layers": 2, "dropout": 0.1, "heads": 2})
    model_type = artifact.get("model_type", "gcn")

    features, temps, labels = load_split(args.dataset)
    metadata = load_metadata(args.dataset.parent)
    lattice_shape = tuple(artifact.get("lattice_shape", infer_lattice_shape(features.shape[1], metadata)))
    periodic = bool(artifact.get("periodic", False))
    edge_index = artifact.get("edge_index")
    if edge_index is None:
        edge_index = build_lattice_edge_index(lattice_shape, periodic=periodic)
    else:
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)

    graphs = make_lattice_graphs(features, temps, lattice_shape, edge_index, labels=labels)
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)

    if model_type == "attn_temp":
        # Use the attention regressor to embed
        model_type = "attn_temp"
    model = build_pyg_lattice_model(
        model_type=model_type,
        input_dim=model_cfg.get("input_dim", 2),
        hidden_dim=model_cfg.get("hidden_dim", 64),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.1),
        heads=model_cfg.get("heads", 2),
    ).to(device)

    state_dict = artifact.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Artifact is missing model_state_dict needed for embeddings.")
    model.load_state_dict(state_dict)
    model.eval()

    embeddings = []
    temps_out = []
    labels_out = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if hasattr(model, "embed"):
                z = model.embed(data)
            else:
                logits = model(data)
                z = logits
            embeddings.append(z.cpu())
            temps_out.append(data.temp.view(-1).cpu())
            if hasattr(data, "y"):
                labels_out.append(data.y.view(-1).cpu())

    embeddings_np = torch.cat(embeddings).numpy()
    temps_np = torch.cat(temps_out).numpy()
    labels_np = torch.cat(labels_out).numpy() if labels_out else np.zeros(len(embeddings_np), dtype=int)

    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=0,
    )
    proj = reducer.fit_transform(embeddings_np)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        proj[:, 0],
        proj[:, 1],
        c=temps_np if model_type == "attn_temp" else labels_np,
        cmap="viridis" if model_type == "attn_temp" else "plasma",
        s=30,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.2,
    )
    cb = plt.colorbar(scatter, ax=ax)
    cb.set_label("Temperature" if model_type == "attn_temp" else "Phase label")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"UMAP embeddings ({model_type})")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Saved UMAP plot to {args.output}")

    # --- Silhouette Analysis for Clustering ---

    print("Running silhouette analysis to determine optimal clusters...")
    range_n_clusters = list(range(2, 11))
    best_n_clusters = 2
    best_score = -1
    best_labels = None

    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = clusterer.fit_transform(proj)
        # We use fit_predict or labels_ from fit. Using fit_predict here.
        cluster_labels = clusterer.fit_predict(proj)
        
        silhouette_avg = silhouette_score(proj, cluster_labels)
        print(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg:.4f}")
        
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_n_clusters = n_clusters
            best_labels = cluster_labels

    print(f"Best number of clusters: {best_n_clusters} with score {best_score:.4f}")

    # Plot clusters
    fig_clust, ax_clust = plt.subplots(figsize=(8, 6))
    scatter_clust = ax_clust.scatter(
        proj[:, 0],
        proj[:, 1],
        c=best_labels,
        cmap="tab10",
        s=30,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.2,
    )
    # Create a legend for clusters instead of a colorbar since they are categorical
    handles, _ = scatter_clust.legend_elements()
    legend_labels = [f"Cluster {i}" for i in range(best_n_clusters)]
    ax_clust.legend(handles, legend_labels, title="Clusters")
    
    ax_clust.set_xlabel("UMAP-1")
    ax_clust.set_ylabel("UMAP-2")
    ax_clust.set_title(f"UMAP Clusters (k={best_n_clusters}, Silhouette={best_score:.2f})")
    fig_clust.tight_layout()
    
    cluster_output = args.output.with_name(args.output.stem + "_clusters" + args.output.suffix)
    fig_clust.savefig(cluster_output, dpi=200)
    plt.close(fig_clust)
    print(f"Saved clustered UMAP plot to {cluster_output}")


    # --- Decision Boundary & Critical Point Estimation ---

    if best_n_clusters == 2:

        print("Fitting SVM to determine decision boundary...")
        # Fit an SVM with RBF kernel to capture non-linear boundaries in UMAP space
        clf = SVC(kernel="rbf", C=1.0)
        clf.fit(proj, best_labels)

        # Create a meshgrid to plot the decision boundary
        x_min, x_max = proj[:, 0].min() - 1, proj[:, 0].max() + 1
        y_min, y_max = proj[:, 1].min() - 1, proj[:, 1].max() + 1
        # Dynamic step size for resolution
        h = max(x_max - x_min, y_max - y_min) / 300
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Predict on meshgrid
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig_bound, ax_bound = plt.subplots(figsize=(8, 6))
        # Plot contour of decision boundary
        ax_bound.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.3)

        # Scatter points colored by temperature
        scatter_bound = ax_bound.scatter(
            proj[:, 0],
            proj[:, 1],
            c=temps_np,
            cmap="viridis",
            s=30,
            edgecolor="black",
            linewidth=0.2,
            alpha=0.8,
        )

        # Find the point closest to the decision boundary
        # For binary classification, decision_function returns distance to hyperplane
        dists = np.abs(clf.decision_function(proj))
        closest_idx = np.argmin(dists)
        closest_temp = temps_np[closest_idx]
        closest_pt = proj[closest_idx]

        print(f"Point closest to decision boundary has Temperature: {closest_temp:.5f}")

        # Highlight the critical point
        ax_bound.scatter(
            [closest_pt[0]],
            [closest_pt[1]],
            s=200,
            c="red",
            marker="*",
            edgecolor="black",
            label=f"Tc ≈ {closest_temp:.4f}",
            zorder=10
        )
        ax_bound.legend(loc="upper right")

        cb_bound = plt.colorbar(scatter_bound, ax=ax_bound)
        cb_bound.set_label("Temperature")
        ax_bound.set_xlabel("UMAP-1")
        ax_bound.set_ylabel("UMAP-2")
        ax_bound.set_title("UMAP Decision Boundary & Critical Point")
        
        fig_bound.tight_layout()
        bound_output = args.output.with_name(args.output.stem + "_boundary" + args.output.suffix)
        fig_bound.savefig(bound_output, dpi=200)
        plt.close(fig_bound)
        print(f"Saved decision boundary plot to {bound_output}")
    else:
        print(f"Skipping decision boundary visualization: requires exactly 2 clusters, found {best_n_clusters}.")

if __name__ == "__main__":
    main()
