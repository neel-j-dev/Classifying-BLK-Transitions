"""Shared utilities for running BLT GNN experiments on a dataset split."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from data_utils import load_split
from graph_utils import build_knn_graph
from metrics import estimate_critical_temperature, phase_metrics
from models import build_model


def run_split(
    split_path: Path,
    split_name: str,
    model_type: str,
    hidden_dim: int,
    n_epochs: int,
    learning_rate: float,
    random_state: int,
    n_neighbors: int,
    metric: str,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, dict, np.ndarray, np.ndarray, StandardScaler]:
    """Train a model on the split and return probabilities + metrics."""

    features, temps, labels = load_split(split_path)
    if scaler is None:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = scaler.transform(features)

    adjacency = build_knn_graph(features_scaled, n_neighbors=n_neighbors, metric=metric)
    model = build_model(
        model_type=model_type,
        input_dim=features_scaled.shape[1],
        hidden_dim=hidden_dim,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    model.fit(adjacency, labels=labels, features=features_scaled)

    probs = model.predict_proba()[:, 1]
    stats = phase_metrics(labels, probs)
    stats["t_c"] = estimate_critical_temperature(temps, probs)
    stats["split"] = split_name
    return probs, stats, temps, model, scaler
