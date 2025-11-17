"""Model factories for the BLT GNN experiments."""

from __future__ import annotations

from typing import Literal
from sknetwork.gnn import GNNClassifier

def build_model(
    model_type: Literal["gnn", "gat"],
    input_dim: int,
    hidden_dim: int,
    n_epochs: int,
    learning_rate: float,
    random_state: int,
):
    dims = [hidden_dim, hidden_dim, 2]
    activations = ["ReLu", "ReLu", "Identity"]
    common_kwargs = {
        "dims": dims,
        "activations": activations,
        "learning_rate": learning_rate,
        #"n_epochs": n_epochs,
        #"random_state": random_state,
        "use_bias": True,
        "self_embeddings": True,
        "verbose": False,
    }

    if model_type == "gnn":
        return GNNClassifier(
            **common_kwargs,
            layer_types=["Conv", "Conv", "Conv"],
            normalizations=["both", "both", "both"],
        )
