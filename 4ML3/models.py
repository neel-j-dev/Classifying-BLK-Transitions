from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from scipy import sparse
from sknetwork.gnn import GNNClassifier

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


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
    elif model_type == "pyg":
        return PyGClassifier(
            dims=dims,
            activations=activations,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model_type='{model_type}'.")


class PyGClassifier:
    """Thin wrapper that imitates the scikit-network API using PyTorch Geometric."""

    def __init__(
        self,
        dims: Sequence[int],
        activations: Sequence[str],
        learning_rate: float,
        n_epochs: int,
        random_state: int,
    ):
        if len(dims) != len(activations):
            raise ValueError("dims and activations must have matching lengths.")

        self.dims = list(dims)
        self.activations = list(activations)
        self.learning_rate = learning_rate
        self.n_epochs = max(1, n_epochs)
        self.random_state = random_state

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self._data: Data | None = None
        self._probabilities: np.ndarray | None = None

    def _initialize(self, input_dim: int):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        self.model = PyGPhaseNet(input_dim, self.dims, self.activations)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def fit(self, adjacency: sparse.csr_matrix, labels: np.ndarray, features: np.ndarray):
        if adjacency.nnz == 0:
            raise ValueError("Adjacency matrix must contain at least one edge.")

        data = sparse_to_data(adjacency, features, labels)
        self._data = data.to(self.device)
        self._initialize(features.shape[1])
        criterion = nn.CrossEntropyLoss()

        assert self.model is not None
        assert self.optimizer is not None

        self.model.train()
        for _ in range(self.n_epochs):
            self.optimizer.zero_grad()
            logits = self.model(self._data.x, self._data.edge_index, self._data.edge_weight)
            loss = criterion(logits, self._data.y)
            loss.backward()
            self.optimizer.step()

        self._probabilities = self._forward_probabilities()
        return self

    def _forward_probabilities(self) -> np.ndarray:
        if self._data is None or self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._data.x, self._data.edge_index, self._data.edge_weight)
            probs = torch.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy()

    def predict_proba(self) -> np.ndarray:
        if self._probabilities is None:
            self._probabilities = self._forward_probabilities()
        return self._probabilities


class PyGPhaseNet(nn.Module):
    """Simple stack of GCNConv layers driven by the existing dims/activations config."""

    def __init__(self, input_dim: int, dims: Sequence[int], activations: Sequence[str]):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = [_activation_from_name(name) for name in activations]

        prev_dim = input_dim
        for dim in dims:
            self.layers.append(GCNConv(prev_dim, dim, add_self_loops=False, normalize=True))
            prev_dim = dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        for conv, activation in zip(self.layers, self.activations):
            x = conv(x, edge_index, edge_weight)
            x = activation(x)
        return x


def sparse_to_data(adjacency: sparse.csr_matrix, features: np.ndarray, labels: np.ndarray) -> Data:
    """Convert a scipy CSR adjacency into a PyG Data instance."""

    adjacency = adjacency.tocoo()
    edge_index = np.vstack([adjacency.row, adjacency.col]).astype(np.int64)
    edge_weight = adjacency.data.astype(np.float32)

    x = torch.from_numpy(features.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.int64))
    edge_index_tensor = torch.from_numpy(edge_index)
    edge_weight_tensor = torch.from_numpy(edge_weight)
    return Data(x=x, edge_index=edge_index_tensor, edge_weight=edge_weight_tensor, y=y)


def _activation_from_name(name: str) -> nn.Module:
    activations: dict[str, nn.Module] = {
        "relu": nn.ReLU(),
        "relu6": nn.ReLU6(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "identity": nn.Identity(),
        "softplus": nn.Softplus(),
    }
    key = name.lower()
    if key not in activations:
        raise ValueError(f"Unsupported activation '{name}'.")
    return activations[key]
