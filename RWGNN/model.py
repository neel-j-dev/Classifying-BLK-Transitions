"""Model definitions for the Random Walk GNN."""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn, Tensor
from torch_geometric.nn import GCNConv, global_mean_pool


LOG_2PI = math.log(2 * math.pi)


def _clamp_log_var(log_var: Tensor, *, min_val: float = -4.0, max_val: float = 4.0) -> Tensor:
    """Bound log-variance predictions to avoid degenerate negative losses."""

    return torch.clamp(log_var, min=min_val, max=max_val)


def _gaussian_nll(pred_mean: Tensor, pred_log_var: Tensor, target: Tensor) -> Tensor:
    """Stable Gaussian negative log-likelihood for 1D targets."""

    log_var = _clamp_log_var(pred_log_var)
    var = torch.exp(log_var)
    return 0.5 * (((pred_mean - target) ** 2) / var + log_var + LOG_2PI)


class RandomWalkGNN(torch.nn.Module):
    """GCN encoder with dual regression heads and a classifier head.

    The encoder digests lattice-node features enriched with random-walk
    positional encodings. The heads are:
    - ``phase_logits``: classifies ordered vs disordered phases.
    - ``temperature``: predicts the measurement temperature (mean & log variance).
    - ``critical_temperature``: estimates the critical temperature with an
      uncertainty-aware head that can be used to form Tc intervals.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128, dropout: float = 0.1):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.gcn3 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(p=dropout)

        self.phase_head = nn.Linear(hidden_channels, 2)

        self.temp_mean = nn.Linear(hidden_channels, 1)
        self.temp_log_var = nn.Linear(hidden_channels, 1)

        self.tc_mean = nn.Linear(hidden_channels, 1)
        self.tc_log_var = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gcn1(x, edge_index).relu()
        x = self.gcn2(x, edge_index).relu()
        x = self.dropout(self.gcn3(x, edge_index).relu())

        pooled = global_mean_pool(x, batch)

        phase_logits = self.phase_head(pooled)
        temp_mean, temp_log_var = self.temp_mean(pooled), self.temp_log_var(pooled)
        tc_mean, tc_log_var = self.tc_mean(pooled), self.tc_log_var(pooled)

        return {
            "phase_logits": phase_logits,
            "temp_mean": temp_mean,
            "temp_log_var": temp_log_var,
            "tc_mean": tc_mean,
            "tc_log_var": tc_log_var,
        }

    def compute_losses(self, outputs, *, phase_target: Tensor, temp_target: Tensor, tc_target: Tensor):
        """Compute the multi-task loss bundle."""

        classification = nn.CrossEntropyLoss()(outputs["phase_logits"], phase_target.view(-1))
        temp_nll = _gaussian_nll(outputs["temp_mean"], outputs["temp_log_var"], temp_target).mean()
        tc_nll = _gaussian_nll(outputs["tc_mean"], outputs["tc_log_var"], tc_target).mean()
        total = classification + temp_nll + tc_nll
        return total, {
            "phase_ce": classification.detach(),
            "temp_nll": temp_nll.detach(),
            "tc_nll": tc_nll.detach(),
        }

    @torch.no_grad()
    def predict_with_interval(self, data) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return phase probabilities and Tc confidence intervals for batches."""

        outputs = self(data)
        phase_probs = outputs["phase_logits"].softmax(dim=-1)

        tc_mean = outputs["tc_mean"].squeeze(-1)
        tc_log_var = _clamp_log_var(outputs["tc_log_var"].squeeze(-1))
        tc_std = torch.exp(0.5 * tc_log_var)
        lower = tc_mean - 1.96 * tc_std
        upper = tc_mean + 1.96 * tc_std
        return phase_probs, lower, upper, tc_mean


__all__ = ["RandomWalkGNN"]
