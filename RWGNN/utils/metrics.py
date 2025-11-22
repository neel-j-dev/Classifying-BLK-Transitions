from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def estimate_critical_temperature(temperatures: np.ndarray, probs: np.ndarray) -> float:
    """Estimate T_c as the temperature where class-1 probability crosses 0.5."""

    order = np.argsort(temperatures)
    temps = temperatures[order]
    p = probs[order]

    labels = p >= 0.5
    transitions = np.where(np.diff(labels.astype(int)) != 0)[0]
    if len(transitions) == 0:
        # No clear crossing, pick the point closest to 0.5.
        idx = np.argmin(np.abs(p - 0.5))
        return float(temps[idx])

    idx = transitions[0]
    t1, t2 = temps[idx], temps[idx + 1]
    p1, p2 = p[idx], p[idx + 1]
    if p2 == p1:
        return float((t1 + t2) / 2)

    # Linear interpolation between the two points.
    weight = (0.5 - p1) / (p2 - p1)
    return float(t1 + weight * (t2 - t1))


def phase_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics
