from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def load_split(split_path: Path | str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    split_path = Path(split_path)
    with np.load(split_path) as npz:
        features = npz["features"].astype(np.float64)
        temps = npz["temperatures"].astype(np.float64)
        labels = npz["labels"].astype(np.int64)
    return features, temps, labels


FEATURE_NAMES = [
    "magnetization_x",
    "magnetization_y",
    "magnetization",
    "correlation_length",
    "specific_heat",
    "energy_density",
    "angle_variance",
]
