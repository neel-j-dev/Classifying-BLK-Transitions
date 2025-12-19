from __future__ import annotations

import numpy as np

import os

os.environ.setdefault("MPLCONFIGDIR", str(os.getcwd()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    UnsupervisedPhasePipeline,
    ising_z2_quotient_l2_dist2_matrix,
    load_ising_configs_bin,
    ordered_fraction_curve,
    tc_eval_ordered_fraction_crossing,
    tc_eval_u_max_slope,
    xy_o2_overlap_dist2_matrix,
)

def run_xy() -> None:
    xy_npz = "xy_dataset_wolff_128x128.npz"
    with np.load(xy_npz) as z:
        theta = np.asarray(z["configurations"], dtype=np.float32)

    meta_path = "xy_dataset_wolff.metadata_128x128.json"
    import json

    meta = json.load(open(meta_path))
    temps_per_sample = np.array([float(r["temperature"]) for r in meta], dtype=float)
    order = np.argsort(temps_per_sample)
    temps = temps_per_sample[order]
    X = theta[order]
    mask = (temps >= 0.0) & (temps <= 1.2)
    temps = temps[mask]
    X = X[mask]

    pipeline = UnsupervisedPhasePipeline(
        dist2_matrix=xy_o2_overlap_dist2_matrix,
        tc_eval=tc_eval_u_max_slope,
    )
    res = pipeline.fit(temps, X)
    print(f"Loaded: {xy_npz}")
    print(f"N_embed={res.embedding.shape[0]}, sites={X.shape[1]}")
    print(f"epsilon (median all-pairs d^2) = {res.epsilon:.6g}")
    print(f"eigvals (top 8): {np.array2string(res.evals[:8], precision=4)}")
    print(f"embed_dim = {res.embedding.shape[1]}")
    print(f"Tc estimate (max slope of mean u) = {res.tc:.6g}")


def main() -> None:
    ising_bin = "Ising/ising_configs_L100.bin"
    ds = load_ising_configs_bin(ising_bin, block=None)
    n_samp = min(10, int(ds.X.shape[1]))
    configs = ds.configs[:, :n_samp].reshape(ds.temps.size * n_samp, -1)
    temps = np.repeat(ds.temps, n_samp)

    pipeline = UnsupervisedPhasePipeline(
        dist2_matrix=ising_z2_quotient_l2_dist2_matrix,
        tc_eval=lambda t, y: tc_eval_ordered_fraction_crossing(t, y, seed=0),
    )

    res = pipeline.fit(temps, configs)
    print(f"Loaded: {ising_bin}")
    print(f"Temps: n={ds.temps.size}, samples/temp={n_samp}, sites={configs.shape[1]}")
    print(f"N_embed={res.embedding.shape[0]}")
    print(f"epsilon (median all-pairs d^2) = {res.epsilon:.6g}")
    print(f"eigvals (top 8): {np.array2string(res.evals[:8], precision=4)}")
    print(f"embed_dim = {res.embedding.shape[1]}")
    print(f"Tc estimate (50% crossing) = {res.tc:.6g}")
if __name__ == "__main__":
    main()
