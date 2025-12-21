# Classifying BKT Transitions

Python tooling, notebooks, and C++ helpers for exploring phase classfication in the Ising Model and  Berezinskii–Kosterlitz–Thouless (BKT) physics in the 2D XY model. The repo includes simulators, dataset builders, and unsupervised analyses (diffusion maps, clustering, critical-temperature heuristics).

## Contents
- `simulator/xy_sim.py` — XY-model simulator (Metropolis or Wolff), dataset builder, and analysis utilities.
- `simulator/XY_Diffusion_Map.ipynb` — end-to-end diffusion-map exploration, clustering, silhouette sweeps, and Tc heuristics vs system size and kernel bandwidth.
- `unsupervised_analysis.py` — scriptable analyses of embeddings and clustering.
- `ising/` — simple Ising reference (C++/notebook).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate XY data (Python)
Use Wolff updates for faster decorrelation near criticality:
```bash
python - <<'PY'
from xy.xy_sim import build_xy_dataset
build_xy_dataset(
    temperatures=[0.9, 1.0, 1.1],
    samples_per_temp=50,
    lattice_shape=(64, 64),
    burn_in_sweeps=1000,
    sweeps_per_sample=50,
    proposal_width=3.14,
    save_path="xy_dataset_wolff_custom",
    update="wolff",
)
PY
```
Outputs a compressed `.npz` plus a `metadata.json` alongside it.

## Run diffusion-map notebook
Open `simulator/XY_Diffusion_Map.ipynb` in Jupyter/VS Code. It:
- Loads a dataset/metadata pair.
- Builds O(2)-invariant diffusion embeddings.
- Visualizes 2D/3D embeddings colored by temperature.
- Runs K-means on the first three coordinates, sweeps silhouette vs `k`, and inspects spectral gaps.
- Estimates Tc vs system size (1/(log L)² fit) and vs diffusion kernel bandwidth; overlays Tc=0.89 reference.

## CLI analysis helper
`unsupervised_analysis.py` provides reusable functions for loading embeddings, clustering, and plotting (useful when running headless). Import and call specific helpers or adapt as needed.

## Ising reference
`ising/ising.cpp` and `ising/Ising.ipynb` contain a minimal Ising implementation/visualization for comparison with the XY results.
`ising/ising.cpp` was forked and originally from `https://github.com/VictorSeven/IsingModel`

## Notes
- Default requirements: NumPy/SciPy, scikit-learn, matplotlib, torch/torch-geometric (for GNN experiments), pacmap/umap-learn for optional manifold learning.
- Large lattice sizes can be slow with plain Metropolis; prefer Wolff (`update="wolff"`) when possible.
