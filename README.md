# BLT Phase Transition Classifier

This folder contains the GNN pipeline that classifies Berezinskii–Kosterlitz–Thouless (BLT) phases from statistics produced by the XY Metropolis simulation in the root of the repository.

## Environment setup

1. Create a dedicated Conda environment (Python 3.10 works well):
   ```bash
   conda create -n blt-gnn python=3.10
   conda activate blt-gnn
   ```
2. Install the runtime requirements:
   ```bash
   pip install -r requirements.txt
   ```

The training scripts depend on NumPy/SciPy, scikit-learn for feature scaling + k-NN graphs, and [scikit-network](https://scikit-network.readthedocs.io/) for the GNN and attention GNN estimators.

## Data generation (outside `4ML3/`)

Run the helper at the repository root to build fresh train/validation/test splits:

```bash
python generate_blt_dataset.py --output-dir blt_dataset --steps 200 --iters-per-step 120
```

The script leverages `xy.py` to sweep a temperature range, extracts coarse observables (magnetization vector, correlation length, specific heat, energy density, and angular variance) for each simulation, and labels snapshots as “vortex dominated” when `T >= T_c` (default `T_c = 0.89`). Three compressed files (`train.npz`, `val.npz`, `test.npz`) plus `metadata.json` are written outside the `4ML3/` folder as required.

## Graph view + architecture

- **Nodes:** each simulation snapshot (feature vector of seven observables).
- **Edges:** k-nearest-neighbor graph (default `k=8`) constructed by `graph_utils.build_knn_graph`; weights use an RBF kernel on scaled features.
- **Models:**
  - `model_type=gnn` builds a stack of dense + convolutional layers via `sknetwork.gnn.GNNClassifier`.
  - `model_type=gat` swaps the convolutional block for `sknetwork.gnn.GATClassifier` with attention-based message passing.
- **Critical temperature:** after inference we sort predictions by temperature and compute where the class-1 probability crosses 0.5 (linear interpolation) to estimate `T_c`.

## Workflow

1. **Train:**
   ```bash
   cd 4ML3
   python train.py --dataset-dir ../blt_dataset --model-type gat --artifact artifacts/gat_model.joblib
   ```
   The script trains on `train.npz`, optionally reruns on a validation split (`--val`), and stores the scaler plus probabilities computed on the training graph (GNNs in scikit-network are transductive, so the saved artifact is only meaningful for the nodes it was trained on).

2. **Validate:**
   ```bash
   python validate.py --split ../blt_dataset/val.npz --model-type gat
   ```

3. **Test:**
   ```bash
   python test.py --split ../blt_dataset/test.npz --model-type gat
   ```

Validation and testing scripts independently fit/evaluate the requested architecture on their respective splits, reporting accuracy, F1, ROC-AUC, and the inferred `T_c`. Repeat with `--model-type gnn` or sweep hyperparameters to compare variants.

4. **Inference / visualization:**
   ```bash
   python inference.py --artifact artifacts/gat_model.joblib --dataset ../blt_dataset/train.npz
   ```
   The script (and matching `inference.ipynb`) focuses purely on model outputs: it renders a global probability-vs-temperature scatter plot with inferred/true `T_c` markers, a heatmap of node probabilities vs. labels, and three zoomed “snapshot” panels that show how predictions evolve across low / mid / high temperature bands. This makes it easy to inspect the classifier’s confidence without rendering XY lattices or running vortex detection.

## Design notes

- Observables rather than raw lattice states keep the data small and stable while retaining phase information.
- Splits stay outside `4ML3/` but the scripts accept arbitrary paths, so you can regenerate data with different temperatures or lattice sizes without modifying the training code.
- The scaler fitted on the train split is stored with the model artifact, ensuring consistent preprocessing for validation and testing.
- Hyperparameters (neighbors, metric, hidden size, epochs, learning rate) are configurable through CLI flags for quick sweeps.
