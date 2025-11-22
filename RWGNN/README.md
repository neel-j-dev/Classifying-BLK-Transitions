# Random Walk GNN for BLT Phase Detection

This folder contains a graph neural network workflow that mirrors the approach
in the cited BLT transition paper. It uses the existing XY model Metropolis
simulator to build periodic lattice graphs, augments node features with
random-walk positional encodings, and trains a PyTorch Geometric model to
predict both the BLT phase and a confidence interval for the critical
temperature.

## Layout
- `data.py`: utilities to translate XY lattice snapshots into `torch_geometric`
  `Data` objects, including random-walk positional encodings and helpers for
  loading the NPZ splits produced by `XYModel/generate_blt_dataset.py`.
- `model.py`: the `RandomWalkGNN` architecture with heads for phase
  classification, temperature regression, and uncertainty-aware critical
  temperature estimation.
- `train.py`: end-to-end training script that can either reuse NPZ splits or
  generate fresh simulation data on the fly.

## Quickstart
1. Generate BLT splits if you do not already have them:
   ```bash
   python XYModel/generate_blt_dataset.py --output-dir XYModel/blt_dataset
   ```
2. Train the Random Walk GNN on the saved splits:
   ```bash
   python -m RWGNN.train --dataset-dir XYModel/blt_dataset --epochs 20 --batch-size 12
   ```
   The script reports validation metrics each epoch, prints Tc intervals for a
   few samples, and saves `artifacts/rwg_nn.pt`.

## Estimating the critical interval
The model exposes `predict_with_interval`, which returns phase probabilities
and a 95% confidence interval around the learned critical temperature. This is
useful for quickly bracketing the transition point when evaluating new lattice
snapshots.
