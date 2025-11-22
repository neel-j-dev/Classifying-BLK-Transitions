# Random Walk GNN pipeline for BLT detection

This module follows the methodology from *Discovering Phase Transitions with Unsupervised Learning from Random Walk Encodings* (arXiv:2206.12575) by:
- encoding lattice connectivity with random-walk positional encodings (RWPE) instead of relying solely on Euclidean coordinates, capturing diffusion-scale structure near the BKT transition;
- feeding the concatenated spin (cos, sin) features and RWPE into a graph neural network that jointly predicts the binary phase (below/above critical temperature) and regresses the temperature itself.

The workflow is:
1. Generate raw XY-model splits with `XYModel/generate_blt_dataset.py` (20×20 lattice by default).
2. Run `RWGNN/gen_dataset_rwgnn.py` to convert each split to PyTorch Geometric graphs, attach RWPE via `AddRandomWalkPE`, and store the processed graphs in `RWGNN/blt_rwgnn_dataset`.
3. Train/evaluate with `RWGNN/inference.py`, which reports accuracy/F1, temperature MAE/RMSE, saves a confusion matrix, and plots true vs. predicted temperatures with a regression line.
