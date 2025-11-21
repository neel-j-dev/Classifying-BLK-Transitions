# !/bin/bash

# Inference and PCA visualization for contrastive model on BLT dataset
# python -m GNN.visualization.inference_contrastive --dataset XYModel/blt_dataset/test.npz  --output GNN/artifacts/contrastive_inference.json --pca-plot GNN/artifacts/contrastive_pca.png

# Inference and PCA visualization for supervised model on BLT dataset
# python -m GNN.visualization.inference
python -m GNN.visualization.plot_pca