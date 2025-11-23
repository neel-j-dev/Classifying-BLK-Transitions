# !/bin/bash

# Train contrastive model on BLT dataset
# python -m GNN.train.train_contrastive --dataset XYModel/blt_dataset/train.npz --output GNN/artifacts/contrastive_model.joblib --hidden-dim 128 --num-layers 3 --projection-dim 64

# Train supervised model on BLT dataset
python -m GNN.train.train_pyg 