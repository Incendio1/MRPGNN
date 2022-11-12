#!/bin/sh

GPU=0

echo "=====Cora====="
echo "---Fiexd Split---"
CUDA_VISIBLE_DEVICES=${GPU} python mrpgnn.py --dataset=Cora --K=10 --dropout=0.85 --dropnode_rate=0.15

echo "=====Cornell====="
echo "---Dense Split---"
CUDA_VISIBLE_DEVICES=${GPU} python mrpgnn.py --dataset=cornell --dropout=0.5 --alpha=0.35 --beta=0.2 --dropnode_rate=0.15






