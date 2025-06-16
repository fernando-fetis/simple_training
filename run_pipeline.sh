#!/bin/bash

# Exportar W&B API key:
export WANDB_API_KEY=[COLOCAR WANDB KEY AQUÍ]

# Entrenamiento:
echo "Iniciando entrenamiento..."
python3 train.py

# Inferencia:
echo "Realizando predicción..."
python3 predict.py