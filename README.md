# MFPOTGF

Official implementation of **MFPOTGF** for **Major Depressive Disorder (MDD)** classification using multi-tissue structural brain imaging features (**GM/WM/CSF**). This repository provides the PyTorch implementation of the full model architecture, including the proposed **FPOTGF** fusion module, together with a dataset loader for subject-wise training/validation splits.

## Implementation & Environment
All experiments were conducted on a **single NVIDIA GeForce RTX 4090 GPU**. The proposed network (overall architecture + FPOTGF module) was implemented with **PyTorch 2.2** and **Python 3.12.4**.

- Hardware: 1 × NVIDIA GeForce RTX 4090
- Software: Python 3.12.4, PyTorch 2.2

## Training Configuration
The final model was trained with the following settings:

- Optimizer: **Adam**
  - initial learning rate: **8e-4**
  - weight decay: **0**
  - (β1, β2) = **(0.9, 0.999)**
- Scheduler: **Cosine annealing**
  - T_max = **200**, η_min = **0**
- Loss: **CrossEntropyLoss**
- Batch size: **16**
- Training epochs: up to **200**
- Early stopping: monitored on **validation accuracy**
  - patience = **30**
- Model selection: best checkpoint chosen by the **highest validation ACC**

## Reproducibility
To ensure reproducibility, random seeds were fixed (**seed = 42**) for Python, NumPy, and PyTorch, and deterministic behavior was enabled in cuDNN where appli
