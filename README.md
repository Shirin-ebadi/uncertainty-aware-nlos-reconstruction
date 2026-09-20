# Uncertainty-Aware Physics-Informed NLOS Reconstruction

A PyTorch-based research project for Non-Line-of-Sight (NLOS) image reconstruction with predictive uncertainty estimation.

This project extends the baseline model from **"Physics to the Rescue: Deep Non-line-of-sight Reconstruction for High-speed Imaging"** by adding a lightweight uncertainty estimation branch to the reconstruction pipeline.

## Project Goal

The goal is to reconstruct hidden scenes while also estimating how reliable each reconstructed pixel is.

Instead of producing only a reconstruction image, the proposed method also predicts a pixel-wise uncertainty map.

## Main Contributions

- Added a predictive uncertainty head to the baseline NLOS reconstruction model
- Used a Heteroscedastic Gaussian Negative Log-Likelihood loss
- Added a direct calibration regularization term
- Kept the original reconstruction backbone frozen during uncertainty training
- Implemented a Physics-Calibrated uncertainty variant for ablation analysis
- Evaluated uncertainty using Pearson and Spearman correlation with reconstruction error

## Baseline Architecture

The baseline pipeline contains:

- Transient measurement input
- Physics-informed encoder
- Rayleigh-Sommerfeld Diffraction (RSD) propagation
- Conditional Neural Radiance Field
- Volume rendering
- NLOS reconstruction

The proposed method adds an uncertainty prediction branch to the radiance field / rendering pipeline.

## Uncertainty Estimation

The uncertainty branch predicts a positive variance value for each reconstructed pixel.

The training objective includes:

- Heteroscedastic Gaussian NLL
- Calibration loss between predicted variance and observed reconstruction error

Only the uncertainty branch is trained, while the main reconstruction backbone remains frozen.

## Training Configuration

- Framework: PyTorch
- Optimizer: Adam
- Learning rate: 3e-4
- Training iterations: 500
- Rays per iteration: 2048
- Additional trainable parameters: 33
- Variance clamp: 0.0001 to 0.25

## Evaluation

The final quantitative evaluation was performed on a 20-sample subset of the **Unseen Objects** split.

### Reconstruction Quality

| Method | RMSE ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|
| Baseline | 0.0712 | 23.049 | 0.7780 |
| Proposed Uncertainty | 0.0712 | 23.049 | 0.7777 |
| Physics-Calibrated | 0.0712 | 23.048 | 0.7776 |

### Uncertainty Metrics

| Metric | Proposed | Physics-Calibrated |
|---|---:|---:|
| Mean Pearson | 0.7081 | 0.7075 |
| Mean Spearman | 0.8384 | 0.8380 |
| Global Pearson | 0.6987 | 0.6982 |
| Global Spearman | 0.8177 | 0.8171 |
| High/Low Error Ratio | 19.41x | 19.30x |

The uncertainty output showed a strong relationship with reconstruction error while preserving reconstruction quality close to the baseline.

## Physics-Calibrated Variant

An additional ablation experiment used physics-based observability signals from the volume renderer, including:

- Ray hit probability
- Depth distribution entropy

In the current experiment, this additional physics regularization did not produce a meaningful improvement over the simpler uncertainty model.

## Files

- `nlos_uncertainty_main.py` — Main Colab-based implementation and experiment pipeline
- `README.md` — Project description and results

## Requirements

Main dependencies include:

```bash
pip install torch pyyaml tensorboard tensorboardX scipy matplotlib opencv-python-headless ninja gdown
