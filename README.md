# MSGeo-PINN-TransitionState

Multi-Scale Geometric Physics-Informed Neural Network for Transition State Structure Prediction.

## Overview

Given reactant and product 3D geometries in XYZ format, the model predicts transition state (TS) coordinates directly. The architecture combines graph neural networks with physics-informed constraints and multi-scale geometric feature extraction.

**Highlights:**
- Multi-scale geometric features (RDF, ADF, SOAP descriptors)
- Dual-branch GAT/GCN encoder with residual connections
- Physics constraint network (energy, force, geometry soft constraints)
- VAE-based coordinate generation with uncertainty estimation
- Two-stage iterative coordinate refinement
- Inference: ~0.06 s per reaction

## Architecture

```
Reactant Graph ─┐
                ├─ GNN Encoder ─ Feature Fusion ─ Physics Constraint ─ VAE Decoder ─ Coord Refiner ─ TS Prediction
Product Graph  ─┘
       │
Geometric Features (RDF + ADF + SOAP) ─ MLP Encoder ─┘
```

### Components

| Module | Description |
|--------|-------------|
| `MolecularGNN` | GCN + GAT with residual connections and layer normalization |
| `GeometricFeatureExtractor` | RDF (50 bins), ADF (36 bins), simplified SOAP (300-dim) |
| `PhysicsConstraintNetwork` | Energy/force/geometry soft constraints via tanh activations |
| `CoordinateRefiner` | Two-stage residual refinement (step sizes 0.1, 0.05) |
| `UncertaintyEstimator` | Per-atom confidence via learned variance (Softplus) |

## Results

### Test Set (500 reactions)

| Metric | Value |
|--------|-------|
| Mean RMSD | 0.713 Å |
| Median RMSD | 0.662 Å |
| Std RMSD | 0.348 Å |
| Success Rate (≤ 0.5 Å) | 27.8% |
| Mean Inference Time | 0.058 s |
| Model Parameters | 3,346,151 |

### Scoring

| Metric | Score |
|--------|-------|
| RMSD Score (40 pts) | 0.00 / 40 |
| Success Rate Score (30 pts) | 8.34 / 30 |
| **Total** | **8.34 / 70** |

### Figures

Training curves and evaluation results are in `results/figures/`.

## Installation

```bash
git clone https://github.com/ldcr6/MSGeo-PINN-TransitionState.git
cd MSGeo-PINN-TransitionState

conda create -n ts-pred python=3.10
conda activate ts-pred

pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric==2.3.1
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install -r requirements.txt
```

## Usage

### Data Format

```
data/
├── rxn0000/
│   ├── RS.xyz    # reactant
│   ├── PS.xyz    # product
│   └── TS.xyz    # transition state (training only)
```

XYZ format:
```
5
energy = -154.123
C    0.000    0.000    0.000
H    0.000    0.000    1.089
H    0.000    1.026   -0.363
H   -0.889   -0.513   -0.363
H    0.889   -0.513   -0.363
```

### Training

```bash
python scripts/train_model.py --data_dir data/processed --epochs 100 --batch_size 4 --lr 3e-4
```

### Prediction

```bash
python scripts/generate_competition_predictions.py \
    --model_path models/best_advanced_ts_model.pth \
    --test_dir data/test \
    --output_dir results/predictions
```

### Evaluation

```bash
python scripts/real_competition_evaluation.py \
    --ts_dir data/test \
    --ts_pred_dir results/predictions
```

## Project Structure

```
MSGeo-PINN-TransitionState/
├── src/                          # Core modules
│   ├── model.py                  # Base GNN model
│   ├── advanced_ts_model.py      # Full model with physics constraints
│   ├── data_processing.py        # XYZ parsing, data pipeline
│   ├── data_augmentation.py      # Coordinate perturbation
│   ├── enhanced_features.py      # RDF, ADF, SOAP extraction
│   ├── train.py                  # Training utilities
│   ├── predict.py                # Prediction utilities
│   └── utils.py                  # Helpers
│
├── scripts/                      # Executable scripts
│   ├── train_model.py            # Training entry point
│   ├── train_advanced_model.py   # Advanced training
│   ├── generate_competition_predictions.py
│   ├── real_competition_evaluation.py
│   └── preprocess_data.py
│
├── configs/
│   └── config.yaml               # Hyperparameters
│
├── results/
│   ├── evaluation/               # Evaluation JSON
│   ├── figures/                  # Training curves, evaluation plots
│   └── training_logs/            # Training iteration records
│
├── docs/
│   └── technical_report.md       # Technical report
│
├── data/examples/                # Example XYZ files
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Loss Function

```
L = 1.0*L_coord + 0.3*L_geo + 0.1*L_KL + 0.2*L_unc + 0.1*L_phys
```

- `L_coord`: 0.6*MSE + 0.4*Huber on coordinates
- `L_geo`: Distance matrix consistency
- `L_KL`: VAE KL-divergence regularization
- `L_unc`: Calibrated uncertainty (negative log-likelihood)
- `L_phys`: Clash penalty (d < 0.5 Å) and dispersion penalty (d > 15 Å)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 1e-4 |
| LR Schedule | CosineAnnealingWarmRestarts (T0=15, Tmult=2) |
| Batch Size | 4 |
| Early Stopping | patience=25 |
| Gradient Clipping | max_norm=1.0 |
| Data Augmentation | Gaussian noise (sigma=0.01, 0.02 A) |

## Dataset

Transition1x: 10,073 DFT-computed organic reactions with TS structures.

- 11 element types: H, C, N, O, F, Si, P, S, Cl, Br, I
- 5-50 atoms per molecule
- Reference: Schreiner et al., Sci. Data 9, 779 (2022)

## License

MIT
