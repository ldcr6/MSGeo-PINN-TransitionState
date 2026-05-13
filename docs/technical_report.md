# MSGeo-PINN: Multi-Scale Geometric Physics-Informed Neural Network for Transition State Prediction

## Technical Report

---

## 1. Introduction

Transition state (TS) structures occupy the highest energy point along a reaction path and govern reaction rates and selectivities. Traditional quantum chemical methods (DFT, CCSD(T)) provide accurate TS geometries but scale poorly to large systems. This work presents a physics-informed graph neural network that predicts TS structures directly from reactant-product pairs.

## 2. Data

### 2.1 Dataset

Source: Transition1x dataset (Schreiner et al., Sci. Data 2022).

- 10,073 reactions with DFT-computed TS structures
- Element types: H, C, N, O, F, Si, P, S, Cl, Br, I (11 total)
- Atoms per molecule: 5-50
- Format: XYZ coordinate files (RS.xyz, PS.xyz, TS.xyz)

### 2.2 Preprocessing

**Graph construction:**
- Nodes: atom type one-hot encoding (11-dim) + normalized atomic number (1-dim)
- Edges: distance threshold 2.0 A, bidirectional
- Edge features: [distance, 1/distance]

**Geometric features (500-dim fixed):**
- Radial distribution function: 50 bins, range 0-10 A
- Angular distribution function: 36 bins, range 0-pi
- Simplified SOAP: neighbor counts at 5 cutoff radii + 10 angular cosine terms per atom, padded to 300-dim
- Atomic environment: min/max/mean/std of neighbor distances per atom, padded to 200-dim

**Data augmentation:**
- Gaussian noise injection: sigma = 0.01 A and 0.02 A
- Each training sample augmented to 3 variants

### 2.3 Train/Test Split

- Training: 4,153 reactions (from Transition1x)
- Test: 500 reactions (competition test set)
- Validation: 10% of training set

## 3. Model Architecture

### 3.1 Graph Neural Network Encoder

```
Input (node features) -> GCNConv + BatchNorm + ReLU (x3 layers)
                       -> GATConv (4-head attention, x3 layers)
                       -> Residual connections + LayerNorm
                       -> Global mean/max pooling -> graph embedding (256-dim)
```

Hidden dimension: 128 (base) / 256 (advanced)
Dropout: 0.1

### 3.2 Feature Fusion

Reactant and product graph embeddings are concatenated (512-dim) and projected through an MLP to produce a reaction context vector (256-dim). Atom-level features from both branches are fused via multi-head attention (8 heads).

### 3.3 Physics Constraint Network

Three parallel branches apply soft constraints:

```
h_energy  = h + 0.10 * tanh(MLP_energy(h))
h_force   = h_energy + 0.10 * tanh(MLP_force(h_energy))
h_geom    = h_force + 0.05 * tanh(MLP_geometry(h_force))
```

Constraints encode:
- Energy conservation: TS should lie between reactant and product
- Force balance: net forces at TS are approximately zero
- Geometry: no atomic clashes (d > 0.5 A), no excessive dispersion (d < 15 A)

### 3.4 VAE Decoder

A variational autoencoder maps the physics-constrained features to TS coordinates:

```
mu, logvar = Encoder(h_phys)
z = mu + sigma * epsilon,  epsilon ~ N(0, I)
T_raw = Decoder(z)  ->  (N_atoms x 3)
```

Latent dimension: hidden_dim / 4

### 3.5 Coordinate Refinement

Two-stage residual refinement:

```
Stage 1: T1 = T_raw + 0.1 * MLP_inter(T_raw)
Stage 2: T_final = T1 + 0.05 * MLP_opt(T1 || h_phys)
```

### 3.6 Uncertainty Estimation

```
sigma_pred = Softplus(MLP_unc(h_phys))
```

Trained via negative log-likelihood loss for calibrated uncertainty.

### 3.7 Model Parameters

Total: 3,346,151

| Component | Parameters |
|-----------|------------|
| GNN encoder | ~500K |
| Physics constraint network | ~200K |
| VAE encoder/decoder | ~1.5M |
| Coordinate refinement | ~800K |
| Attention mechanism | ~200K |
| Other (embeddings, norms) | ~150K |

## 4. Training

### 4.1 Loss Function

```
L = 1.0*L_coord + 0.3*L_geo + 0.1*L_KL + 0.2*L_unc + 0.1*L_phys
```

- L_coord = 0.6*MSE + 0.4*Huber (coordinate accuracy)
- L_geo = MSE(D_pred, D_true) (distance matrix consistency)
- L_KL = KL(q(z) || p(z)) (VAE regularization)
- L_unc = sum((pred-true)^2/sigma^2 + log(sigma^2)) (uncertainty calibration)
- L_phys = clash_penalty + dispersion_penalty

### 4.2 Optimization

- Optimizer: AdamW (lr=3e-4, weight_decay=1e-4)
- LR schedule: CosineAnnealingWarmRestarts (T0=15, Tmult=2, eta_min=1e-7)
- Warmup: 5 epochs linear ramp
- Batch size: 4
- Early stopping: patience=25 epochs
- Gradient clipping: max_norm=1.0

### 4.3 Training Details

- Hardware: NVIDIA RTX 4060 (8 GB VRAM)
- Framework: PyTorch 2.7.1 + PyTorch Geometric 2.3.1
- Training time: ~70 min for 18 epochs (early stopped)
- Best validation loss: 0.073

## 5. Evaluation

### 5.1 Metrics

**RMSD**: Root-mean-square deviation after Kabsch rigid alignment (rotation + translation).

```
RMSD = sqrt(1/N * sum(||U*T_pred + d - T_true||^2))
```

**Success rate**: Fraction of reactions with RMSD <= 0.5 A.

**Scoring (competition standard):**
- RMSD score (40 pts): 40 if RMSD <= 0.2; 40 - ((RMSD-0.2)/0.3)*40 if 0.2 < RMSD < 0.5; 0 if >= 0.5
- Success score (30 pts): success_rate * 30

### 5.2 Test Set Results (500 reactions)

| Metric | Value |
|--------|-------|
| Mean RMSD | 0.713 A |
| Median RMSD | 0.662 A |
| Std RMSD | 0.348 A |
| Min RMSD | 0.166 A |
| Max RMSD | 2.500 A |
| Success Rate (<=0.5 A) | 27.8% (139/500) |
| Mean Inference Time | 0.058 s |
| RMSD Score | 0.00 / 40 |
| Success Score | 8.34 / 30 |
| Total Score | 8.34 / 70 |

### 5.3 Ablation Study

| Configuration | RMSD (A) | Success Rate (%) |
|---------------|----------|------------------|
| Full model | 0.713 | 27.8 |
| - Physics constraints | 0.782 | 22.4 |
| - Geometric features | 0.841 | 18.6 |
| - Uncertainty loss | 0.739 | 25.2 |
| - VAE component | 0.731 | 26.0 |
| GNN only (baseline) | 0.924 | 14.2 |

Physics constraints and geometric features provide the largest improvements.

### 5.4 Training Iteration Log

See `results/training_logs/training_iterations.md` for per-epoch loss curves across 4 experiments.

## 6. Limitations

1. Accuracy gap with DFT: 0.713 A vs 0.05-0.1 A limits quantitative barrier predictions
2. Element coverage: 11 common elements; transition metals not covered
3. Conformational flexibility: large conformational changes remain challenging
4. Training data: 4,153 reactions may be insufficient for broad generalization

## 7. References

1. Eyring, H. J. Chem. Phys. 3, 107 (1935).
2. Schreiner et al. Sci. Data 9, 779 (2022).
3. Schutt et al. NeurIPS 992 (2017) - SchNet.
4. Kipf & Welling, ICLR 2017 - GCN.
5. Velickovic et al. ICLR 2018 - GAT.
6. Kabsch, W. Acta Cryst. A 32, 922 (1976).
7. Loshchilov & Hutter, ICLR 2019 - AdamW.
8. Gasteiger et al. ICLR 2020 - DimeNet++.
9. Liu et al. ICLR 2022 - SphereNet.
10. Grambow et al. Sci. Data 7, 137 (2020).
