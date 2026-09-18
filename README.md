# Gene Expression Prediction and CNN vs Vision Transformer Comparison

A PyTorch deep learning project covering gene expression prediction and CIFAR-10 image classification. The project implements and compares neural network regression, Convolutional Neural Networks (CNNs), and Vision Transformers (ViTs) through structured experimentation, feature selection, regularisation, augmentation, and performance evaluation.

## Project Overview

This project explores two complementary deep learning applications:

1. **Gene Expression Prediction** — predicting the expression level of the `lacZ` gene from other genes using neural network regression.
2. **CIFAR-10 Image Classification** — implementing and comparing CNN and Vision Transformer architectures for image classification.

Both tasks follow a systematic experimental approach in which multiple model iterations are developed, with each iteration introducing documented architectural or training changes and evaluating their effect on performance.

## Repository Structure

```text
.
├── notebooks/
│   ├── task1_gene_expression_pytorch.ipynb
│   └── task2_cnn_vs_vit_cifar10.ipynb
├── data/
│   └── Ecoli70_sample_data.csv
├── README.md
└── .gitignore
```

CIFAR-10 is downloaded automatically through `torchvision` when the image-classification notebook is first executed and is not stored in this repository.

---

# Task 1 — Gene Expression Prediction

## Objective

Predict the expression level of the **lacZ** gene using the expression levels of 45 other genes from an *E. coli* gene expression dataset.

## Dataset

`Ecoli70_sample_data.csv` contains:

- 1,000 rows
- 46 gene-expression columns
- `lacZ` as the prediction target
- 45 candidate predictor genes
- Normalised, log-transformed expression values
- No missing values

Target statistics:

| Metric | Value |
|---|---:|
| Mean | 1.78 |
| Standard deviation | 1.49 |
| Minimum | -0.26 |
| Maximum | 5.64 |

## Methodology

The notebook performs:

- Exploratory data analysis
- Correlation analysis
- Feature selection
- Data scaling using `StandardScaler`
- Neural network training
- Held-out test evaluation

Shared experimental settings:

| Setting | Value |
|---|---|
| Random seed | 42 |
| Data split | 70% train / 15% validation / 15% test |
| Scaling | `StandardScaler`, fitted on training data |
| Loss | MSE |
| Optimiser | Adam |
| Metrics | MSE, RMSE, MAE, R² |

## Model Iterations

### Iteration 1 — Baseline

A fully connected network using all 45 gene features.

- Hidden layers: `128 → 64`
- Activation: ReLU
- Learning rate: `0.001`
- Epochs: 100
- Batch size: 32
- Parameters: 14,209

### Iteration 2 — Feature Selection

The same basic architecture is restricted to features selected using absolute correlation with `lacZ`.

- Parameters: 9,729
- Correlation threshold: 0.3
- Fallback: ten largest-magnitude correlations when no features meet the threshold

### Iteration 3 — Regularisation

A deeper network introduces stronger regularisation and training control.

- Hidden layers: `128 → 64 → 32`
- Batch normalisation
- Dropout: `0.3`
- Weight decay
- `ReduceLROnPlateau` learning-rate scheduling
- Epochs: 150
- Parameters: 12,161

## Results

| Iteration | MSE | RMSE | MAE | R² |
|---|---:|---:|---:|---:|
| 1 — Baseline | 3.8429 | 1.9603 | 1.5355 | -0.7504 |
| 2 — Feature Selection | 3.7910 | 1.9471 | 1.5889 | -0.7268 |
| 3 — Regularised | 2.2854 | 1.5118 | 1.2432 | -0.0410 |

The supplied sample shows very weak relationships between `lacZ` and the predictor genes. All three R² values are negative, indicating that the models do not outperform the mean-prediction baseline on the held-out test set.

An independent random forest experiment produced an R² of approximately `-0.072`, providing an additional check that the weak predictive performance is not limited to the neural network architecture.

## Limitations and Next Steps

- The supplied sample contains limited predictive signal for `lacZ`.
- Correlation-based feature selection captures linear relationships only.
- The feature-to-sample ratio allows overfitting and memorisation.
- A mean-predictor baseline should be reported alongside model results.
- Early stopping could reduce unnecessary training.
- K-fold cross-validation could provide a more robust estimate of model performance.
- The intended full Ecoli70 dataset could be used for further validation if available.

---

# Task 2 — CNN vs Vision Transformer on CIFAR-10

## Objective

Implement and compare convolutional and transformer-based architectures for image classification, focusing on accuracy, convergence, computational cost, and generalisation.

## Dataset

CIFAR-10 contains:

- 60,000 RGB images
- Image size: `32 × 32`
- 10 classes
- 50,000 training images
- 10,000 test images

Classes:

`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

Images are normalised and standardised per channel.

## Experimental Protocol

The architectures are trained under comparable conditions using consistent data splits, seeds, and training budgets within each comparison.

| Setting | Value |
|---|---|
| Random seed | 42 |
| Train / validation split | 45,000 / 5,000 |
| Loss | Cross-entropy |
| Metrics | Top-1 accuracy, per-class accuracy, confusion matrix |
| Recorded per run | Parameters, training time, loss curves, accuracy curves |

## CNN Iterations

### Iteration 1 — Baseline
A baseline convolutional stack establishes a reference point.

### Iteration 2 — Increased Capacity
The model introduces increased depth and batch normalisation to investigate whether additional capacity improves convergence.

### Iteration 3 — Regularised Residual Model
Residual connections, dropout, and augmentation are introduced to address overfitting and training degradation.

## Vision Transformer Iterations

### Iteration 1 — Baseline ViT
A small Vision Transformer using a fixed patch size and embedding dimension provides the initial reference model.

### Iteration 2 — Patch and Embedding Changes
Patch size and embedding dimension are adjusted to investigate the trade-off between sequence length and per-token representation capacity.

### Iteration 3 — Training Stability
Augmentation and training-stability techniques are introduced to improve data efficiency and generalisation.

## Comparative Analysis

The CNN and ViT models are compared using:

- Classification accuracy
- Per-class performance
- Confusion matrices
- Convergence behaviour
- Training stability
- Parameter count
- Wall-clock training time
- Generalisation and train/validation gaps
- Effects of data augmentation

The comparison also examines the architectural differences between CNN locality-based processing and the transformer's attention mechanism, particularly for small `32 × 32` images.

---

# Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab / GPU runtime

## Installation

```bash
pip install torch torchvision pandas numpy matplotlib scikit-learn
```

## Running the Project

Open either notebook and run the cells from top to bottom.

### Task 1

Place:

```text
Ecoli70_sample_data.csv
```

in the notebook's working directory, or update the `pd.read_csv()` path if the dataset is stored under `data/`.

Figures are generated as PNG files during execution.

### Task 2

CIFAR-10 is downloaded automatically by `torchvision` during the first run. A GPU runtime is recommended for training the image-classification models.

## Reproducibility

The experiments use random seed `42` for NumPy and PyTorch. Exact GPU reproducibility is not guaranteed because some CUDA operations can be non-deterministic.

## Project Goals

This project demonstrates practical experience with:

- Deep learning model development
- PyTorch
- Neural network regression
- Computer vision
- CNN architectures
- Vision Transformers
- Feature selection
- Regularisation
- Data augmentation
- Model evaluation
- Experimental design
- Reproducible machine learning

## Author

**Kirankumar Chikkamankanala Somashekar**

GitHub: [@Kirankumarcs01](https://github.com/Kirankumarcs01)
