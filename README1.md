# Prediction of Magnetic Ordering Temperature (TN) in Perovskite-Related Magnetic Compounds

This repository contains the implementation of a machine learning pipeline to predict the magnetic ordering temperature (TN) of perovskite-related magnetic compounds using **matminer** for feature generation and **CatBoost** for prediction.

## Overview

The project aims to develop a data-driven approach to predict the Néel temperature (TN) of perovskite-related magnetic compounds based on their chemical composition. Using **matminer** to generate a comprehensive set of compositional and electronic descriptors, we train and evaluate several machine learning models to achieve accurate predictions.

This work was conducted as part of a **Summer Research Project** at IIT Madras.

## Dataset

- **File:** `perovskites_clean.csv`
- **Total Samples:** 224
- **Target Variable:** TN (Néel temperature in Kelvin)
- **Features:** Chemical formula, crystal structure, lattice parameters, space group information

### Target Variable Statistics

| Statistic | Value |
|-----------|-------|
| Count     | 224   |
| Mean      | 201.02 K |
| Std       | 211.46 K |
| Min       | 0.91 K  |
| 25%       | 46.00 K |
| 50%       | 129.00 K |
| 75%       | 254.25 K |
| Max       | 750.00 K |

## Feature Engineering

### Matminer Feature Generation

The following featurizers from matminer were used to generate compositional descriptors:

1. **ElementProperty (Magpie)**: Generates 132 elemental property-based descriptors using the Magpie dataset
2. **Stoichiometry**: Creates stoichiometric descriptors
3. **ValenceOrbital**: Generates valence orbital statistics
4. **AtomicOrbitals**: Creates atomic orbital-based descriptors
5. **IonProperty**: Generates ion property descriptors

### Final Feature Set

- **Total Features:** 149 numerical features
- **Data Cleaning:** Missing values were imputed with column means

## Models Evaluated

1. **Random Forest Regressor**
2. **Extra Trees Regressor**
3. **XGBoost Regressor**
4. **Gradient Boosting Regressor**
5. **CatBoost Regressor**
6. **Support Vector Regressor (SVR)** (with StandardScaler)

## Model Performance

### Test Set Performance (80/20 Train-Test Split)

| Model | R² Score | MAE (K) |
|-------|----------|---------|
| **CatBoost** | **0.780** | **50.96** |
| Extra Trees | 0.745 | 51.76 |
| Random Forest | 0.648 | 58.67 |
| Gradient Boosting | 0.663 | 52.53 |
| XGBoost | 0.596 | 62.13 |
| SVR | 0.221 | 109.36 |

### Cross-Validation Performance (5-Fold)

| Model | Mean R² (CV) |
|-------|--------------|
| **CatBoost** | **0.791** |
| Extra Trees | 0.746 |
| Random Forest | 0.715 |

## Feature Importance Analysis (SHAP)

SHAP (SHapley Additive exPlanations) analysis was performed on the best-performing model (CatBoost) to identify the most influential features.

### Top 10 Features by SHAP Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | MagpieData avg_dev GSmagmom | 40.24 |
| 2 | MagpieData mean GSmagmom | 27.14 |
| 3 | MagpieData range GSmagmom | 24.08 |
| 4 | MagpieData maximum GSmagmom | 20.52 |
| 5 | MagpieData avg_dev GSvolume_pa | 6.91 |
| 6 | ... | ... |

### Feature Subset Testing

We tested the performance of models using only the top SHAP features. Interestingly, using a subset of features (Top 20, Top 50) often achieved comparable or better performance than using all features.

| Model | All Features | Top 50 | Top 20 | Top 10 |
|-------|--------------|--------|--------|--------|
| RF | 0.704 | 0.717 | 0.725 | 0.700 |
| ET | 0.752 | 0.773 | 0.764 | 0.765 |
| XGB | 0.662 | 0.670 | 0.700 | 0.684 |
| **CatBoost** | **0.791** | **0.800** | **0.791** | **0.744** |

## Key Insights

1. **Magnetic Moment Features Dominate**: The most important features are related to the ground state magnetic moment (GSmagmom) and magnetic moment averages, highlighting the critical role of magnetic properties in determining TN.

2. **Volume and Atomic Properties Also Matter**: Volume-related features (GSvolume_pa) and atomic properties show moderate importance.

3. **Feature Reduction is Viable**: Using only the top 20 SHAP-important features often achieves performance comparable to using the full feature set.

4. **CatBoost is the Best Performer**: CatBoost consistently outperforms other models across all feature subsets.

## Requirements

### Python Dependencies
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
catboost
matminer
pymatgen
shap
jupyter


## How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
jupyter notebook "Final Project.ipynb"

##File structure
├── Final Project.ipynb        # Main Jupyter notebook
├── perovskites_clean.csv      # Input dataset
├── perovskites_magpie.csv     # Magpie features output
├── perovskites_magpie_features.csv  # All features output
└── README.md                  # This file
