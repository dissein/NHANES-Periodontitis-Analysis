# XGBoost Periodontitis Analysis Pipeline

This repository contains a statistically robust machine learning pipeline for analyzing NHANES data related to periodontitis. It uses XGBoost with Optuna for hyperparameter optimization, SHAP for model interpretation, and spline-based cutoff analysis for clinical threshold extraction.

## Features
- **Data Preprocessing**: Handles complex NHANES weights and categorical variables.
- **Automated Tuning**: Hyperparameter search using Optuna with pruning to prevent overfitting.
- **Model Training**: Stratified K-Fold cross-validation for robust performance estimation.
- **Interpretation**: Global and local feature importance via SHAP.
- **Clinical Thresholds**: Automated extraction of risk/safety thresholds using Univariate Splines on SHAP values.
- **Visualization**: Academic-style plots for ROC curves, SHAP importance, and dependence.

## Structure
- `code/`: Source code modules.
- `LICENSE`: MIT License.
- `.gitignore`: Rules to exclude local data and results from the repository.

## Getting Started
1. **Prepare Data**: Place your `data.csv` in the project root directory.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the analysis**:
   ```bash
   python code/main.py
   ```

## Requirements
- Python 3.8+
- XGBoost
- Optuna
- SHAP
- Scikit-Learn
- Pandas
- Matplotlib
- Seaborn
- SciPy
