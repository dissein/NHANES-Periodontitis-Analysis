"""
Main entry point for the XGBoost Periodontitis Analysis Pipeline.

This script coordinates:
1. Data loading and preprocessing.
2. Hyperparameter optimization using Optuna.
3. Model training and evaluation.
4. SHAP-based model interpretation.
5. Clinical threshold (cutoff) analysis.
"""

import os
import sys
import warnings
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Set backend for non-interactive plotting before importing pyplot
matplotlib.use("Agg")

# Add current directory to path if needed for local module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
import data_loader
import model_tuner
import model_trainer
import shap_analyzer
import cutoff_analyzer


def main():
    """Execute the full analysis pipeline."""
    # 0. Initial Setup
    warnings.filterwarnings("ignore")
    config.set_academic_style()
    
    print("=" * 60)
    print("XGBoost Periodontitis Analysis Pipeline (Statistically Robust)")
    print("=" * 60)

    # 1. Load and Preprocess Data
    df, X, y, w_norm, feature_cols = data_loader.load_and_preprocess_data()
    
    # Construct feature type mapping for XGBoost: 'q' for numeric, 'c' for categorical
    feature_types = ["c" if c in config.CAT_FEATURES else "q" for c in feature_cols]
    cat_detected = [f for f in feature_cols if f in config.CAT_FEATURES]
    print(f"[Model] Categorical features identified: {cat_detected}")

    # Step A: Split into Train (80%) and Test (20%)
    # Hyperparameter tuning must only be performed on the training set to avoid data leakage.
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w_norm, test_size=0.2, stratify=y, random_state=config.SEED
    )

    # 2. Hyperparameter Search (On Training Set Only)
    best_params, best_value, study = model_tuner.tune_hyperparameters(
        X_train, y_train, w_train, feature_types, n_trials=config.TUNER_TRIALS
    )
    
    # Save Optimization History Plot
    model_tuner.save_optimization_history(study)

    # 3. Find Optimal Iterations (via OOF validation on Train set)
    mean_best_iters = model_trainer.train_and_eval_oof(
        X_train, y_train, w_train, best_params, feature_types
    )
    
    # Save optimal hyperparameters to CSV
    final_params_to_log = best_params.copy()
    final_params_to_log["best_n_estimators"] = int(mean_best_iters)
    pd.DataFrame([final_params_to_log]).to_csv(
        os.path.join(config.OUTPUT_DIR, "Optimal_Hyperparameters.csv"), index=False
    )
    print(f"[Final] Optimal hyperparameters saved to: Optimal_Hyperparameters.csv")
    
    # 4. Train Validation Model (Check Generalization)
    val_model = model_trainer.train_final_model(
        X_train, y_train, w_train, best_params, mean_best_iters, feature_types
    )
    
    # 5. Evaluate Generalization (ROC Curve - Train vs Test)
    model_trainer.plot_roc_curve(X_train, y_train, w_train, X_test, y_test, w_test, val_model)

    # 6. Final "Production" Model (Train on Full Dataset for interpretation)
    print("\n[Final] Training production model on full dataset for interpretation...")
    final_model = model_trainer.train_final_model(
        X, y, w_norm, best_params, mean_best_iters, feature_types,
        scale_lr=True, save_name="XGBoost_Periodontitis_Model.json"
    )

    # 7. SHAP Analysis
    X_shap, shap_values, shap_importance, important_feats = shap_analyzer.run_shap_analysis(
        final_model, X, feature_cols
    )
    shap_analyzer.plot_shap_summary(shap_values, X, feature_cols, important_feats)
    shap_analyzer.plot_shap_dependence(shap_values, X_shap, feature_cols, important_feats)

    # 8. Clinical Cutoff Analysis
    cutoff_analyzer.analyze_cutoffs(X_shap, shap_values, feature_cols, important_feats, y, w_norm)

    print("\n" + "=" * 60)
    print("[DONE] Statistically robust modularized pipeline completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
