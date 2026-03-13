"""
Training and evaluation core for XGBoost model.
"""

import os
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import config


def train_and_eval_oof(X, y, w_norm, best_params, feature_types, n_folds=5):
    """
    Evaluate Out-Of-Fold performance on training set to find optimal iteration count.

    Parameters
    ----------
    X : pd.DataFrame
        Training feature matrix.
    y : np.ndarray
        Training target vector.
    w_norm : np.ndarray
        Normalized training weights.
    best_params : dict
        Hyperparameters found during search.
    feature_types : list
        List of feature type indicators.
    n_folds : int, optional
        Number of folds for CV, by default 5.

    Returns
    -------
    mean_best_iteration : int
        The average best iteration (n_estimators) across all folds.
    """
    print("\n[Trainer] Evaluating OOF Performance on Training Set...")

    final_params = {
        "objective":         "binary:logistic",
        "eval_metric":       "auc",
        "tree_method":       "hist",
        "n_jobs":            -1,
        "verbosity":         0,
        "seed":              config.SEED,
        "enable_categorical": True,
        **best_params,
    }

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)
    best_iterations = []

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        w_tr, w_va = w_norm[tr_idx], w_norm[va_idx]

        model = xgb.XGBClassifier(n_estimators=5000, early_stopping_rounds=50,
                                  feature_types=feature_types, **final_params)
        model.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_va, y_va)],
                  sample_weight_eval_set=[w_va],
                  verbose=False)

        best_iterations.append(model.best_iteration)
        print(f"  Fold {fold_i + 1}: Best Iteration = {model.best_iteration}")

    return int(np.mean(best_iterations))


def train_final_model(X, y, w_norm, best_params, iters, feature_types, scale_lr=False, save_name=None):
    """
    Train a production model on the provided dataset.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Target vector.
    w_norm : np.ndarray
        Sample weights.
    best_params : dict
        Hyperparameters.
    iters : int
        Number of trees (iterations).
    feature_types : list
        Feature type indicators.
    scale_lr : bool, optional
        If True, decreases learning rate by 5x and increases iterations by 5x 
        for a smoother final model, by default False.
    save_name : str, optional
        Filename to save the JSON model, by default None.

    Returns
    -------
    model : xgb.XGBClassifier
        The trained model.
    """
    params = {
        "objective":         "binary:logistic",
        "eval_metric":       "auc",
        "tree_method":       "hist",
        "n_jobs":            -1,
        "verbosity":         0,
        "seed":              config.SEED,
        "enable_categorical": True,
        **best_params,
    }

    if scale_lr:
        params["learning_rate"] = best_params["learning_rate"] / 5.0
        n_est = iters * 5
        print(f"  [Final Training] Learning Rate scaled: {params['learning_rate']:.4f}, Estimators: {n_est}")
    else:
        n_est = iters
        print(f"  [Training] Training with {n_est} estimators...")

    model = xgb.XGBClassifier(n_estimators=n_est, feature_types=feature_types, **params)
    model.fit(X, y, sample_weight=w_norm, verbose=False)

    if save_name:
        model_path = os.path.join(config.OUTPUT_DIR, save_name)
        model.save_model(model_path)
        print(f"  Model saved to: {model_path}")

    return model


def plot_roc_curve(X_tr, y_tr, w_tr, X_te, y_te, w_te, model):
    """
    Generate and save ROC curve comparing Train and Test set performance.

    Parameters
    ----------
    X_tr, y_tr, w_tr : pd.DataFrame, np.ndarray, np.ndarray
        Training set features, labels, and weights.
    X_te, y_te, w_te : pd.DataFrame, np.ndarray, np.ndarray
        Test set features, labels, and weights.
    model : xgb.XGBClassifier
        Trained model to evaluate.
    """
    print("\n[Plot] Generating ROC Curve (Train vs Test)...")
    fig, ax = plt.subplots(figsize=(6, 6))

    # Train ROC
    prob_tr = model.predict_proba(X_tr)[:, 1]
    fpr_tr, tpr_tr, _ = roc_curve(y_tr, prob_tr, sample_weight=w_tr)
    auc_tr = auc(fpr_tr, tpr_tr)
    ax.plot(fpr_tr, tpr_tr, lw=2.5, color="#4E79A7", label=f"Train (AUC = {auc_tr:.3f})")

    # Test ROC
    prob_te = model.predict_proba(X_te)[:, 1]
    fpr_te, tpr_te, _ = roc_curve(y_te, prob_te, sample_weight=w_te)
    auc_te = auc(fpr_te, tpr_te)
    ax.plot(fpr_te, tpr_te, lw=2.5, color="#E15759", label=f"Test  (AUC = {auc_te:.3f})")

    # Aesthetics
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set(xlim=[-0.02, 1.02], ylim=[-0.02, 1.05],
           xlabel="1 - Specificity (False Positive Rate)",
           ylabel="Sensitivity (True Positive Rate)",
           title="Receiver Operating Characteristic")
    ax.legend(loc="lower right", frameon=True, edgecolor="gray")
    ax.set_aspect("equal")

    save_path = os.path.join(config.OUTPUT_DIR, "ROC_Curve.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ROC Curve saved to: {save_path}")
