"""
Hyperparameter optimization utilities using Optuna for XGBoost.
"""

import os
import numpy as np
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
from optuna.samplers import TPESampler
from optuna.integration import XGBoostPruningCallback
from optuna.exceptions import TrialPruned
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import config


def tune_hyperparameters(X, y, w_norm, feature_types, n_trials=200, n_folds=5):
    """
    Perform hyperparameter search using Optuna with Stratified K-Fold CV.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Target vector.
    w_norm : np.ndarray
        Normalized sample weights.
    feature_types : list
        List of feature type indicators ('c' or 'q').
    n_trials : int, optional
        Number of Optuna trials, by default 200.
    n_folds : int, optional
        Number of folds for cross-validation, by default 5.

    Returns
    -------
    best_params : dict
        The best hyperparameters found.
    best_value : float
        The highest AUC achieved.
    study : optuna.study.Study
        The completed Optuna study object.
    """
    print("\n[Optuna] Starting Hyperparameter Search...")

    # Calculate base scale_pos_weight to handle class imbalance
    pos_w = np.sum(w_norm[y == 1])
    neg_w = np.sum(w_norm[y == 0])
    base_scale_pos_weight = neg_w / pos_w

    def objective(trial):
        """Optuna objective function for a single trial."""
        grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

        params = {
            "objective":          "binary:logistic",
            "eval_metric":        "auc",
            "tree_method":        "hist",
            "n_jobs":             -1,
            "verbosity":          0,
            "seed":               config.SEED,
            "scale_pos_weight":   trial.suggest_float("scale_pos_weight", base_scale_pos_weight * 0.8, base_scale_pos_weight * 1.2),
            "n_estimators":       5000,
            "max_depth":          trial.suggest_int("max_depth", 3, 7),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_weight":   trial.suggest_int("min_child_weight", 5, 50),
            "subsample":          trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "colsample_bylevel":  trial.suggest_float("colsample_bylevel", 0.5, 0.9),
            "colsample_bynode":   trial.suggest_float("colsample_bynode", 0.5, 0.9),
            "gamma":              trial.suggest_float("gamma", 0.0, 2.0),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "max_delta_step":     trial.suggest_int("max_delta_step", 1, 5),
            "grow_policy":        grow_policy,
        }
        if grow_policy == "lossguide":
            params["max_leaves"] = trial.suggest_int("max_leaves", 15, 64)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)
        fold_aucs = []

        for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            w_tr, w_val = w_norm[train_idx], w_norm[val_idx]

            # Pruning Logic: Only bind to Optuna callback on the first fold to avoid step conflicts
            callbacks = []
            if fold_i == 0:
                pruning_callback = XGBoostPruningCallback(trial, "validation_0-auc")
                callbacks.append(pruning_callback)

            model = xgb.XGBClassifier(
                **params,
                enable_categorical=True,
                early_stopping_rounds=50,
                callbacks=callbacks,
                feature_types=feature_types
            )
            model.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_val, y_val)],
                sample_weight_eval_set=[w_val],
                verbose=False,
            )
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            fpr, tpr, _ = roc_curve(y_val, y_pred_proba, sample_weight=w_val)
            fold_auc = auc(fpr, tpr)
            fold_aucs.append(fold_auc)

            # Manual pruning check for sub-sequent folds
            if fold_i > 0:
                current_mean_auc = np.mean(fold_aucs)
                # Report with a large step offset to avoid overlap with the fold_0 callback steps
                trial.report(current_mean_auc, step=10000 + fold_i)
                if trial.should_prune():
                    raise TrialPruned()

        return np.mean(fold_aucs)

    # Initialize Study
    sampler = TPESampler(seed=config.SEED, multivariate=True, n_startup_trials=30)
    pruner = optuna.pruners.HyperbandPruner(min_resource=10, max_resource=5000, reduction_factor=3)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study.best_value, study


def save_optimization_history(study, output_dir=None):
    """
    Plot and save the Optuna optimization history.

    Parameters
    ----------
    study : optuna.study.Study
        The Optuna study object to visualize.
    output_dir : str, optional
        Directory to save the plot. Defaults to config.OUTPUT_DIR.
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR

    print(f"\n[Plot] Generating Optimization History Plot...")

    try:
        from optuna.visualization.matplotlib import plot_optimization_history

        # Use context manager to enforce clean white background for academic publication
        with plt.rc_context({
            "axes.facecolor":  "white",
            "figure.facecolor": "white",
            "axes.edgecolor":  "black",
            "grid.color":      "#EEEEEE"
        }):
            if hasattr(config, 'set_academic_style'):
                config.set_academic_style()

            ax = plot_optimization_history(study)

            # Refine labels and grid
            plt.title("Hyperparameter Optimization History", fontsize=14, pad=20)
            plt.grid(True, linestyle='--', alpha=0.5)

            save_path = os.path.join(output_dir, "Optuna_Optimization_History.png")
            plt.savefig(save_path, facecolor='white', edgecolor='none')
            plt.close()
            print(f"[Plot] Optimization History saved to: {save_path}")

    except Exception as e:
        print(f"[Warning] Failed to generate plot: {e}")
        print("Please ensure 'optuna' and 'matplotlib' are correctly installed.")
