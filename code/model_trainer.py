import os
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
import config


def calculate_best_metrics(y_true, y_prob, weights=None):
    """
    Find the optimal threshold using Youden's Index and calculate clinical metrics.
    
    Youden's Index = Sensitivity + Specificity - 1
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, sample_weight=weights)
    
    # Calculate Youden's Index for each threshold
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_idx]
    
    # Predictions at best threshold
    y_pred = (y_prob >= best_threshold).astype(int)
    
    # Confusion Matrix (handle weights if provided)
    if weights is not None:
        # Weighted confusion matrix is tricky with standard sklearn, 
        # but we can calculate Sens/Spec manually
        pos_weight = np.sum(weights[y_true == 1])
        neg_weight = np.sum(weights[y_true == 0])
        
        tp = np.sum(weights[(y_true == 1) & (y_pred == 1)])
        tn = np.sum(weights[(y_true == 0) & (y_pred == 0)])
        
        sensitivity = tp / pos_weight if pos_weight > 0 else 0
        specificity = tn / neg_weight if neg_weight > 0 else 0
    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
    
    f1 = f1_score(y_true, y_pred, sample_weight=weights)
    
    return {
        "threshold": best_threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "auc": auc(fpr, tpr)
    }


def train_and_eval_oof(X, y, w_norm, best_params, feature_types, n_folds=5):
    """
    Evaluate Out-Of-Fold performance and generate 5-fold CV ROC curve.
    """
    print(f"\n[Trainer] Performing {n_folds}-fold Cross-Validation...")

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
    cv_metrics = []
    
    # For ROC Plotting
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    fig, ax = plt.subplots(figsize=(7, 7))

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

        # Collect results
        best_iterations.append(model.best_iteration)
        y_prob = model.predict_proba(X_va)[:, 1]
        
        metrics = calculate_best_metrics(y_va, y_prob, weights=w_va)
        cv_metrics.append(metrics)
        
        # ROC curve for this fold
        fpr, tpr, _ = roc_curve(y_va, y_prob, sample_weight=w_va)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # Interp for mean ROC
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        ax.plot(fpr, tpr, lw=1, alpha=0.3, label=f"Fold {fold_i+1} (AUC = {roc_auc:.3f})")
        print(f"  Fold {fold_i + 1}: AUC = {roc_auc:.3f}, Sens = {metrics['sensitivity']:.3f}, Spec = {metrics['specificity']:.3f}")

    # Calculate Mean and Std
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    ax.plot(mean_fpr, mean_tpr, color="b", label=f"Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})", lw=2, alpha=0.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.")

    # Aesthetics
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=f"{n_folds}-Fold Cross-Validation ROC")
    ax.set_xlabel("1 - Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.legend(loc="lower right", fontsize="small")
    
    # Save in multiple formats for convenience and academic quality
    for ext in [".png", ".svg"]:
        save_path = os.path.join(config.OUTPUT_DIR, f"ROC_Curve{ext}")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    # Summary Metrics
    df_metrics = pd.DataFrame(cv_metrics)
    summary_stats = df_metrics.mean().to_dict()
    summary_stats["auc_std"] = df_metrics["auc"].std()
    
    print("\n[Summary] 5-Fold CV Results (at optimal Youden threshold):")
    print(f"  Mean AUC:         {summary_stats['auc']:.4f} ± {summary_stats['auc_std']:.4f}")
    print(f"  Mean Sensitivity: {summary_stats['sensitivity']:.4f}")
    print(f"  Mean Specificity: {summary_stats['specificity']:.4f}")
    print(f"  Mean F1-Score:    {summary_stats['f1_score']:.4f}")
    
    # Save these metrics to a CSV
    df_metrics.to_csv(os.path.join(config.OUTPUT_DIR, "CV_Performance_Metrics.csv"), index=False)
    
    return int(np.mean(best_iterations))


def train_final_model(X, y, w_norm, best_params, iters, feature_types, scale_lr=False, save_name=None):
    """
    Train a production model on the provided dataset.
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
    Simplified ROC plot for visual reference (Legacy support).
    """
    print("\n[Plot] Generating Reference ROC Curve...")
    fig, ax = plt.subplots(figsize=(6, 6))

    for X_set, y_set, w_set, label, color in [
        (X_tr, y_tr, w_tr, "Train", "#4E79A7"),
        (X_te, y_te, w_te, "Test", "#E15759")
    ]:
        prob = model.predict_proba(X_set)[:, 1]
        fpr, tpr, _ = roc_curve(y_set, prob, sample_weight=w_set)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.5, color=color, label=f"{label} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set(xlim=[-0.02, 1.02], ylim=[-0.02, 1.05],
           xlabel="1 - Specificity", ylabel="Sensitivity",
           title="Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")

    # Do not overwrite the main ROC_Curve.png if we want to keep CV as primary
    save_path = os.path.join(config.OUTPUT_DIR, "ROC_Holdout_Reference.png")
    fig.savefig(save_path)
    plt.close(fig)

