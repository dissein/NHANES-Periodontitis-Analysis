"""
SHAP-based model interpretation and visualization utilities.
"""

import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import seaborn as sns
import config


def get_labeled_features(feature_list):
    """Map feature names to human-readable labels from config."""
    return [config.FEATURE_LABELS.get(f, f) for f in feature_list]


def run_shap_analysis(final_model, X, feature_cols, top_n=None):
    """
    Generate SHAP values and identify top important features.

    Parameters
    ----------
    final_model : xgb.XGBClassifier
        The trained XGBoost model.
    X : pd.DataFrame
        Dataset to explain.
    feature_cols : list
        List of feature names.
    top_n : int, optional
        Number of top features to extract, by default config.SHAP_TOP_N.

    Returns
    -------
    X_shap : pd.DataFrame
        Numeric version of features for plotting (categorical codes converted to floats).
    shap_values : np.ndarray
        Raw SHAP values.
    shap_importance : pd.DataFrame
        Sorted feature importance table.
    important_feats : list
        List of the top N important feature names.
    """
    if top_n is None:
        top_n = config.SHAP_TOP_N

    print("\n[SHAP] Running SHAP Analysis (Interpretation)...")

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X)

    # Calculate Mean Absolute SHAP for importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Save importance results
    shap_importance = pd.DataFrame({
        "Feature":       feature_cols,
        "Label":         get_labeled_features(feature_cols),
        "Mean_abs_SHAP": mean_abs_shap
    }).sort_values("Mean_abs_SHAP", ascending=False)

    imp_csv_path = os.path.join(config.OUTPUT_DIR, "SHAP_Feature_Importance.csv")
    shap_importance.to_csv(imp_csv_path, index=False)
    print(f"  Feature importance saved to: {imp_csv_path}")

    important_feats = shap_importance.head(top_n)["Feature"].tolist()

    # Create X_shap: numeric version for calculation/plotting
    # Categorical string features are converted back to their numeric codes
    X_shap = X.copy()
    for col in config.CAT_FEATURES:
        X_shap[col] = pd.to_numeric(X_shap[col].astype(str), errors='coerce')

    return X_shap, shap_values, shap_importance, important_feats


def plot_shap_summary(shap_values, X, feature_cols, important_feats, top_n=None):
    """
    Generate Bar and Beeswarm summary plots.

    Parameters
    ----------
    shap_values : np.ndarray
        Raw SHAP values.
    X : pd.DataFrame
        Feature matrix.
    feature_cols : list
        Feature names.
    important_feats : list
        Top important feature names.
    top_n : int, optional
        Top N to plot, by default config.SHAP_TOP_N.
    """
    if top_n is None:
        top_n = config.SHAP_TOP_N

    print("[Plot] Generating SHAP summary plots...")

    # 1. Horizontal Bar Plot
    imp_df = pd.DataFrame({
        "Feature":       feature_cols,
        "Label":         get_labeled_features(feature_cols),
        "Mean_abs_SHAP": np.abs(shap_values).mean(axis=0)
    })
    imp_df = imp_df.nlargest(top_n, "Mean_abs_SHAP").sort_values("Mean_abs_SHAP", ascending=True)

    fig_bar, ax_bar = plt.subplots(figsize=(8, max(4, len(imp_df) * 0.45)))
    colors_bar = plt.cm.RdYlBu_r(np.linspace(0.2, 0.85, len(imp_df)))[::-1]
    bars = ax_bar.barh(imp_df["Label"], imp_df["Mean_abs_SHAP"],
                       color=colors_bar, edgecolor="white", height=0.65)

    # Add value labels
    for bar, val in zip(bars, imp_df["Mean_abs_SHAP"]):
        ax_bar.text(val + 0.002, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=8)

    ax_bar.set_xlabel("Mean |SHAP value|")
    ax_bar.set_title("SHAP Global Feature Importance")
    fig_bar.savefig(os.path.join(config.OUTPUT_DIR, "SHAP_Bar_Important.png"))
    plt.close(fig_bar)

    # 2. Optimized Beeswarm Plot
    # We manually override categorical feature colors to gray to avoid misleading color bars
    feat_idx = [feature_cols.index(f) for f in important_feats]
    shap_sub = shap_values[:, feat_idx]

    X_sub = X[important_feats].copy()
    for col in important_feats:
        if col in config.CAT_FEATURES:
            X_sub[col] = pd.to_numeric(X_sub[col].astype(str), errors='coerce')

    X_sub.columns = get_labeled_features(important_feats)

    plt.figure(figsize=(10, max(5, len(important_feats) * 0.45)))
    # Use sort=False to maintain order for manual color indexing
    shap.summary_plot(shap_sub, X_sub, plot_type="dot", show=False, sort=False)

    ax = plt.gca()
    n_feats = len(important_feats)

    # COLOR MANIPULATION LOGIC:
    # Iterate through matplotlib objects to find beeswarm points and gray out categorical rows.
    for child in ax.get_children():
        if isinstance(child, PathCollection):
            offsets = child.get_offsets()
            array_data = child.get_array()

            if len(offsets) > 0 and array_data is not None:
                # Map array data to RGBA using the current colormap
                facecolors = child.to_rgba(array_data)
                edgecolors = child.get_edgecolors()

                if len(edgecolors) <= 1:
                    edgecolors = np.repeat(edgecolors if len(edgecolors) == 1 else [[0, 0, 0, 0]], len(offsets), axis=0)

                new_colors = facecolors.copy()
                new_edges = edgecolors.copy()

                for cat_idx, feat in enumerate(important_feats):
                    if feat in config.CAT_FEATURES:
                        # Identify the Y-coordinate for this categorical feature
                        y_val = n_feats - 1 - cat_idx
                        mask = np.abs(offsets[:, 1] - y_val) < 0.6

                        # Apply gray colors to points in this row
                        new_colors[mask] = [0.6, 0.6, 0.6, 1.0]
                        new_edges[mask] = [0.4, 0.4, 0.4, 1.0]

                # Re-assign colors and disable dynamic colormap to keep the grays fixed
                child.set_facecolors(new_colors)
                child.set_edgecolors(new_edges)
                child.set_array(None)
                child.set_alpha(None)

    plt.title("SHAP Beeswarm Plot (Categorical Factors in Gray)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "SHAP_Beeswarm_Important.png"), dpi=300)
    print(f"  Beeswarm plot saved to: {os.path.join(config.OUTPUT_DIR, 'SHAP_Beeswarm_Important.png')}")
    plt.close("all")


def plot_shap_dependence(shap_values, X_shap, feature_cols, important_feats):
    """
    Generate SHAP Dependence plots for top features.

    Categorical features are plotted as strip plots.
    Continuous features are plotted as scatter plots.
    """
    print("[Plot] Generating SHAP dependence plots...")
    n_imp = len(important_feats)
    if n_imp == 0:
        return

    dep_dir = os.path.join(config.OUTPUT_DIR, "dependence_plots")
    os.makedirs(dep_dir, exist_ok=True)

    ncols = min(3, n_imp)
    nrows = int(np.ceil(n_imp / ncols))
    fig_grid, axes_grid = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows * ncols == 1:
        axes_grid = np.array([axes_grid])
    axes_grid = axes_grid.flatten()

    for i, feat in enumerate(important_feats):
        label = config.FEATURE_LABELS.get(feat, feat)
        idx = feature_cols.index(feat)
        s_vals = shap_values[:, idx]

        def create_plot(ax_obj, is_standalone=False):
            """Shared plotting implementation for grid and standalone files."""
            if feat in config.CAT_FEATURES:
                mapping = config.CAT_FEATURES[feat]
                codes = X_shap[feat].values
                cat_names = [mapping.get(int(c), "Missing") if pd.notnull(c) else "Missing" for c in codes]
                order = [mapping[k] for k in sorted(mapping.keys()) if k in mapping]

                sns.stripplot(x=cat_names, y=s_vals, ax=ax_obj, order=order,
                              jitter=True, size=3 if not is_standalone else 5,
                              palette="RdBu_r", hue=cat_names, legend=False, alpha=0.6)

                ax_obj.set_xticklabels(ax_obj.get_xticklabels(), rotation=25, ha='right',
                                       fontsize=9 if not is_standalone else 10)
            else:
                x_vals = X_shap[feat].values
                ax_obj.scatter(x_vals, s_vals, c=s_vals, cmap="RdBu_r", s=4 if not is_standalone else 12,
                               alpha=0.4, edgecolors='none')

            ax_obj.axhline(0, color="indianred", lw=1.2, ls="--", alpha=0.8)  # Risk Baseline
            ax_obj.set_xlabel(label, fontsize=11 if not is_standalone else 13, fontweight='bold', labelpad=8)
            ax_obj.set_ylabel("SHAP value", fontsize=11 if not is_standalone else 13, labelpad=8)
            ax_obj.set_title(label, fontsize=12 if not is_standalone else 15, pad=12)
            ax_obj.spines['top'].set_visible(False)
            ax_obj.spines['right'].set_visible(False)
            ax_obj.grid(axis='y', linestyle=':', alpha=0.6)

        # 1. Add to main grid
        create_plot(axes_grid[i], is_standalone=False)

        # 2. Save individual plot (clean filename)
        fig_ind, ax_ind = plt.subplots(figsize=(8, 6))
        create_plot(ax_ind, is_standalone=True)
        plt.tight_layout()
        clean_name = "".join([c if c.isalnum() else "_" for c in label])
        fig_ind.savefig(os.path.join(dep_dir, f"Dependence_{clean_name}.png"), dpi=300)
        plt.close(fig_ind)

    # Hide empty subplots
    for j in range(i + 1, len(axes_grid)):
        axes_grid[j].set_visible(False)

    fig_grid.tight_layout()
    fig_grid.savefig(os.path.join(config.OUTPUT_DIR, "SHAP_Dependence_Summary.png"), dpi=300)
    plt.close(fig_grid)
    print(f"  Dependence plots completed in: {dep_dir}")
