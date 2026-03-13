"""
Automated clinical threshold (cutoff) extraction using SHAP values and Univariate Splines.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import config


def analyze_cutoffs(X_shap, shap_values, feature_cols, important_feats, y, w_norm):
    """
    Extract clinical thresholds for continuous features where SHAP = 0 or risk accelerates.

    Parameters
    ----------
    X_shap : pd.DataFrame
        Numeric version of features.
    shap_values : np.ndarray
        Raw SHAP values.
    feature_cols : list
        List of all feature names.
    important_feats : list
        Top important features to analyze.
    y : np.ndarray
        Target vector.
    w_norm : np.ndarray
        Normalized sample weights.

    Returns
    -------
    cutoff_results : list
        List of dictionaries containing extracted thresholds and associated prevalences.
    """
    print("\n[Cutoff] Running SHAP-based Cutoff Analysis...")
    
    # Only analyze continuous features (exclude categorical)
    cont_important = [f for f in important_feats if f not in config.CAT_FEATURES]
    cutoff_results = []
    
    if not cont_important:
        return []

    plot_dir = os.path.join(config.OUTPUT_DIR, "cutoff_plots")
    os.makedirs(plot_dir, exist_ok=True)

    ncols = min(3, len(cont_important))
    nrows = int(np.ceil(len(cont_important) / ncols))
    fig_grid, axes_grid = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows * ncols == 1:
        axes_grid = np.array([axes_grid])
    axes_grid = axes_grid.flatten()

    for i, feat in enumerate(cont_important):
        label = config.FEATURE_LABELS.get(feat, feat)
        idx = feature_cols.index(feat)
        
        raw_x = X_shap[feat].values.astype(float)
        raw_s = shap_values[:, idx]
        
        # Remove NaNs
        mask_valid = ~np.isnan(raw_x) & ~np.isnan(raw_s)
        if np.sum(mask_valid) < 30:
            continue
            
        x_vals = raw_x[mask_valid]
        s_vals = raw_s[mask_valid]
        y_vals = y[mask_valid]
        w_vals = w_norm[mask_valid]

        # Filter out outliers (1st and 99th percentiles) for stable spline fitting
        q1, q99 = np.percentile(x_vals, [1.0, 99.0])
        mask_range = (x_vals >= q1) & (x_vals <= q99)
        xs, ss = x_vals[mask_range], s_vals[mask_range]
        
        # STABILITY FIX: Add tiny random noise (jitter) to ensure strictly increasing x-values for spline
        xs_jitter = xs + np.random.normal(0, 1e-9, size=len(xs))
        sort_idx = np.argsort(xs_jitter)
        xs_jitter, ss = xs_jitter[sort_idx], ss[sort_idx]
        
        cutoff, method = np.nan, "N/A"
        
        try:
            # Fit Univariate Spline
            spline = UnivariateSpline(xs_jitter, ss, s=len(xs) * 2.0)
            x_grid = np.linspace(xs_jitter.min(), xs_jitter.max(), 1000)
            
            # Find Roots (where SHAP = 0)
            roots = spline.roots()
            roots = roots[(roots >= xs_jitter.min()) & (roots <= xs_jitter.max())]
            
            if len(roots) > 0:
                derivs_at_roots = spline.derivative()(roots)
                
                # Determine if the feature is a risk factor (positive trend) or protective (negative)
                overall_trend = np.corrcoef(xs_jitter, ss)[0, 1]
                
                # Identify points where risk starts increasing (positive derivative)
                pos_trend_roots = roots[derivs_at_roots > 0]
                # Identify points where risk starts decreasing (negative derivative)
                neg_trend_roots = roots[derivs_at_roots < 0]
                
                if overall_trend > 0 and len(pos_trend_roots) > 0:
                    cutoff = pos_trend_roots[0]
                    method = "Risk-Increase Threshold"
                elif overall_trend < 0 and len(neg_trend_roots) > 0:
                    cutoff = neg_trend_roots[0]
                    method = "Risk-Decrease (Safety) Threshold"
                else:
                    # Fallback: choose the root with the highest derivative magnitude
                    cutoff = roots[np.argmax(np.abs(derivs_at_roots))]
                    method = "Steepest SHAP=0 Equilibrium"
            else:
                # If no SHAP=0 root exists, find point of maximum acceleration (max derivative)
                d1 = spline.derivative()(x_grid)
                overall_trend = np.corrcoef(xs_jitter, ss)[0, 1]
                cutoff = x_grid[np.argmax(d1)] if overall_trend > 0 else x_grid[np.argmin(d1)]
                method = "Max Risk/Safety Acceleration"
                
            if cutoff < xs.min() or cutoff > xs.max():
                cutoff, method = np.nanmedian(x_vals), "Fallback (Median)"
                
        except Exception as e:
            print(f"      [Warning] Extraction failed for {feat}: {e}")
            cutoff, method = np.nanmedian(x_vals), "Fallback (Median)"

        # Calculate population prevalence above/below the threshold
        if not np.isnan(cutoff):
            above = x_vals >= cutoff
            pos_rate_high = np.average(y_vals[above], weights=w_vals[above]) if np.any(above) else np.nan
            pos_rate_low  = np.average(y_vals[~above], weights=w_vals[~above]) if np.any(~above) else np.nan
        else:
            pos_rate_high = pos_rate_low = np.nan

        cutoff_results.append({
            "Feature":          feat,
            "Label":            label,
            "Cutoff":           round(float(cutoff), 4) if not np.isnan(cutoff) else np.nan,
            "Method":           method,
            "Prevalence_Below": round(float(pos_rate_low), 4) if not np.isnan(pos_rate_low) else np.nan,
            "Prevalence_Above": round(float(pos_rate_high), 4) if not np.isnan(pos_rate_high) else np.nan
        })

        # --- Subplot Plotting ---
        ax = axes_grid[i]
        ax.scatter(x_vals, s_vals, c=s_vals, cmap="RdBu_r", s=2, alpha=0.3)
        ax.axhline(0, color="black", lw=0.6)
        if not np.isnan(cutoff):
            ax.axvline(cutoff, color="red", lw=1.2, ls="--")
            ax.set_title(f"{label}\n{cutoff:.2f}", fontsize=9)
        else:
            ax.set_title(f"{label}\n(N/A)", fontsize=9)

        # --- Individual Detailed Plot ---
        fig_ind, ax_ind = plt.subplots(figsize=(8, 6))
        ax_ind.scatter(x_vals, s_vals, c=s_vals, cmap="RdBu_r", s=15, alpha=0.5, edgecolors='none')
        ax_ind.axhline(0, color="gray", lw=1, ls="-")
        
        if not np.isnan(cutoff):
            ax_ind.axvline(cutoff, color="#E63946", lw=2, ls="--", label=f"Threshold: {cutoff:.2f}")
            ax_ind.legend(loc='best')
        
        ax_ind.set_title(f"Threshold Analysis: {label}", fontsize=14, pad=15)
        ax_ind.set_xlabel(label, fontsize=11)
        ax_ind.set_ylabel("SHAP value", fontsize=11)
        
        clean_name = "".join([c if c.isalnum() or c == '_' else '_' for c in label])
        fig_ind.savefig(os.path.join(plot_dir, f"Threshold_{clean_name}.png"), dpi=300)
        plt.close(fig_ind)

    # Hide unused grid slots
    for j in range(i + 1, len(axes_grid)):
        axes_grid[j].set_visible(False)
        
    fig_grid.tight_layout()
    fig_grid.savefig(os.path.join(config.OUTPUT_DIR, "SHAP_Cutoff_Summary.png"))
    plt.close(fig_grid)
    
    # Save CSV results
    results_path = os.path.join(config.OUTPUT_DIR, "SHAP_Cutoff_Results.csv")
    pd.DataFrame(cutoff_results).to_csv(results_path, index=False)
    print(f"  Cutoff results saved to: {results_path}")
    
    return cutoff_results



