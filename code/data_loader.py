"""
Data loading and preprocessing utilities for NHANES dataset.
"""

import os
import pandas as pd
import numpy as np
import config


def load_and_preprocess_data():
    """
    Load the raw CSV data and perform NHANES-specific preprocessing.

    This includes:
    - Loading from config.DATA_PATH.
    - Robust conversion of specified numeric columns to categorical types.
    - Normalizing survey weights.

    Returns
    -------
    df : pd.DataFrame
        The full preprocessed dataframe.
    X : pd.DataFrame
        Feature matrix (categorical columns as 'category' type).
    y : np.ndarray
        Target vector (Periodontitis Status).
    w_norm : np.ndarray
        Normalized sample weights.
    feature_cols : list
        List of strings identifying the feature columns.
    """
    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {config.DATA_PATH}")

    df = pd.read_csv(config.DATA_PATH)
    print(f"[Data] Loaded raw data: {df.shape[0]} rows x {df.shape[1]} columns")

    # Identify feature columns by excluding non-feature constants
    feature_cols = [c for c in df.columns if c not in config.NON_FEATURE_COLS]

    # Robust Categorical Transformation
    # Handled carefully to avoid mixed-type issues (e.g., 1.0 vs "1")
    for col in config.CAT_FEATURES:
        # 1. Convert to numeric first to handle string representation of floats
        s_num = pd.to_numeric(df[col], errors='coerce')
        # 2. Map to string integers (e.g., 1.0 -> "1") to remove decimal points
        s_str = s_num.map(lambda x: str(int(x)) if pd.notnull(x) else np.nan)
        # 3. Convert to pandas categorical type
        df[col] = s_str.astype("category")
        print(f"  - Processed Categorical '{col}': {df[col].dtype}")

    X = df[feature_cols].copy()
    y = df[config.TARGET].values
    w = df[config.WEIGHT_COL].values

    # Normalize weights so they average to 1.0 (maintains sample size scale)
    w_norm = w / w.mean()

    print(f"[Data] Feature data types count: \n{X.dtypes.value_counts()}")

    return df, X, y, w_norm, feature_cols
