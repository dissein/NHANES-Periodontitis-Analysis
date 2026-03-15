import os
import sys
import pandas as pd
import numpy as np

# Ensure this script can find the config and other modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import config
import data_loader
import model_trainer

def run_cv_only():
    # Load data
    df, X, y, w_norm, feature_cols = data_loader.load_and_preprocess_data()
    feature_types = ["c" if c in config.CAT_FEATURES else "q" for c in feature_cols]
    
    # Path to optimal hyperparameters (expected in the root directory)
    param_path = os.path.join(BASE_DIR, "Optimal_Hyperparameters.csv")
    
    if not os.path.exists(param_path):
        print(f"[Error] Could not find: {param_path}")
        return

    # Load optimal params
    params_df = pd.read_csv(param_path)
    params = params_df.iloc[0].to_dict()
    if "best_n_estimators" in params:
        del params["best_n_estimators"]
        
    print(f"[Script] Running 5-fold CV with parameters from {param_path}...")
    model_trainer.train_and_eval_oof(X, y, w_norm, params, feature_types)
    print(f"[Script] Done. Results saved to: {config.OUTPUT_DIR}")

if __name__ == "__main__":
    run_cv_only()
