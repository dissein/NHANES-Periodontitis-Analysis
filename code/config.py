"""
Configuration settings for NHANES Periodontitis Analysis Pipeline.

This module contains:
- Directory paths
- Random seeds and hyperparameter search settings
- Feature definitions (target, weights, IDs)
- Categorical mappings and readable labels for NHANES codes
- Academic plotting styles
"""

import os
import matplotlib.pyplot as plt

# --- Directory Paths ---
# Use the directory containing the 'code' folder as the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = BASE_DIR
DATA_PATH = os.path.join(OUTPUT_DIR, "data.csv")

# --- Random Seed ---
SEED = 42

# --- Model & Search Settings ---
SHAP_TOP_N = 15      # Top N features to visualize in summary plots
TUNER_TRIALS = 10000  # Number of Optuna trials for hyperparameter search

# --- Column Definitions ---
TARGET = "Periodontitis_Status"
WEIGHT_COL = "WTMEC2YR"
ID_COLS = ["SDMVPSU", "SDMVSTRA", "SEQN"]
NON_FEATURE_COLS = [TARGET, WEIGHT_COL] + ID_COLS

# --- Categorical Mapping ---
# Dictionary mapping NHANES integer codes to human-readable strings
CAT_FEATURES = {
    "RIAGENDR": {1: "Male", 2: "Female"},
    "RIDRETH3": {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
                 4: "Non-Hispanic Black", 6: "Non-Hispanic Asian", 7: "Other Race - Including Multi-Racial"},
    "DMDEDUC2": {1: "Less than 9th grade", 2: "9-11th grade", 3: "High school graduate/GED or equivalent",
                 4: "Some college or AA degree", 5: "College graduate or above"},
}

# --- Clinical Labels for NHANES Feature Codes ---
FEATURE_LABELS = {
    "RIAGENDR": "Gender",
    "RIDRETH3": "Race",
    "DMDEDUC2": "Education Level",
    "RIDAGEYR": "Age",
    "BMXBMI": "Body Mass Index (kg/m²)",
    "LBXGLU": "Fasting Glucose (mg/dL)",
    "LBDHDD": "Direct HDL-Cholesterol (mg/dL)",
    "LBXTR": "Triglyceride (mg/dL)",
    "LBXTC": "Total Cholesterol (mg/dL)",
    "LBXSGL": "Glucose, refrigerated serum (mg/dL)",
    "INDFMPIR": "Ratio of family income to poverty",
    "LBXEOPCT": "Eosinophils percent (%)",
    "LBXGH": "Glycohemoglobin (%)",
    "LBXHGB": "Hemoglobin (g/dL)",
    "LBXLYPCT": "Lymphocyte percent (%)",
    "LBXMCVSI": "Mean cell volume (fL)",
    "LBXMOPCT": "Monocyte percent (%)",
    "LBXMPSI": "Mean platelet volume (fL)",
    "LBXNEPCT": "Segmented neutrophils percent (%)",
    "LBXPLTSI": "Platelet count (1000 cells/uL)",
    "LBXRDW": "Red cell distribution width (%)",
    "LBXSAL": "Albumin (g/dL)",
    "LBXSAPSI": "Alkaline phosphatase (IU/L)",
    "LBXSASSI": "Aspartate aminotransferase AST (U/L)",
    "LBXSATSI": "Alanine aminotransferase ALT (U/L)",
    "LBXSBU": "Blood urea nitrogen (mg/dL)",
    "LBXSC3SI": "Bicarbonate (mmol/L)",
    "LBXSCA": "Total calcium (mg/dL)",
    "LBXSCK": "Creatine Phosphokinase(CPK) (IU/L)",
    "LBXSCLSI": "Chloride (mmol/L)",
    "LBXSCR": "Creatinine (mg/dL)",
    "LBXSGB": "Globulin (g/dL)",
    "LBXSGTSI": "Gamma glutamyl transferase (U/L)",
    "LBXSIR": "Iron, refrigerated serum (ug/dL)",
    "LBXSKSI": "Potassium (mmol/L)",
    "LBXSLDSI": "Lactate dehydrogenase (U/L)",
    "LBXSNASI": "Sodium (mmol/L)",
    "LBXSPH": "Phosphorus (mg/dL)",
    "LBXSTB": "Total bilirubin (mg/dL)",
    "LBXSTR": "Triglycerides, refrigerated (mg/dL)",
    "LBXSUA": "Uric acid (mg/dL)",
    "LBXWBCSI": "White blood cell count (1000 cells/uL)",
    "UACR": "Urinary Albumin-to-Creatinine Ratio (mg/g)",
    "LBDFOTSI": "Serum total folate (nmol/L)",
    "BPXSY1": "Systolic: Blood pres (1st rdg) mm Hg",
    "BPXDI1": "Diastolic: Blood pres (1st rdg) mm Hg",
    "LBXCOT": "Cotinine, Serum (ng/mL)",
    "LBXVIDMS": "25OHD2+25OHD3 (nmol/L)",
    "BMXWAIST": "Waist Circumference (cm)",
    "WTMEC2YR": "Full sample 2 year MEC exam weight"
}


def set_academic_style():
    """Configure matplotlib parameters for high-resolution academic publishing."""
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
