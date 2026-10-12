# NHANES Periodontitis Analysis

[中文](README_CN.md) · [Interactive demo](https://dissein.github.io/NHANES-Periodontitis-Analysis/)

This study compares the value of additional laboratory measurements and model complexity for identifying current moderate-to-severe periodontitis. NHANES 2013–2014 is used for development and 2011–2012 for validation, with 6,940 adults in total.

## Findings

- The seven-variable questionnaire Logistic model achieved a survey-weighted cross-cycle AUC of 0.801. Additional laboratory measurements and more complex algorithms did not produce a consistent improvement.
- Age and smoking contributed relatively stable information. Self-reported smoking and cotinine carried partly overlapping information.
- AUC was approximately 0.646 among adults aged 65 or older. Predicted probabilities needed recalibration across cycles.

## Methods

The outcome follows the CDC/AAP 2012 surveillance definition. Five information panels (7, 9, 11, 19, and 47 variables) are compared using Logistic regression, EBM, and XGBoost. Development uses survey weights and PSU-grouped nested validation. Paired comparisons separate information gains within an algorithm from algorithm gains within a panel.

The main laboratory comparison is the 47-variable panel versus the 11-variable panel, with blood pressure considered separately. Interpretation includes SHAP, EBM response curves, and refitting after variable removal.

## Repository

| Directory | Contents |
|---|---|
| `src/` | Data construction, training, validation, and checks |
| `data/` | Variable dictionaries and public NHANES-derived analysis tables |
| `results/` | Model comparisons, uncertainty estimates, saved models, and predictions |
| `explainability_20261011/` | SHAP, EBM curves, and variable-removal analyses |
| `figures/` | Research figures |
| `docs/` | Manuscript, supplementary results, and English abstract |
| `web/demo_v1/` | Offline interactive demo and its source |

The previous XGBoost-only pipeline remains in Git history.

## Run

Use Python 3.12. Dependency versions are recorded in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
python src/run_all.py
```

The first run downloads official XPT files into `data/raw/`. Existing model results are cached. For complete retraining, use a fresh directory containing `src/` and the variable dictionaries from `data/`.

Interpretation analyses:

```bash
python explainability_20261011/src/explainability_extension.py
```

Open `web/demo_v1/index.html` in a browser to use the demo offline. Rebuild it from saved results:

```bash
python web/demo_v1/src/build_nhanes_demo_v1.py
```

The builder can embed the installed Noto Sans SC font on Windows; otherwise it uses the system font.

## Data

NHANES public-use data: [2011–2012](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2011), [2013–2014](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2013). Module URLs and file hashes are in `results/raw_sources.json`.

This cross-sectional analysis identifies existing disease rather than future disease risk. Code is released under the [MIT license](LICENSE).
