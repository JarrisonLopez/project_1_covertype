# Forest Cover Type Classification

This project predicts forest cover type from cartographic variables using supervised Machine Learning classifiers.

## Authors

- Jarrison Andres Lopez Roldan
- Juliana Andrea Betancur Blandon

## Project Overview

The repository compares **Logistic Regression** and **Support Vector Machines (SVM)** for predicting `Cover_Type` from cartographic predictors in the UCI Covertype dataset.

The task is a **seven-class multiclass classification** problem: each 30 × 30 meter observation belongs to exactly one forest cover class.

## Dataset

**Dataset:** Covertype — UCI Machine Learning Repository  
**URL:** https://archive.ics.uci.edu/dataset/31/covertype

Key characteristics of the dataset used in this project:

- **581,012** observations
- **54** predictor variables
- **7** forest cover classes (`Cover_Type` ∈ {1,…,7})
- Cartographic predictors (quantitative terrain/distance/hillshade measures plus wilderness-area and soil-type indicators)
- **No missing values** in the prepared CSV used here

**Data usage in the workflow:**

- The full dataset (**581,012** rows) is used for description, verification, and Exploratory Data Analysis.
- Model training/evaluation uses a reproducible **stratified working sample of 25,000 observations** created in memory inside the notebook (`random_state=42`).

The deliverable file `data/covertype.csv` contains the **full** dataset with headers. It does **not** contain only 25,000 rows. The 25,000-row sample is generated inside the notebook for Logistic Regression / SVM evaluation because nonlinear SVM cross-validation and hyperparameter search on all 581,012 observations would be computationally expensive.

## Methods

- Exploratory Data Analysis on the full dataset (distributions, descriptive statistics, class imbalance, correlations, and written interpretation)
- Preprocessing with `ColumnTransformer` + `Pipeline`:
  - `StandardScaler` on the **10** quantitative predictors
  - **44** binary wilderness/soil indicators passed through unchanged
- Stratified **10-fold** cross-validation (`StratifiedKFold`, `shuffle=True`, `random_state=42`)
- **Logistic Regression** with a controlled hyperparameter grid
- **SVM (`SVC`)** with **linear**, **RBF**, and **polynomial** kernels and controlled hyperparameter grids
- Evaluation metrics:
  - Macro-Precision
  - Macro-Recall
  - Macro-F1

**Primary model-selection metric:** Macro-F1.

## Results

Cross-validated results on the stratified 25,000-observation working sample:

| Model | Best configuration | Macro-Precision | Macro-Recall | Macro-F1 |
|-------|--------------------|----------------:|-------------:|---------:|
| Logistic Regression | `C=10`, `class_weight=None` | 0.5749 | 0.5042 | 0.5213 |
| SVM | `kernel=RBF`, `C=10`, `gamma=scale` | 0.7840 | 0.6692 | 0.7039 |

Under the experimental conditions used in this project, the **RBF SVM** was selected because it achieved the highest cross-validated Macro-F1.

Relative to Logistic Regression, SVM improved Macro-F1 by approximately **0.1827** absolute points (**≈ 35.04%** relative improvement). These figures describe observed differences under the shared protocol and are **not** statistical significance claims.

## Repository Structure

```text
project_1_covertype/
├── README.md
├── Proyecto_1_Covertype.ipynb
└── data/
    └── covertype.csv
```

## How to Run

1. Clone or download the repository.
2. Install the Python dependencies listed below (for example with `pip`).
3. Open `Proyecto_1_Covertype.ipynb` in Jupyter or VS Code.
4. Run the notebook cells from top to bottom.

Notes:

- The notebook loads `data/covertype.csv` with a **relative path**.
- SVM grid evaluation is computationally expensive. In the development environment used for this project, the official SVM hyperparameter grid took approximately **48 minutes**. Runtime will vary by machine.

## Main Dependencies

Libraries imported/used by the notebook:

- Python 3
- pandas
- numpy
- matplotlib
- scikit-learn

## Limitations

- Model comparison used a stratified **25,000**-row working sample rather than all 581,012 rows, for nonlinear SVM computational feasibility.
- Hyperparameter grids were controlled for runtime and are not exhaustive.
- The same 10-fold CV results were used for hyperparameter comparison and reported performance (without nested cross-validation; may include some selection optimism vs nested CV).
- Strong class imbalance remains.
- Only Logistic Regression and SVM were compared.

## Dataset Citation

Blackard, J. (1998). Covertype [Dataset]. UCI Machine Learning Repository.  
https://doi.org/10.24432/C50K5N

