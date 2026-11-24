# Insurance Price Detection

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)

A small machine-learning project that predicts individual medical insurance charges from demographic and health-related attributes. The project loads the `insurance.csv` dataset, encodes categorical values, divides the observations into training and test sets, trains a regression model, and reports the coefficient of determination ($R^2$) for both sets.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Feature Reference](#feature-reference)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Model and Evaluation](#model-and-evaluation)
- [Observed Results](#observed-results)
- [Requirements](#requirements)
- [Installation and Setup](#installation-and-setup)
- [Running the Project](#running-the-project)
- [Configuration and Customization](#configuration-and-customization)
- [Source Code Walkthrough](#source-code-walkthrough)
- [Limitations and Known Considerations](#limitations-and-known-considerations)
- [Troubleshooting](#troubleshooting)
- [Suggested Improvements](#suggested-improvements)
- [License](#license)

## Project Overview

The primary executable is `main.py`. It demonstrates the following workflow:

1. Read medical insurance records from a CSV file.
2. Convert the text categories `sex`, `smoker`, and `region` into numeric values.
3. Separate `charges`, the prediction target, from the explanatory features.
4. Split the data into a 90% training set and a 10% test set.
5. Train a Lasso regression model.
6. Produce predictions for training and test observations.
7. Print the training and test $R^2$ scores.

This project is intended for learning supervised regression, basic data preprocessing, train/test evaluation, and scikit-learn workflows. It does not expose a web interface or command-line prediction interface, and it does not save a trained model for later reuse.

## Repository Structure

| Path | Description |
|---|---|
| [`main.py`](main.py) | Main Python script containing data loading, preprocessing, model training, prediction, and evaluation. |
| [`insurance.csv`](insurance.csv) | Tabular dataset containing 1,338 insurance records and 7 columns. |
| [`requirements.txt`](requirements.txt) | Pinned Python package versions used by the project environment. |
| [`README.md`](README.md) | Project documentation. |
| [`.venv/`](.venv/) | Optional local Python virtual environment; it is not required if dependencies are installed elsewhere. |

## Dataset

The dataset contains **1,338 records** and **7 columns**:

- 6 input features
- 1 target/output column

| Column | Type | Description |
|---|---|---|
| `age` | Integer | Age of the insured person, from 18 to 64. |
| `sex` | Categorical text | `male` or `female`. |
| `bmi` | Float | Body mass index of the insured person. |
| `children` | Integer | Number of dependents, from 0 to 5. |
| `smoker` | Categorical text | Whether the person is a smoker: `yes` or `no`. |
| `region` | Categorical text | Residential US region: northeast, northwest, southwest, or southeast. |
| `charges` | Float | Medical insurance charges in US dollars. |

### Dataset Summary

| Statistic | age | bmi | children | charges |
|---|---:|---:|---:|---:|
| Count | 1,338 | 1,338 | 1,338 | 1,338 |
| Mean | 39.207025 | 30.663397 | 1.094918 | 13,270.422265 |
| Standard deviation | 14.049960 | 6.098187 | 1.205493 | 12,110.011237 |
| Minimum | 18 | 15.960000 | 0 | 1,121.873900 |
| 25th percentile | 27 | 26.296250 | 0 | 4,740.287150 |
| Median | 39 | 30.400000 | 1 | 9,382.033000 |
| 75th percentile | 51 | 34.693750 | 2 | 16,639.912515 |
| Maximum | 64 | 53.130000 | 5 | 63,770.428010 |

### Categorical Distribution

| Column | Categories and frequencies |
|---|---|
| `sex` | male: 676; female: 662 |
| `smoker` | no: 1,064; yes: 274 |
| `region` | southeast: 364; southwest: 325; northwest: 325; northeast: 324 |

The dataset has no missing values in any of its columns. It contains one duplicated row.

## Feature Reference

The script replaces original text values with integer identifiers:

### Sex Encoding

| Original value | Encoded value |
|---|---:|
| `male` | 0 |
| `female` | 1 |

### Smoker Encoding

| Original value | Encoded value |
|---|---:|
| `no` | 0 |
| `yes` | 1 |

### Region Encoding

| Original value | Encoded value |
|---|---:|
| `northeast` | 0 |
| `northwest` | 1 |
| `southwest` | 2 |
| `southeast` | 3 |

These mappings are performed by in-place calls to `pandas.Series.map()`. Unknown or missing category values would become `NaN` and require explicit handling before model fitting.

> Important: the region values are ordinal integers even though the regions are not ordered categories. The model may interpret numeric distance between region codes. One-hot encoding would represent region membership more appropriately.

## Machine Learning Pipeline

### 1. Data Loading

`main.py` reads the dataset with:

```python
dataframe = pd.read_csv("D:\\insurance_price_detection\\insurance.csv")
```

This absolute Windows path is currently required on the machine for which the script was written. It prevents the script from locating the CSV when the project is moved to another directory or is run on Linux or macOS.

### 2. Categorical Encoding

The script maps `sex`, `smoker`, and `region` into numeric representations so scikit-learn can accept the feature matrix.

### 3. Feature and Target Separation

```python
x = dataframe.drop(["charges"], axis=1)
y = dataframe["charges"]
```

- `x` contains `age`, `sex`, `bmi`, `children`, `smoker`, and `region`.
- `y` contains the continuous `charges` target.

### 4. Data Split

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.1,
    random_state=2
)
```

| Dataset | Size | Percentage |
|---|---:|---:|
| Training | 1,204 | 90% |
| Test | 134 | 10% |

The fixed `random_state=2` makes the sample split reproducible across runs.

### 5. Model Training

The active model is Lasso regression:

```python
model = Lasso()
model.fit(x_train, y_train)
```

Lasso (Least Absolute Shrinkage and Selection Operator) is a linear model that uses L1 regularization. The default scikit-learn configuration uses `alpha=1.0`. Lasso can shrink coefficients and may set some of them exactly to zero, making it useful for regularized and sparse linear modeling.

Linear regression is also imported and remains available in a commented-out line:

```python
# model = LinearRegression()
```

### 6. Prediction

The script generates predictions for both partitions:

```python
train_data_score = model.predict(x_train)
test_data_score = model.predict(x_test)
```

### 7. Evaluation

Both prediction sets are evaluated with the R-squared metric:

```python
metrics.r2_score(y_train, train_data_score)
metrics.r2_score(y_test, test_data_score)
```

## Model and Evaluation

R-squared, or $R^2$, is the proportion of target variance explained by the model:

$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

- A value close to 1 indicates stronger explanatory performance.
- A value of 0 indicates that the model explains no more variance than predicting the target mean.
- Negative values indicate performance worse than predicting the target mean.
- $R^2$ does not indicate the average dollar error; use metrics such as MAE or RMSE for absolute error.

## Observed Results

With the active Lasso configuration and `random_state=2`, the script produced:

| Dataset | R-squared |
|---|---:|
| Training | 0.7538541105899441 |
| Test | 0.7059663266727683 |

The test score is approximately **70.6%** of the target variance. The training score is higher by about **0.048**, which can indicate mild overfitting, although R-squared alone is not sufficient to determine model quality.

These scores were produced with the local virtual environment and pinned package versions. Package-version changes can alter exact floating-point values, though a large regression model update may also alter the result.

## Requirements

The project uses Python and the packages pinned in `requirements.txt`:

| Package | Purpose |
|---|---|
| pandas | CSV loading and tabular data manipulation |
| NumPy | Numerical operations and array support |
| scikit-learn | Train/test splitting, regression models, and metrics |
| Matplotlib | Plotting support |
| Seaborn | Statistical visualization support |
| SciPy | Numerical dependency used by the scientific stack |
| joblib, threadpoolctl | Model execution and parallelism support |
| cloudpickle | Serialization support |
| Supporting packages | Plotting, font, date, parsing, imaging, and packaging dependencies |

The pinned set includes versions that are specific to Python 3.13 in the current environment. Newer or older Python releases may require compatible versions of these packages.

## Installation and Setup

### Option 1: Create a Virtual Environment with PowerShell

From the repository root:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, allow scripts for the current process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

To deactivate the environment:

```powershell
deactivate
```

### Option 2: Run with a Virtual Environment on Windows

The repository already contains `.venv\Scripts\python.exe`. After cloning or copying the project, create a fresh environment instead of relying on a machine-specific environment:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Option 3: Install Globally

A virtual environment is recommended, but package installation is also possible globally:

```powershell
python -m pip install -r requirements.txt
```

## Running the Project

Before running, ensure `insurance.csv` exists at the absolute path specified in `main.py`, or update the path.

### Activate and Run

```powershell
cd D:\insurance_price_detection
.venv\Scripts\Activate.ps1
python main.py
```

### Run Directly from the Virtual Environment

```powershell
D:\insurance_price_detection\.venv\Scripts\python.exe D:\insurance_price_detection\main.py
```

### Expected Output

The script prints two decimal values, in this order:

```text
0.7538541105899441
0.7059663266727683
```

The first value is training $R^2$. The second value is test $R^2$.

## Configuration and Customization

### Change the CSV Location

Replace the absolute path in `main.py` with a location-relative path:

```python
from pathlib import Path

dataset_path = Path(__file__).resolve().parent / "insurance.csv"
dataframe = pd.read_csv(dataset_path)
```

This is portable across Windows, Linux, and macOS.

### Switch to Linear Regression

Uncomment the LinearRegression assignment and comment out the Lasso assignment:

```python
model = LinearRegression()
# model = Lasso()
```

### Change the Test-Set Size

Adjust `test_size` to another fraction:

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=2
)
```

Changing this to `0.2` reserves 20% of records for testing.

### Tune Lasso Regularization

Pass an alternate regularization strength:

```python
model = Lasso(alpha=0.1)
```

Smaller positive `alpha` values reduce regularization; larger values increase it.

### Add Additional Evaluation Metrics

Useful metrics include mean absolute error and root mean squared error:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

test_predictions = model.predict(x_test)
print("MAE:", mean_absolute_error(y_test, test_predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, test_predictions)))
```

### Inspect Dataset Details

Several commented diagnostic statements are already present in `main.py`, including:

- `dataframe.head()`
- `dataframe.shape`
- `dataframe.describe()`
- `dataframe.info()`
- Missing-value counts
- Categorical value counts
- Feature missing-value counts

Uncomment these statements during exploratory analysis.

## Source Code Walkthrough

| Code section | Behavior |
|---|---|
| Import statements | Loads Pandas, NumPy, plotting libraries, scikit-learn metrics, split utilities, and regression models. |
| `pd.read_csv(...)` | Reads the CSV into a Pandas DataFrame. |
| `sex.map(...)` | Maps male/female to 0/1. |
| `smoker.map(...)` | Maps no/yes to 0/1. |
| `region.map(...)` | Maps four region names to integer codes. |
| `drop(["charges"], axis=1)` | Removes the target column from the feature matrix. |
| `dataframe["charges"]` | Selects the target series. |
| `train_test_split(...)` | Randomizes and separates the rows into train and test partitions. |
| `Lasso()` | Creates the active regularized linear model. |
| `fit(...)` | Learns coefficients from training data. |
| `predict(...)` | Generates estimated charges for training and test data. |
| `r2_score(...)` | Measures the proportion of variance explained. |

## Limitations and Known Considerations

### Absolute Dataset Path

The script uses `D:\insurance_price_detection\insurance.csv`. It will fail if:

- the project is cloned to a different drive or folder;
- the repository name or directory hierarchy changes;
- the script is run on Linux or macOS;
- `insurance.csv` is renamed or moved.

Use `Path(__file__)` or a configurable path for portability.

### No Cross-Validation

The script evaluates one deterministic split only. It does not use k-fold cross-validation or repeated splits.

### Small Test Set

Only 134 records, 10% of the dataset, are reserved for testing. Test metrics can vary substantially with another split or random seed.

### Ordinal Region Encoding

Regions are nominal categories but are encoded as integer values 0 through 3. This imposes a nonexistent numerical relationship among regions.

### Potential Binary Encoding Issues

One-hot encoding is also not necessarily required for `sex` or `smoker` because each has only two categories, but using explicit named mappings and preserving preprocessing for future records would make deployment safer.

### Duplicate Record

The source data contains one duplicated row. The script neither identifies nor removes it.

### No Missing-Value Strategy

There are no missing values in the current CSV, but the script has no production-style imputation or validation if the input data later contains null values.

### No Input Validation

No range, type, or category validation is applied to future input values.

### No Train-Test Leakage Protection

Preprocessing is performed on the full dataset before splitting. For the mappings used here, this does not use target values and does not change the row values, but a scalable pipeline should encapsulate preprocessing inside a scikit-learn `Pipeline` fitted only on training data.

### No Model Persistence

The fitted model is not exported as a `.joblib` or `.pkl` file. Predictions are not exposed outside the script.

### No CLI or Interactive Input

The script cannot currently accept custom patient values from command-line arguments or a user interface.

### Limited Metrics

It reports R-squared only. It does not report MAE, RMSE, residual distributions, coefficient details, or prediction intervals.

### Imports Not Used at Runtime

`numpy`, `matplotlib.pyplot`, `seaborn`, and `LinearRegression` are available in the script or dependencies, but are not required for the active Lasso execution. The plotting imports do not create charts because no plotting statements are active.

## Troubleshooting

### `FileNotFoundError`

The current dataset path is likely not valid for the system or project location.

- Confirm `insurance.csv` is in the repository.
- Use a path relative to `main.py`, as shown in [Change the CSV Location](#change-the-csv-location).

### `ModuleNotFoundError`

A dependency may not be installed in the active Python environment.

```powershell
python -m pip install -r requirements.txt
```

Confirm the environment:

```powershell
python -m pip --version
python -c "import pandas, sklearn, matplotlib, seaborn; print('Dependencies OK')"
```

### Activation Script Is Blocked

Set the execution policy only for the current PowerShell process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### Package Version Conflicts

Create a fresh virtual environment rather than mixing packages from other projects:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Unexpected or Non-Reproducible Scores

- Confirm `random_state=2` is unchanged.
- Confirm the `insurance.csv` file has not been modified.
- Confirm `Lasso()` is still the active model.
- Compare package versions with `requirements.txt`.

## Suggested Improvements

1. Replace the absolute CSV path with `Path(__file__).resolve().parent / "insurance.csv"`.
2. Add one-hot encoding for `region`, preferably inside a scikit-learn pipeline.
3. Add cross-validation and hyperparameter tuning for Lasso `alpha`.
4. Compare Lasso with linear, Ridge, polynomial, and tree-based regression models.
5. Add MAE and RMSE reporting.
6. Add residual, actual-versus-predicted, and coefficient visualizations.
7. Add duplicate detection and removal if duplicates are confirmed to be accidental.
8. Add missing-value validation or imputing.
9. Use a `Pipeline` and `ColumnTransformer` to keep preprocessing reproducible.
10. Save the fitted pipeline with `joblib`.
11. Add a command-line interface or user-facing form for custom predictions.
12. Split exploratory analysis, preprocessing, training, and evaluation into functions.
13. Add unit tests for encoding, data shape, split behavior, and metric output.
14. Add dependency and data validation at startup.
15. Add a dataset source/license attribution if the project is published or redistributed.

## License

No license file is currently included in this repository. If other people need to use, modify, or distribute this project, add an appropriate open-source license or a clear usage statement first.
