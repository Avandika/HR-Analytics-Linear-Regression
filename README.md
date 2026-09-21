# 📊 HR Analytics Turnover Prediction Using Linear Regression Baseline

## 📌 Project Overview
This repository contains an end-to-end data science notebook pipeline that applies a **Linear Regression Baseline Model** to a dataset of 19,158 records to evaluate candidate flight risks (`target`). The focus is on tracking continuous developmental indicators against binary behavioral states.

## ⚙️ Implemented Notebook Workflow Sequence
- **Load Data & Details:** Analyzed initial feature boundaries and row densities `(19158, 14)`.
- **Dimensional Column Drops:** Dropped unnecessary structural indicators (`enrollee_id`, `city`, etc.) to isolate numeric modeling matrices.
- **ETL Cleaning & String Splitting:** Cleaned character tags (`>`, `<`) from experience entries and resolved textual null conflicts.
- **Statistical Imputation Matrix:** Patched missing cells using median filling values to optimize line convergence boundaries safely.
- **Exploratory Analytics Charts:** Constructed Seaborn distribution bar plots and feature pairplot tracking grids.
- **Train-Test Split Partitioning:** Segmented a 70/30 randomized sample validation collection split.
- **Model Evaluation:** Calculated absolute slope vectors and baseline intercept parameters.

## 📈 Realized Regression Performance Metrics
- **Constant Intercept Flag:** `1.2804`
- **Mean Absolute Error (MAE):** `0.3119`
- **Root Mean Squared Error (RMSE):** `0.3960`
- **R² Fit Accuracy Score:** `0.1678 (16.78%)`

*Mathematical Note:* An R² variance coverage of **16.78%** is common and expected when trying to map a linear regression line across binary human behavioral trends. The true business value sits in the calculated feature weights, highlighting **`city_development_index`** as the strongest negative anchor trend coefficient (`-1.0718`).
