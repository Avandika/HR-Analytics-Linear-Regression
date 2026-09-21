# 📊 HR Analytics: Candidate Turnover Prediction Using Linear Regression Baseline

## 📌 Project Overview
This repository contains an end-to-end machine learning pipeline built inside a **Google Colab Notebook** (`HR_Analytics.ipynb`). The objective is to utilize a **Linear Regression Baseline Model** to estimate candidate flight risk (`target`) based on continuous developmental metrics and encoded employee profiles from a dataset of 19,158 records.

## ⚙️ Data Pipeline Architecture (Step-by-Step)
- **📦 Load Data & Details:** Analyzed initial feature boundaries and row densities `(19158, 14)` to assess the feature columns map.
- **🗑️ Dimensional Column Drops:** Dropped unnecessary structural indicators (`enrollee_id`, `city`, etc.) to isolate numeric modeling matrices.
- **🧼 ETL Cleaning & String Splitting:** Cleaned character tags (`>`, `<`) from experience entries and resolved textual null conflicts.
- **🔀 Dtype Changing:** Cast the raw experience strings into clean numeric float variables.
- **🩹 Missing Value Imputation Matrix:** Patched missing cells using median filling values to optimize line convergence boundaries safely for `city_development_index`, `training_hours`, and `experience`.
- **📈 Exploratory Analytics Charts:** Constructed Seaborn distribution bar plots for discrete categorical attributes and a complete feature grid `pairplot` matrix to identify outlier data skewness.
- **📥 Train-Test Split Partitioning:** Segmented a 70/30 randomized data split matrix configuration.
- **🧠 Model Construction:** Fitted a `LinearRegression()` model to calculate absolute slope vectors and baseline intercept parameters.

## 📈 Baseline Model Performance Metrics
Evaluating the model against the verification test partition yielded the following exact parameters:

- **Constant Intercept:** `1.2804`
- **Mean Absolute Error (MAE):** `0.3119`
- **Mean Absolute Percentage Error (MAPE):** `701056726024022.6`
- **Mean Squared Error (MSE):** `0.1568`
- **Root Mean Squared Error (RMSE):** `0.3960`
- **R² Fit Accuracy Score:** `0.1678 (16.78%)`

---

## 🔍 Technical Insights & Slope Interpretation
An R² variance coverage of **16.78%** is common and expected when trying to map a linear regression line across binary human behavioral trends. The true business value sits in the calculated feature weights (`model.coef_`), highlighting the features pulling hardest on the prediction score:

* **`city_development_index` (Weight: -1.0718):** Strongest negative coefficient. As the index of city development rises, the estimated value drops sharply, signaling it as a massive anchor point against corporate churn.
* **`major_discipline_Business Degree` (Weight: +0.1291):** Strongest positive coefficient pushing the baseline up.
* **`experience` (Weight: -0.0033):** Holds a slight negative coefficient, indicating that attrition probability decreases marginally with years of career history.

## 🛠️ Environment and Dependencies
To replicate the environment locally or back in Google Colab, ensure the following core libraries are installed:
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`
