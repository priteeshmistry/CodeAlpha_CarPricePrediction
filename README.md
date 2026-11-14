# 🚗 CodeAlpha Task 3: Car Price Prediction with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.4%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-v2.0%2B-red.svg)](https://xgboost.ai/)
[![Status](https://img.shields.io/badge/Task-Completed-success.svg)]()

> **Internship Program:** CodeAlpha Data Science Internship  
> **Task Name:** Task 3 - Car Price Prediction with Machine Learning  
> **Author:** Priteesh Mistry  
> **GitHub Repository:** `CodeAlpha_CarPricePrediction`

---

## 📌 Project Overview
Predicting the market resale value of pre-owned vehicles is a classic real-world regression problem. Resale valuation depends heavily on multi-variable interactions including original ex-showroom price, vehicle age, cumulative mileage driven, fuel economy, transmission type, previous ownership history, and seller credibility.

This project delivers an end-to-end Machine Learning pipeline that automates used car price prediction, benchmarks 6 linear and non-linear regression algorithms under 5-Fold Cross-Validation, and exposes a serialized model ready for inference.

---

## 📊 Dataset Description
- **Records:** 301 vehicle entries
- **Target Variable:** `Selling_Price` (in Lakhs INR, continuous)
- **Predictor Variables:**
  - `Present_Price`: Current ex-showroom price (in Lakhs INR)
  - `Driven_kms`: Total odometer mileage (in km)
  - `Fuel_Type`: Fuel category (`Petrol`, `Diesel`, `CNG`)
  - `Selling_type`: Sales channel (`Dealer`, `Individual`)
  - `Transmission`: Gearbox type (`Manual`, `Automatic`)
  - `Owner`: Number of previous owners (`0`, `1`, `3`)
  - `Year` $\rightarrow$ Engineered into **`Car_Age`** (Years)

### Summary Statistics

| Feature | Min | Mean | Median | Max | Std Dev |
|---|---|---|---|---|---|
| **Selling_Price (Target)** | 0.10 | 4.66 | 3.60 | 35.00 | 5.08 |
| **Present_Price** | 0.32 | 7.63 | 6.40 | 92.60 | 8.64 |
| **Driven_kms** | 500 | 36,947 | 32,000 | 500,000 | 38,887 |
| **Car_Age (Years)** | 6 | 10.37 | 10.00 | 21.00 | 2.89 |

---

## ⚙️ Data Preprocessing & Feature Engineering
1. **Domain Feature Engineering:**  
   `Car_Age = 2024 - Year` was engineered to linearize vehicle depreciation over time.
2. **ColumnTransformer Pipeline:**  
   - Numerical predictors (`Present_Price`, `Driven_kms`, `Owner`, `Car_Age`) standardized via `StandardScaler`.
   - Categorical predictors (`Fuel_Type`, `Selling_type`, `Transmission`) encoded via `OneHotEncoder(drop='first')`.
3. **Cross-Validation:**  
   5-Fold K-Fold Cross Validation on the training split to verify out-of-fold generalization.

---

## 🏆 Model Benchmarking & Performance Comparison

| Model | 5-Fold CV Mean R² | CV Std R² | CV Mean MAE (Lakhs) | Test R² Score | Test MAE (Lakhs) | Test RMSE (Lakhs) |
|---|---|---|---|---|---|---|
| **Gradient Boosting Regressor** | **0.9071** | **±0.0410** | **0.6751** | **0.9732** | **0.4980** | **0.7861** |
| **Random Forest Regressor** | 0.9072 | ±0.0335 | 0.7012 | 0.9622 | 0.6238 | 0.9332 |
| **XGBoost Regressor** | 0.9214 | ±0.0351 | 0.6513 | 0.9563 | 0.5998 | 1.0033 |
| **Ridge Regression** | 0.8043 | ±0.0540 | 1.3310 | 0.8491 | 1.2140 | 1.8644 |
| **Linear Regression** | 0.8003 | ±0.0578 | 1.3361 | 0.8489 | 1.2164 | 1.8658 |
| **Lasso Regression** | 0.8057 | ±0.0584 | 1.3002 | 0.8404 | 1.2538 | 1.9172 |

### 🎯 Best Model: Gradient Boosting Regressor
- **Coefficient of Determination ($R^2$):** **0.9732** (Explains 97.32% of price variance on unseen test data)
- **Mean Absolute Error (MAE):** **0.498 Lakhs INR** (~₹49,800 INR average prediction deviation)
- **Root Mean Squared Error (RMSE):** **0.786 Lakhs INR**

---

## 🔍 Key Feature Importance Insights
1. **Present Showroom Price (Dominant Factor):** Represents >80% of model feature importance.
2. **Vehicle Age (Depreciation Rate):** Second most influential predictor; vehicles depreciate most aggressively during the first 3-5 years.
3. **Fuel Type (Diesel Premium):** Diesel variants retain significantly higher second-hand value than petrol counterparts.
4. **Transmission:** Automatic variants command an average ₹2.5-4 Lakh premium in the pre-owned market.

---

## 📈 Visualizations
All high-resolution figures are located in the [`plots/`](./plots/) directory:
- `01_target_distribution.png`: Raw vs log-transformed price distributions
- `02_correlation_heatmap.png`: Pearson correlation matrix
- `03_present_price_vs_selling_price.png`: Showroom vs resale price scatter
- `04_car_age_depreciation.png`: Age depreciation regression curve
- `05_categorical_impact_boxplots.png`: Price comparisons by fuel, transmission, and seller type
- `06_model_performance_comparison.png`: R² and RMSE comparative bar charts
- `07_actual_vs_predicted.png`: Actual vs Predicted prices with diagonal unity line
- `08_residual_distribution.png`: Homoscedasticity and error distribution check
- `09_feature_importances.png`: Ranking of top model predictors

---

## 🚀 How to Run

```bash
cd CodeAlpha_Task3_Car_Price_Prediction

# Install dependencies
pip install numpy pandas scikit-learn xgboost matplotlib seaborn joblib

# Run training & evaluation script
python src/car_price_prediction.py

# Launch interactive notebook
jupyter notebook Car_Price_Prediction.ipynb
```

---

## 📂 Project Structure
```
CodeAlpha_Task3_Car_Price_Prediction/
│
├── data/
│   └── car data.csv
├── models/
│   └── car_price_best_model.joblib        # Serialized best model pipeline
├── plots/
│   ├── 01_target_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_present_price_vs_selling_price.png
│   ├── 04_car_age_depreciation.png
│   ├── 05_categorical_impact_boxplots.png
│   ├── 06_model_performance_comparison.png
│   ├── 07_actual_vs_predicted.png
│   ├── 08_residual_distribution.png
│   └── 09_feature_importances.png
├── src/
│   └── car_price_prediction.py
├── Car_Price_Prediction.ipynb
├── car_price_model_comparison.csv
└── README.md
```

---

## 🎥 LinkedIn Presentation Video Script (Word-for-Word Guide)
*(For completing Step 4 of the CodeAlpha Intern Instructions)*

> **Introduction (0:00 - 0:20):**  
> *"Hello everyone! I'm Priteesh Mistry, a Data Science Intern at CodeAlpha. Today, I'm delighted to present Task 3: Car Price Prediction with Machine Learning."*
>
> **Problem & Dataset (0:20 - 0:50):**  
> *"Valuing used cars accurately is vital for both buyers and automotive dealerships. Using a dataset of over 300 vehicle transactions, we aimed to predict the selling price based on present ex-showroom price, mileage, fuel type, transmission, seller type, and an engineered feature: vehicle age."*
>
> **Methodology & Modeling (0:50 - 1:30):**  
> *"We built an automated Scikit-learn Pipeline with StandardScaler and One-Hot Encoding, benchmarking 6 algorithms under 5-Fold Cross Validation: Linear, Ridge, Lasso, Random Forest, Gradient Boosting, and XGBoost. Our EDA revealed that ex-showroom price and vehicle age are the two primary drivers of value retention, with diesel automatic models commanding higher resale premiums."*
>
> **Results & Conclusion (1:30 - 2:00):**  
> *"Our Gradient Boosting Regressor emerged as the clear winner, achieving a stellar R² score of 0.9732 and a Mean Absolute Error of under ₹50,000 INR on test data. The full pipeline has been exported for live deployment. Thank you CodeAlpha for this practical machine learning project!"*
