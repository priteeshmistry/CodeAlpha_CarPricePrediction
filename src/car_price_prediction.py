"""
=============================================================================
CodeAlpha Data Science Internship - Task 3: Car Price Prediction
Author: Priteesh Mistry
Description:
    End-to-end Machine Learning Regression pipeline to predict used car
    selling prices based on present showroom price, driven kilometers,
    vehicle age, fuel type, transmission, seller type, and owner count.
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Plot configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.titlesize'] = 14

def main():
    print("=" * 75)
    print("      CODEALPHA DATA SCIENCE INTERNSHIP: TASK 3 - CAR PRICE PREDICTION")
    print("=" * 75)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'car data.csv')
    plots_dir = os.path.join(base_dir, 'plots')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load dataset
    print(f"\n[1] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"    Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\n[2] First 5 records:")
    print(df.head())
    print("\n[3] Statistical Overview:")
    print(df.describe().T)

    # 2. Feature Engineering
    print("\n[4] Performing Feature Engineering...")
    # Reference year for vehicle age
    current_year = 2024
    df['Car_Age'] = current_year - df['Year']
    print(f"    Created 'Car_Age' feature (range: {df['Car_Age'].min()} to {df['Car_Age'].max()} years).")

    # Drop Car_Name and Year for regression modelling
    df_clean = df.drop(columns=['Car_Name', 'Year'])

    # 3. Exploratory Data Analysis
    print("\n[5] Generating Exploratory Data Analysis Visualizations...")

    # Plot 1: Target Variable Distribution (Selling_Price)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df_clean['Selling_Price'], kde=True, color='#2B5B84', ax=axes[0], bins=25)
    axes[0].set_title('Selling Price Distribution (in Lakhs INR)', weight='bold')
    axes[0].set_xlabel('Selling Price (Lakhs)')

    sns.histplot(np.log1p(df_clean['Selling_Price']), kde=True, color='#2ECC71', ax=axes[1], bins=25)
    axes[1].set_title('Log-Transformed Selling Price Distribution', weight='bold')
    axes[1].set_xlabel('Log(1 + Selling Price)')
    plt.tight_layout()
    p1 = os.path.join(plots_dir, '01_target_distribution.png')
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"    Saved: {p1}")

    # Plot 2: Correlation Heatmap
    plt.figure(figsize=(8, 6))
    num_cols = ['Selling_Price', 'Present_Price', 'Driven_kms', 'Owner', 'Car_Age']
    corr = df_clean[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap='RdBu_r', fmt='.3f', linewidths=1, square=True)
    plt.title('Correlation Heatmap of Numerical Features', weight='bold')
    plt.tight_layout()
    p2 = os.path.join(plots_dir, '02_correlation_heatmap.png')
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"    Saved: {p2}")

    # Plot 3: Present Price vs Selling Price
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x='Present_Price', y='Selling_Price', hue='Fuel_Type', 
                    style='Transmission', s=90, alpha=0.85, palette='Set1')
    plt.title('Selling Price vs Present Showroom Price (by Fuel & Transmission)', weight='bold')
    plt.xlabel('Present Ex-Showroom Price (Lakhs INR)')
    plt.ylabel('Selling Price (Lakhs INR)')
    plt.tight_layout()
    p3 = os.path.join(plots_dir, '03_present_price_vs_selling_price.png')
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"    Saved: {p3}")

    # Plot 4: Car Age vs Selling Price (Depreciation Trend)
    plt.figure(figsize=(9, 5.5))
    sns.regplot(data=df_clean, x='Car_Age', y='Selling_Price',
                scatter_kws={'color': '#E67E22', 'alpha': 0.7, 's': 50},
                line_kws={'color': '#C0392B', 'linewidth': 2.5})
    plt.title('Vehicle Age vs Selling Price (Depreciation Curve)', weight='bold')
    plt.xlabel('Car Age (Years)')
    plt.ylabel('Selling Price (Lakhs INR)')
    plt.tight_layout()
    p4 = os.path.join(plots_dir, '04_car_age_depreciation.png')
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"    Saved: {p4}")

    # Plot 5: Categorical Impact Boxplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.boxplot(data=df_clean, x='Fuel_Type', y='Selling_Price', ax=axes[0], palette='Pastel1', hue='Fuel_Type', legend=False)
    axes[0].set_title('Selling Price by Fuel Type', weight='bold')
    axes[0].set_ylabel('Selling Price (Lakhs)')

    sns.boxplot(data=df_clean, x='Selling_type', y='Selling_Price', ax=axes[1], palette='Pastel2', hue='Selling_type', legend=False)
    axes[1].set_title('Selling Price by Seller Type', weight='bold')
    axes[1].set_ylabel('')

    sns.boxplot(data=df_clean, x='Transmission', y='Selling_Price', ax=axes[2], palette='Set3', hue='Transmission', legend=False)
    axes[2].set_title('Selling Price by Transmission', weight='bold')
    axes[2].set_ylabel('')
    plt.tight_layout()
    p5 = os.path.join(plots_dir, '05_categorical_impact_boxplots.png')
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f"    Saved: {p5}")

    # 4. Data Preparation & Pipeline
    print("\n[6] Preparing Features and Splitting Dataset...")
    X = df_clean.drop(columns=['Selling_Price'])
    y = df_clean['Selling_Price']

    numeric_features = ['Present_Price', 'Driven_kms', 'Owner', 'Car_Age']
    categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"    Training instances: {X_train.shape[0]} | Testing instances: {X_test.shape[0]}")

    # 5. Model Benchmarking
    print("\n[7] Benchmarking Regression Models...")
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42),
        'XGBoost Regressor': XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
    }

    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_model_name = None
    best_r2 = -float('inf')
    best_pipeline = None

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        # 5-fold cross-validation on training data
        cv_r2_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='r2')
        cv_mae_scores = -cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')

        # Fit on full train set
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        results.append({
            'Model': name,
            'CV Mean R2': cv_r2_scores.mean(),
            'CV Std R2': cv_r2_scores.std(),
            'CV Mean MAE': cv_mae_scores.mean(),
            'Test R2 Score': r2,
            'Test MAE (Lakhs)': mae,
            'Test RMSE (Lakhs)': rmse
        })

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline

    results_df = pd.DataFrame(results).sort_values(by='Test R2 Score', ascending=False)
    print("\n" + "=" * 90)
    print("                     REGRESSION MODEL BENCHMARK RESULTS")
    print("=" * 90)
    print(results_df.to_string(index=False))
    print("=" * 90)

    # Save metrics table
    metrics_csv = os.path.join(base_dir, 'car_price_model_comparison.csv')
    results_df.to_csv(metrics_csv, index=False)
    print(f"Saved performance metrics to: {metrics_csv}")

    # Plot 6: Model Comparison Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x_idx = np.arange(len(results_df))
    axes[0].barh(x_idx, results_df['Test R2 Score'], color='#3498DB', height=0.6)
    axes[0].set_yticks(x_idx)
    axes[0].set_yticklabels(results_df['Model'])
    axes[0].set_xlabel('Test R² Score')
    axes[0].set_title('Model Test R² Score Comparison', weight='bold')
    axes[0].set_xlim(0.7, 1.0)
    for i, v in enumerate(results_df['Test R2 Score']):
        axes[0].text(v + 0.005, i, f"{v:.4f}", va='center', weight='bold', fontsize=10)

    axes[1].barh(x_idx, results_df['Test RMSE (Lakhs)'], color='#E74C3C', height=0.6)
    axes[1].set_yticks(x_idx)
    axes[1].set_yticklabels(results_df['Model'])
    axes[1].set_xlabel('Test RMSE (Lakhs INR)')
    axes[1].set_title('Model Test RMSE Comparison (Lower is Better)', weight='bold')
    for i, v in enumerate(results_df['Test RMSE (Lakhs)']):
        axes[1].text(v + 0.02, i, f"{v:.3f}", va='center', weight='bold', fontsize=10)
    plt.tight_layout()
    p6 = os.path.join(plots_dir, '06_model_performance_comparison.png')
    plt.savefig(p6, dpi=300)
    plt.close()
    print(f"    Saved: {p6}")

    # 6. Best Model Evaluation
    print(f"\n[8] Best Model Selected: {best_model_name} (Test R² = {best_r2:.4f})")
    best_preds = best_pipeline.predict(X_test)

    # Plot 7: Actual vs Predicted Scatter Plot
    plt.figure(figsize=(7.5, 6.5))
    plt.scatter(y_test, best_preds, color='#2980B9', edgecolors='black', alpha=0.8, s=60, label='Predictions')
    # Ideal line
    min_val = min(y_test.min(), best_preds.min())
    max_val = max(y_test.max(), best_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit (y = x)')
    plt.title(f'Actual vs Predicted Selling Prices ({best_model_name})', weight='bold')
    plt.xlabel('Actual Selling Price (Lakhs INR)', weight='bold')
    plt.ylabel('Predicted Selling Price (Lakhs INR)', weight='bold')
    plt.annotate(f"R² = {r2_score(y_test, best_preds):.4f}\nMAE = {mean_absolute_error(y_test, best_preds):.2f} Lakhs\nRMSE = {np.sqrt(mean_squared_error(y_test, best_preds)):.2f} Lakhs",
                 xy=(0.05, 0.78), xycoords='axes fraction', fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))
    plt.legend(loc='lower right')
    plt.tight_layout()
    p7 = os.path.join(plots_dir, '07_actual_vs_predicted.png')
    plt.savefig(p7, dpi=300)
    plt.close()
    print(f"    Saved: {p7}")

    # Plot 8: Residual Distribution
    residuals = y_test - best_preds
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(residuals, kde=True, color='#8E44AD', ax=axes[0], bins=20)
    axes[0].set_title('Residuals Error Distribution', weight='bold')
    axes[0].set_xlabel('Residuals (Actual - Predicted, Lakhs)')

    axes[1].scatter(best_preds, residuals, color='#D35400', edgecolors='black', alpha=0.7)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1.5)
    axes[1].set_title('Residuals vs Predicted Values (Homoscedasticity Check)', weight='bold')
    axes[1].set_xlabel('Predicted Selling Price (Lakhs)')
    axes[1].set_ylabel('Residual (Lakhs)')
    plt.tight_layout()
    p8 = os.path.join(plots_dir, '08_residual_distribution.png')
    plt.savefig(p8, dpi=300)
    plt.close()
    print(f"    Saved: {p8}")

    # Plot 9: Feature Importances
    regressor = best_pipeline.named_steps['regressor']
    if hasattr(regressor, 'feature_importances_'):
        cat_encoder = best_pipeline.named_steps['preprocessor'].named_transformers_['cat']
        encoded_cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
        all_feature_names = numeric_features + encoded_cat_names

        importances = regressor.feature_importances_
        feat_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=True)

        plt.figure(figsize=(9, 5.5))
        plt.barh(feat_df['Feature'], feat_df['Importance'], color='#16A085', height=0.6)
        plt.title(f'Feature Importances ({best_model_name})', weight='bold')
        plt.xlabel('Relative Feature Importance')
        for i, val in enumerate(feat_df['Importance']):
            plt.text(val + 0.005, i, f"{val:.3f}", va='center', weight='bold', fontsize=9)
        plt.tight_layout()
        p9 = os.path.join(plots_dir, '09_feature_importances.png')
        plt.savefig(p9, dpi=300)
        plt.close()
        print(f"    Saved: {p9}")

    # 7. Save Best Model Pipeline
    best_model_file = os.path.join(models_dir, 'car_price_best_model.joblib')
    joblib.dump(best_pipeline, best_model_file)
    print(f"\n[9] Saved trained model pipeline to: {best_model_file}")

    # 8. Sample Predictions on Hypothetical Cars
    print("\n[10] Real-World Inference Test on 3 Vehicles:")
    sample_cars = pd.DataFrame([
        {
            'Present_Price': 8.50,
            'Driven_kms': 35000,
            'Fuel_Type': 'Petrol',
            'Selling_type': 'Dealer',
            'Transmission': 'Manual',
            'Owner': 0,
            'Car_Age': 6
        },
        {
            'Present_Price': 32.00,
            'Driven_kms': 45000,
            'Fuel_Type': 'Diesel',
            'Selling_type': 'Dealer',
            'Transmission': 'Automatic',
            'Owner': 0,
            'Car_Age': 4
        },
        {
            'Present_Price': 0.75,
            'Driven_kms': 18000,
            'Fuel_Type': 'Petrol',
            'Selling_type': 'Individual',
            'Transmission': 'Manual',
            'Owner': 0,
            'Car_Age': 8
        }
    ])

    sample_preds = best_pipeline.predict(sample_cars)
    descriptions = [
        "Mid-Range Sedan (e.g. Maruti Ciaz / Honda City Petrol)",
        "Luxury SUV (e.g. Toyota Fortuner Diesel Automatic)",
        "Budget Commuter Two-Wheeler / Bike (Petrol Manual)"
    ]

    for idx, row in sample_cars.iterrows():
        print(f"\n    Vehicle {idx+1}: {descriptions[idx]}")
        print(f"      Inputs: Present_Price = Rs {row['Present_Price']} Lakhs, Driven = {row['Driven_kms']:,} km, Age = {row['Car_Age']} yrs, Fuel = {row['Fuel_Type']}, Trans = {row['Transmission']}")
        print(f"      --> PREDICTED SELLING PRICE: Rs {sample_preds[idx]:.2f} Lakhs")

    print("\n" + "=" * 75)
    print(" TASK 3: CAR PRICE PREDICTION COMPLETED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == '__main__':
    main()
