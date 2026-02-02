# Bangalore House Price Prediction (90% Accuracy)

## Overview
This project implements a high-accuracy Machine Learning model to predict house prices in Bangalore, India. 
Using a **Segmented Linear Regression** approach (separating Plots from Apartments) and advanced feature engineering, we achieved an **R² Score of ~0.90**, pushing the limits of what linear models can achieve on this dataset.

## Key Results
- **Test Set Accuracy**: **89.8%** (0.89785)
- **Cross Validation**: **89.3%**
- **Model Type**: Segmented Ridge Regression (Plot vs Apartment)

## Files
- `model_training.py`: Main script for data cleaning, training, and evaluation.
- `bengaluru_house_prices.csv`: The dataset (from Kaggle).
- `prediction_accuracy_optimized.png`: Visual graph of Actual vs Predicted values.
- `train_vs_test_analysis.png`: Bias-Variance analysis graph.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Training Script
```bash
python model_training.py
```
This will:
- Load and clean the data (removing outliers).
- Engineer features (Target Encoding, Polynomials).
- Train the Segmented Ridge Model.
- Print the accuracy scores.
- Generate performance graphs.

## Methodology Highlights
1.  **Segmented Modeling**: We found that "Plot Areas" and "Apartments" follow different pricing laws. Training separate models for each increased accuracy significantly.
2.  **Strict Data Cleaning**: We prioritized data quality over quantity, using strict 1-STD outlier removal to eliminate noise.

For full technical details, see [DOCUMENTATION.md](DOCUMENTATION.md).
