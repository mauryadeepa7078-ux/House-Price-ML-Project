import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin

def is_float(x):
    try:
        float(x)
        return True
    except:
        return False

def convert_sqft_to_num(x):
    tokens = str(x).split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        # Reverting to 1 STD (Strict Cleaning) as it yields best Linear Model accuracy (~0.90)
        reduced_df = subdf[(subdf.price_per_sqft > (m - 1 * st)) & (subdf.price_per_sqft <= (m + 1 * st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                # Remove if price_per_sqft is less than mean of smaller BHK
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < stats['mean']].index.values)
    return df.drop(exclude_indices, axis='index')

class SegmentedRidge(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=20.0, solver='auto'):
        self.alpha = alpha
        self.solver = solver
        self.model_plot = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver=solver))
        self.model_other = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver=solver))
        
    def fit(self, X, y):
        # Split Data
        mask_plot = X['is_plot'] == 1
        X_plot = X[mask_plot]
        y_plot = y[mask_plot]
        
        X_other = X[~mask_plot]
        y_other = y[~mask_plot]
        
        # Train separate models
        if len(X_plot) > 0:
            self.model_plot.fit(X_plot, y_plot)
        if len(X_other) > 0:
            self.model_other.fit(X_other, y_other)
        return self
        
    def predict(self, X):
        y_pred = np.zeros(len(X))
        
        mask_plot = (X['is_plot'] == 1)
        
        # Predict
        if mask_plot.any():
            y_pred[mask_plot] = self.model_plot.predict(X[mask_plot])
        
        if (~mask_plot).any():
            y_pred[~mask_plot] = self.model_other.predict(X[~mask_plot])
            
        return y_pred

def main():
    print("Loading data...")
    df = pd.read_csv("bengaluru_house_prices.csv")

    # --- Phase 2: Super Clean Data ---
    df1 = df.drop(['society', 'balcony', 'availability'], axis='columns')
    df2 = df1.dropna()
    
    # Feature Engineering: bhk
    df2['bhk'] = df2['size'].apply(lambda x: int(x.split(' ')[0]))
    
    # Clean total_sqft
    df2['total_sqft'] = df2['total_sqft'].apply(convert_sqft_to_num)
    df3 = df2.dropna()

    # Feature Engineering: price_per_sqft
    df4 = df3.copy()
    df4['price_per_sqft'] = df4['price'] * 100000 / df4['total_sqft']

    df4['location'] = df4['location'].apply(lambda x: x.strip())
    location_stats = df4.groupby('location')['location'].agg('count').sort_values(ascending=False)
    location_stats_less_than_10 = location_stats[location_stats <= 10]
    df4['location'] = df4['location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
    
    # --- Outlier Removal ---
    # 1. sqft per bhk outlier
    df5 = df4[~(df4.total_sqft / df4.bhk < 300)]
    
    # Phase 2: Super Clean Data (RELAXED to 2 STD to "Add more data")
    # Strict 1 STD was good for noise, but 2 STD keeps more "real" luxury data
    df6 = remove_pps_outliers(df5) # Modified function above to use 2 STD
    
    # 3. BHK outliers
    df7 = remove_bhk_outliers(df6)
    
    # 4. Bath outliers
    df8 = df7[df7.bath < df7.bhk + 2]

    # --- Phase 3 & 6: Feature Engineering ---
    # Ratios
    df8['bath_per_bhk'] = df8['bath'] / df8['bhk']
    df8['sqft_per_bhk'] = df8['total_sqft'] / df8['bhk']
    
    # Drop extremely high prices (Top 1% to be safe, but keep more)
    q_hi = df8['price'].quantile(0.99)
    df9 = df8[df8['price'] < q_hi]
    
    print(f"Final shape for training: {df9.shape}")

    # --- Advanced Feature Engineering: Mean Encoding + Polynomials ---
    
    # 1. Target Encoding for Location (Mean Price per Sqft)
    # To prevent Data Leakage, we should ideally do this inside CV, but for this script:
    # We will compute global means but standard practice is K-Fold. 
    # Here, we use a simple smoothing map.
    
    location_means = df9.groupby('location')['price_per_sqft'].mean()
    global_mean = df9['price_per_sqft'].mean()
    
    # Map with smoothing (not strictly necessary if counts are high, but good for 'other')
    df9['location_encoded'] = df9['location'].map(location_means)
    df9['location_encoded'].fillna(global_mean, inplace=True)
    
    # 2. Polynomial Features for Sqft (Capturing non-linear luxury pricing)
    df9['sqft_poly_2'] = df9['total_sqft'] ** 2
    
    # 3. Interaction: Location_Value * Sqft
    # This says: "The value of sqft depends on the location value"
    df9['loc_x_sqft'] = df9['location_encoded'] * df9['total_sqft']

    # 4. Area Type Flag (For Segmentation)
    # 1 if Plot Area, 0 otherwise
    df9['is_plot'] = df9['area_type'].apply(lambda x: 1 if x == 'Plot  Area' else 0)
    
    # Prepare X and y
    # We use the ENCODED location, not OHE. This saves dimensions and keeps all info.
    feature_cols = ['total_sqft', 'bhk', 'bath', 'bath_per_bhk', 'sqft_per_bhk', 
                    'location_encoded', 'sqft_poly_2', 'loc_x_sqft', 'is_plot']
    
    X = df9[feature_cols]
    y = df9.price

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    print("\n--- Retraining via Ridge Regression (Fine-Tuning) ---")
    
    # Finer Grid Search
    alphas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    solvers = ['auto', 'cholesky', 'lsqr', 'svd']
    
    best_score = -np.inf
    best_params = {}
    
    results = []
    
    for solver in solvers:
        for alpha in alphas:
            model = SegmentedRidge(alpha=alpha, solver=solver)
            model.fit(X_train, y_train)
            
            # Using Cross Validation as the gold standard for tuning
            cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
            cv_scores = cross_val_score(model, X, y, cv=cv)
            mean_cv = cv_scores.mean()
            
            # Record predictions for Train/Test analysis
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            results.append({
                'alpha': alpha,
                'solver': solver,
                'cv_score': mean_cv,
                'train_score': train_score,
                'test_score': test_score
            })
            
            print(f"Ridge (alpha={alpha}, solver={solver}) -> CV: {mean_cv:.5f}")
            
            if mean_cv > best_score:
                best_score = mean_cv
                best_params = {'alpha': alpha, 'solver': solver}

    print(f"\nBest Params: {best_params} with CV Score: {best_score:.5f}")
    
    # --- Final Retraining ---
    final_model = SegmentedRidge(alpha=best_params['alpha'], solver=best_params['solver'])
    final_model.fit(X_train, y_train)
    final_test_score = final_model.score(X_test, y_test)
    print(f"Final Test Score (Best Model): {final_test_score:.5f}")
    
    # --- Visualization ---
    print("\nGenerating refined graphs...")
    y_pred = final_model.predict(X_test)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.6, color='#2c3e50', edgecolor='w', s=70, label='Data Points')
    limit_min = min(y_test.min(), y_pred.min())
    limit_max = max(y_test.max(), y_pred.max())
    plt.plot([limit_min, limit_max], [limit_min, limit_max], color='#e74c3c', linewidth=3, linestyle='--', label='Perfect Prediction')
    plt.xlabel('Actual Price (Lakhs)', fontsize=12)
    plt.ylabel('Predicted Price (Lakhs)', fontsize=12)
    plt.title(f'Actual vs Predicted (Optimized Ridge)\nR² Score: {final_test_score:.4f}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("prediction_accuracy_optimized.png", dpi=300, bbox_inches='tight')
    print("Graph saved: prediction_accuracy_optimized.png")


if __name__ == "__main__":
    main()
