import pandas as pd
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_models():
    # Load test data
    test_df = pd.read_parquet('data/test.parquet')
    
    # Metadata prints
    print(f"Total test rows: {len(test_df)}")
    print(f"Test date range: {test_df['date'].min()} to {test_df['date'].max()}")
    
    TARGET_COLS = ['pm25_next_48h', 'o3_next_48h', 'no2_next_48h']
    pollutants = ['pm25', 'o3', 'no2']
    
    # Define FEATURE_COLS (same logic as training)
    ignore_cols = ['city', 'date'] + TARGET_COLS
    FEATURE_COLS = [c for c in test_df.columns if c not in ignore_cols]
    
    results = []
    
    for pollutant, target in zip(pollutants, TARGET_COLS):
        print(f"\n--- Evaluating {pollutant} ---")
        
        # Load model
        model_path = f'models/saved/{pollutant}_model.pkl'
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}")
            continue
            
        model = joblib.load(model_path)
        
        # Predictions
        X_test = test_df[FEATURE_COLS]
        y_test = test_df[target]
        y_pred = model.predict(X_test)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Persistence Baseline (predict tomorrow = today)
        # Using {pollutant}_lag1 as the baseline
        persistence_col = f"{pollutant}_lag1"
        if persistence_col in test_df.columns:
            y_persistence = test_df[persistence_col]
            p_rmse = np.sqrt(mean_squared_error(y_test, y_persistence))
            improvement = (p_rmse - rmse) / p_rmse * 100
        else:
            print(f"Warning: Persistence column {persistence_col} not found in test data.")
            p_rmse = np.nan
            improvement = np.nan
            
        print(f"Model RMSE: {rmse:.4f}")
        print(f"Model R²: {r2:.4f}")
        print(f"Persistence RMSE: {p_rmse:.4f}")
        if not np.isnan(improvement):
            print(f"Improvement: {improvement:.2f}%")
            
        results.append({
            'pollutant': pollutant,
            'model_rmse': rmse,
            'model_r2': r2,
            'persistence_rmse': p_rmse,
            'improvement_pct': improvement
        })
        
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('models/results.csv', index=False)
    print(f"\nResults saved to models/results.csv")

if __name__ == "__main__":
    evaluate_models()
