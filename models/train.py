import pandas as pd
import xgboost as xgb
import optuna
import joblib
import os
from sklearn.metrics import mean_squared_error
import numpy as np

# Set random state for reproducibility
RANDOM_STATE = 42

def train_models():
    # Load data
    train_df = pd.read_parquet('data/train.parquet')
    val_df = pd.read_parquet('data/val.parquet')

    TARGET_COLS = ['pm2.5_next_48h', 'o3_next_48h', 'no2_next_48h']
    
    # Define FEATURE_COLS: exclude metadata, raw pollutants, and targets
    ignore_cols = set(['city_name', 'date', 'time', 'PM2.5', 'O3', 'NO2'] + 
                      TARGET_COLS + 
                      [c for c in train_df.columns if 'next' in c])
    FEATURE_COLS = [c for c in train_df.columns if c not in ignore_cols]
    
    print(f"Features: {FEATURE_COLS}")
    
    # Ensure save directory exists
    os.makedirs('models/saved', exist_ok=True)

    for target in TARGET_COLS:
        print(f"\n--- Tuning for {target} ---")
        
        X_train = train_df[FEATURE_COLS]
        y_train = train_df[target]
        X_val = val_df[FEATURE_COLS]
        y_val = val_df[target]

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                'tree_method': 'hist',
                'random_state': RANDOM_STATE,
                'n_jobs': -1
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            return rmse

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(objective, n_trials=40)
        
        print(f"Best trial RMSE: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")

        # Retrain on train + val combined
        print(f"Retraining {target} model on combined data...")
        X_combined = pd.concat([X_train, X_val])
        y_combined = pd.concat([y_train, y_val])
        
        best_params = study.best_params
        best_params['tree_method'] = 'hist'
        best_params['random_state'] = RANDOM_STATE
        best_params['n_jobs'] = -1
        
        final_model = xgb.XGBRegressor(**best_params)
        final_model.fit(X_combined, y_combined)
        
        # Save the model
        model_path = f"models/saved/{target.split('_')[0].replace('.','')}_model.pkl"
        joblib.dump(final_model, model_path)
        print(f"Saved model to {model_path}")

if __name__ == "__main__":
    train_models()
