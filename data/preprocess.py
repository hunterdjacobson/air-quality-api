import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime

def get_season(month: int) -> int:
    """Helper to map month to season (1: Winter, 2: Spring, 3: Summer, 4: Fall)."""
    if month in [12, 1, 2]:
        return 1
    elif month in [3, 4, 5]:
        return 2
    elif month in [6, 7, 8]:
        return 3
    else:
        return 4

def preprocess():
    print("Loading raw EPA data...")
    pollutants = ["PM2.5", "O3", "NO2"]
    epa_dfs = {}
    for p in pollutants:
        path = f"data/raw_{p}.parquet"
        if os.path.exists(path):
            df = pd.read_parquet(path)
            # Standardize column names for join
            # EPA data has 'date_local' and 'city_name'
            df['date'] = pd.to_datetime(df['date_local'])
            # Pollutant concentration column is usually 'arithmetic_mean'
            # We'll rename it to the pollutant name for clarity
            df = df.rename(columns={'arithmetic_mean': p})
            
            # Aggregate: EPA data has multiple monitors per city. Take the mean.
            df = df.groupby(['date', 'city_name'])[p].mean().reset_index()

            # Keep only necessary columns for the join
            epa_dfs[p] = df[['date', 'city_name', p]]
        else:
            print(f"Warning: {path} not found.")

    if not epa_dfs:
        print("Error: No EPA data found.")
        return

    # Merge EPA dataframes
    merged_epa = None
    for p, df in epa_dfs.items():
        if merged_epa is None:
            merged_epa = df
        else:
            merged_epa = pd.merge(merged_epa, df, on=['date', 'city_name'], how='inner')

    print("Loading weather data...")
    weather_files = glob.glob("data/weather_*.parquet")
    weather_dfs = []
    for f in weather_files:
        df = pd.read_parquet(f)
        df['date'] = pd.to_datetime(df['time'])
        weather_dfs.append(df)
    
    if not weather_dfs:
        print("Error: No weather data found.")
        return
        
    all_weather = pd.concat(weather_dfs, ignore_index=True)

    print("Joining EPA and Weather data...")
    # Join on city and date
    df = pd.merge(merged_epa, all_weather, on=['date', 'city_name'], how='inner')
    
    # Sort for time-series operations
    df = df.sort_values(['city_name', 'date']).reset_index(drop=True)

    print("Engineering features...")
    for p in pollutants:
        if p in df.columns:
            # Lag features
            df[f"{p.lower()}_lag1"] = df.groupby('city_name')[p].shift(1)
            df[f"{p.lower()}_lag2"] = df.groupby('city_name')[p].shift(2)
            df[f"{p.lower()}_lag7"] = df.groupby('city_name')[p].shift(7)
            
            # Rolling 7-day mean and std
            df[f"{p.lower()}_roll7_mean"] = df.groupby('city_name')[p].transform(lambda x: x.rolling(7).mean())
            df[f"{p.lower()}_roll7_std"] = df.groupby('city_name')[p].transform(lambda x: x.rolling(7).std())

    # Time-based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    df['season'] = df['month'].apply(get_season)

    # Wind direction sin/cos encoding
    # wind_direction_10m_dominant
    df['wind_dir_sin'] = np.sin(df['wind_direction_10m_dominant'] * np.pi / 180)
    df['wind_dir_cos'] = np.cos(df['wind_direction_10m_dominant'] * np.pi / 180)
    df = df.drop(columns=['wind_direction_10m_dominant'])

    # Interactions
    df['temp_max_times_wind'] = df['temperature_2m_max'] * df['wind_speed_10m_max']
    df['humidity_times_pressure'] = df['relative_humidity_2m_mean'] * df['surface_pressure_mean']

    print("Creating target columns...")
    for p in pollutants:
        if p in df.columns:
            # Shift by -1 (next day/24h) and -2 (day after/48h)
            df[f"{p.lower()}_next_24h"] = df.groupby('city_name')[p].shift(-1)
            df[f"{p.lower()}_next_48h"] = df.groupby('city_name')[p].shift(-2)

    # Drop rows with NaN targets (the last 2 days of each city's time series)
    target_cols = [f"{p.lower()}_next_{h}h" for p in pollutants for h in [24, 48] if p in df.columns]
    df = df.dropna(subset=target_cols)

    print("Splitting data chronologically...")
    # Train: 2018-2021, Val: 2022, Test: 2023
    train_df = df[df['date'].dt.year.isin([2018, 2019, 2020, 2021])]
    val_df = df[df['date'].dt.year == 2022]
    test_df = df[df['date'].dt.year == 2023]

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    train_df.to_parquet("data/train.parquet", index=False)
    val_df.to_parquet("data/val.parquet", index=False)
    test_df.to_parquet("data/test.parquet", index=False)
    print("Preprocessed files saved to data/")

if __name__ == "__main__":
    preprocess()
