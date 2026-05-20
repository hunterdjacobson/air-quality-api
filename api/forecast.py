import os
import asyncio
import joblib
import httpx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
from functools import lru_cache
from fastapi import HTTPException
from api.schemas import CityInfo, HourlyForecast, ForecastResponse

# Load environment variables
AIRNOW_API_KEY = os.getenv("AIRNOW_API_KEY")

# 20 Cities from data collection
CITIES = {
    "new-york": {
        "name": "New York", "state": "NY", "lat": 40.7128, "lon": -74.0060, 
        "reporting_area": "New York City"
    },
    "los-angeles": {
        "name": "Los Angeles", "state": "CA", "lat": 34.0522, "lon": -118.2437, 
        "reporting_area": "Los Angeles"
    },
    "chicago": {
        "name": "Chicago", "state": "IL", "lat": 41.8781, "lon": -87.6298, 
        "reporting_area": "Chicago"
    },
    "houston": {
        "name": "Houston", "state": "TX", "lat": 29.7604, "lon": -95.3698, 
        "reporting_area": "Houston"
    },
    "phoenix": {
        "name": "Phoenix", "state": "AZ", "lat": 33.4484, "lon": -112.0740, 
        "reporting_area": "Phoenix"
    },
    "philadelphia": {
        "name": "Philadelphia", "state": "PA", "lat": 39.9526, "lon": -75.1652, 
        "reporting_area": "Philadelphia"
    },
    "san-antonio": {
        "name": "San Antonio", "state": "TX", "lat": 29.4241, "lon": -98.4936, 
        "reporting_area": "San Antonio"
    },
    "dallas": {
        "name": "Dallas", "state": "TX", "lat": 32.7767, "lon": -96.7970, 
        "reporting_area": "Dallas"
    },
    "san-jose": {
        "name": "San Jose", "state": "CA", "lat": 37.3382, "lon": -121.8863, 
        "reporting_area": "San Jose"
    },
    "austin": {
        "name": "Austin", "state": "TX", "lat": 30.2672, "lon": -97.7431, 
        "reporting_area": "Austin"
    },
    "jacksonville": {
        "name": "Jacksonville", "state": "FL", "lat": 30.3322, "lon": -81.6557, 
        "reporting_area": "Jacksonville"
    },
    "fort-worth": {
        "name": "Fort Worth", "state": "TX", "lat": 32.7555, "lon": -97.3308, 
        "reporting_area": "Fort Worth"
    },
    "columbus": {
        "name": "Columbus", "state": "OH", "lat": 39.9612, "lon": -82.9988, 
        "reporting_area": "Columbus"
    },
    "charlotte": {
        "name": "Charlotte", "state": "NC", "lat": 35.2271, "lon": -80.8431, 
        "reporting_area": "Charlotte"
    },
    "san-francisco": {
        "name": "San Francisco", "state": "CA", "lat": 37.7749, "lon": -122.4194, 
        "reporting_area": "San Francisco"
    },
    "indianapolis": {
        "name": "Indianapolis", "state": "IN", "lat": 39.7684, "lon": -86.1581, 
        "reporting_area": "Indianapolis"
    },
    "seattle": {
        "name": "Seattle", "state": "WA", "lat": 47.6062, "lon": -122.3321, 
        "reporting_area": "Seattle"
    },
    "denver": {
        "name": "Denver", "state": "CO", "lat": 39.7392, "lon": -104.9903, 
        "reporting_area": "Denver"
    },
    "nashville": {
        "name": "Nashville", "state": "TN", "lat": 36.1627, "lon": -86.7816, 
        "reporting_area": "Nashville"
    },
    "boston": {
        "name": "Boston", "state": "MA", "lat": 42.3601, "lon": -71.0589, 
        "reporting_area": "Boston"
    },
}

MODEL_RMSE = {
    "pm2.5": 7.0351,
    "o3": 0.0065,
    "no2": 4.0312
}

MODEL_R2 = {
    "pm2.5": 0.0733,
    "o3": 0.5554,
    "no2": 0.5606
}

@lru_cache(maxsize=None)
def load_model(pollutant: str):
    """Load and cache the XGBoost model for a given pollutant."""
    p_key = pollutant.lower().replace(".", "")
    model_path = f"models/saved/{p_key}_model.pkl"
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

async def fetch_current_aqi(lat: float, lon: float, pollutant: str) -> List[Dict[str, Any]]:
    """
    Fetch last 7 days of AQI readings from AirNow.
    We fetch current and then historical for the past 7 days.
    """
    if not AIRNOW_API_KEY:
        return []

    async with httpx.AsyncClient() as client:
        readings = []
        today = datetime.utcnow()
        
        # Mapping for AirNow parameter names
        airnow_param = {
            "pm2.5": "PM2.5",
            "o3": "O3",
            "no2": "NO2"
        }.get(pollutant.lower())

        # Fetch current and historical (last 7 days)
        tasks = []
        for i in range(8):  # 0 to 7 days ago
            date_str = (today - timedelta(days=i)).strftime("%Y-%m-%dT%H:%M")
            url = "https://www.airnowapi.org/aq/observation/latLong/historical/"
            params = {
                "latitude": lat,
                "longitude": lon,
                "date": date_str,
                "distance": 25,
                "API_KEY": AIRNOW_API_KEY,
                "format": "application/json"
            }
            tasks.append(client.get(url, params=params))
        
        responses = await asyncio.gather(*tasks)
        
        for resp in responses:
            if resp.status_code == 200:
                data = resp.json()
                for obs in data:
                    if obs.get("ParameterName") == airnow_param:
                        readings.append({
                            "date": obs.get("DateObserved"),
                            "value": obs.get("ParameterValue")
                        })
        
        # Sort by date
        readings.sort(key=lambda x: x["date"])
        return readings

async def fetch_weather_forecast(lat: float, lon: float) -> Dict[str, Any]:
    """Fetch next 48 hours (2 days) of weather forecast from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "wind_speed_10m_max",
            "wind_direction_10m_dominant",
            "precipitation_sum",
            "shortwave_radiation_sum",
            "relative_humidity_2m_mean",
            "surface_pressure_mean"
        ],
        "timezone": "auto",
        "past_days": 1,  # Get yesterday too for 24h prediction logic
        "forecast_days": 2
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json().get("daily", {})

def build_features(
    pollutant: str, 
    history: List[Dict[str, Any]], 
    weather: Dict[str, Any], 
    day_idx: int
) -> pd.DataFrame:
    """
    Construct the feature vector for a specific prediction day.
    day_idx: 0 for yesterday (to predict tomorrow/24h), 1 for today (to predict day after/48h)
    Wait, let's align with training: X(t) predicts y(t+2).
    So X(today) predicts Day 2 (48h).
    X(yesterday) predicts Day 1 (24h).
    In Open-Meteo with past_days=1: idx 0 is yesterday, idx 1 is today.
    """
    # Daily weather features for the target 'X' day
    # indices: 0=yesterday, 1=today, 2=tomorrow
    # X(today) is idx 1. X(yesterday) is idx 0.
    
    # Pollutant history for 'X' day
    # We need lags from the perspective of 'X' day
    # If X is today (idx 1), lag1 is yesterday, etc.
    
    # Prepare a dataframe with all pollutant history
    pdf = pd.DataFrame(history)
    if pdf.empty:
        # Fallback if no history
        pdf = pd.DataFrame([{"value": 0.0, "date": "2000-01-01"}] * 10)
    
    pdf['value'] = pdf['value'].astype(float)
    
    # We need at least 7 days of history to compute roll7
    # For now, we'll take the last values
    current_vals = pdf['value'].tolist()
    
    def get_lags(vals):
        # vals should end at 'X' day
        v = vals[-1] if vals else 0
        l1 = vals[-2] if len(vals) > 1 else v
        l2 = vals[-3] if len(vals) > 2 else v
        l7 = vals[-8] if len(vals) > 7 else v
        roll7_mean = np.mean(vals[-7:]) if len(vals) >= 7 else v
        roll7_std = np.std(vals[-7:]) if len(vals) >= 7 else 0
        return l1, l2, l7, roll7_mean, roll7_std

    # Features for Day 2 (48h) uses today's data (idx 1 in weather)
    # Features for Day 1 (24h) uses yesterday's data (idx 0 in weather)
    w_idx = day_idx 
    
    # Calculate lags based on history up to that day
    # history is sorted by date. Assume last entry is 'today'.
    if day_idx == 1: # Today
        vals = current_vals
    else: # Yesterday
        vals = current_vals[:-1] if len(current_vals) > 1 else current_vals

    l1, l2, l7, r7m, r7s = get_lags(vals)

    # Weather variables
    t_max = weather["temperature_2m_max"][w_idx]
    t_min = weather["temperature_2m_min"][w_idx]
    w_max = weather["wind_speed_10m_max"][w_idx]
    precip = weather["precipitation_sum"][w_idx]
    rad = weather["shortwave_radiation_sum"][w_idx]
    hum = weather["relative_humidity_2m_mean"][w_idx]
    pres = weather["surface_pressure_mean"][w_idx]
    w_dir = weather["wind_direction_10m_dominant"][w_idx]

    # Time features
    dt = datetime.utcnow() if day_idx == 1 else datetime.utcnow() - timedelta(days=1)
    dow = dt.weekday()
    month = dt.month
    weekend = 1 if dow >= 5 else 0
    season = 1 if month in [12,1,2] else 2 if month in [3,4,5] else 3 if month in [6,7,8] else 4

    # Derived
    w_sin = np.sin(w_dir * np.pi / 180)
    w_cos = np.cos(w_dir * np.pi / 180)
    temp_wind = t_max * w_max
    hum_pres = hum * pres

    # Feature dict - MUST MATCH ORDER OF FEATURE_COLS
    # ['temperature_2m_max', 'temperature_2m_min', 'wind_speed_10m_max', 'precipitation_sum', 
    #  'shortwave_radiation_sum', 'relative_humidity_2m_mean', 'surface_pressure_mean', 
    #  'pm2.5_lag1', 'pm2.5_lag2', 'pm2.5_lag7', 'pm2.5_roll7_mean', 'pm2.5_roll7_std', 
    #  'o3_lag1', 'o3_lag2', 'o3_lag7', 'o3_roll7_mean', 'o3_roll7_std', 
    #  'no2_lag1', 'no2_lag2', 'no2_lag7', 'no2_roll7_mean', 'no2_roll7_std', 
    #  'day_of_week', 'month', 'is_weekend', 'season', 'wind_dir_sin', 'wind_dir_cos', 
    #  'temp_max_times_wind', 'humidity_times_pressure']

    features = {
        'temperature_2m_max': t_max,
        'temperature_2m_min': t_min,
        'wind_speed_10m_max': w_max,
        'precipitation_sum': precip,
        'shortwave_radiation_sum': rad,
        'relative_humidity_2m_mean': hum,
        'surface_pressure_mean': pres,
    }

    # Add pollutant-specific lags
    # For the target pollutant, use calculated lags. For others, use 0 or some default.
    # In a real app, we'd fetch all 3 pollutants' history.
    for p in ["pm2.5", "o3", "no2"]:
        if p == pollutant.lower():
            features[f"{p}_lag1"] = l1
            features[f"{p}_lag2"] = l2
            features[f"{p}_lag7"] = l7
            features[f"{p}_roll7_mean"] = r7m
            features[f"{p}_roll7_std"] = r7s
        else:
            # Placeholder for other pollutants if not fetched
            features[f"{p}_lag1"] = 0
            features[f"{p}_lag2"] = 0
            features[f"{p}_lag7"] = 0
            features[f"{p}_roll7_mean"] = 0
            features[f"{p}_roll7_std"] = 0

    features.update({
        'day_of_week': dow,
        'month': month,
        'is_weekend': weekend,
        'season': season,
        'wind_dir_sin': w_sin,
        'wind_dir_cos': w_cos,
        'temp_max_times_wind': temp_wind,
        'humidity_times_pressure': hum_pres
    })

    return pd.DataFrame([features])

async def get_forecast(city_slug: str, pollutant: str) -> ForecastResponse:
    if city_slug not in CITIES:
        raise HTTPException(status_code=404, detail="City not found")
    
    city = CITIES[city_slug]
    
    # Internal normalization: pm25 -> pm2.5
    pollutant = pollutant.lower()
    if pollutant == "pm25":
        pollutant = "pm2.5"
        
    if pollutant not in MODEL_RMSE:
        raise HTTPException(status_code=400, detail="Invalid pollutant")

    # Fetch data concurrently
    history_task = fetch_current_aqi(city["lat"], city["lon"], pollutant)
    weather_task = fetch_weather_forecast(city["lat"], city["lon"])
    
    history, weather = await asyncio.gather(history_task, weather_task)
    
    model = load_model(pollutant)
    
    # Predict for 24h (using yesterday's features) and 48h (using today's features)
    # Open-Meteo past_days=1 means idx 0 is yesterday, idx 1 is today.
    
    forecasts = []
    for h, day_idx in [(24, 0), (48, 1)]:
        X = build_features(pollutant, history, weather, day_idx)
        prediction = float(model.predict(X)[0])
        
        # Ensure non-negative prediction
        prediction = max(0.0, prediction)
        
        # Confidence bounds ±20%
        conf_low = prediction * 0.8
        conf_high = prediction * 1.2
        
        timestamp = (datetime.utcnow() + timedelta(hours=h)).isoformat()
        
        unit = "µg/m³" if pollutant == "pm2.5" else "ppb"
        
        forecasts.append(HourlyForecast(
            timestamp=timestamp,
            predicted=round(prediction, 2),
            unit=unit,
            confidence_low=round(conf_low, 2),
            confidence_high=round(conf_high, 2)
        ))

    return ForecastResponse(
        city=city["name"],
        pollutant=pollutant,
        generated_at=datetime.utcnow().isoformat(),
        forecasts=forecasts,
        model_rmse=MODEL_RMSE[pollutant],
        model_r2=MODEL_R2[pollutant]
    )
