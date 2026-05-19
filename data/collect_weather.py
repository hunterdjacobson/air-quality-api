import os
import asyncio
import pandas as pd
import httpx
from typing import List, Dict, Any

# 20 major US cities with lat/lon coordinates
CITIES = [
    {"name": "New York", "lat": 40.7128, "lon": -74.0060},
    {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
    {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
    {"name": "Houston", "lat": 29.7604, "lon": -95.3698},
    {"name": "Phoenix", "lat": 33.4484, "lon": -112.0740},
    {"name": "Philadelphia", "lat": 39.9526, "lon": -75.1652},
    {"name": "San Antonio", "lat": 29.4241, "lon": -98.4936},
    {"name": "Dallas", "lat": 32.7767, "lon": -96.7970},
    {"name": "San Jose", "lat": 37.3382, "lon": -121.8863},
    {"name": "Austin", "lat": 30.2672, "lon": -97.7431},
    {"name": "Jacksonville", "lat": 30.3322, "lon": -81.6557},
    {"name": "Fort Worth", "lat": 32.7555, "lon": -97.3308},
    {"name": "Columbus", "lat": 39.9612, "lon": -82.9988},
    {"name": "Charlotte", "lat": 35.2271, "lon": -80.8431},
    {"name": "San Francisco", "lat": 37.7749, "lon": -122.4194},
    {"name": "Indianapolis", "lat": 39.7684, "lon": -86.1581},
    {"name": "Seattle", "lat": 47.6062, "lon": -122.3321},
    {"name": "Denver", "lat": 39.7392, "lon": -104.9903},
    {"name": "Nashville", "lat": 36.1627, "lon": -86.7816},
    {"name": "Boston", "lat": 42.3601, "lon": -71.0589},
]

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2018-01-01"
END_DATE = "2023-12-31"

VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "precipitation_sum",
    "shortwave_radiation_sum",
    "relative_humidity_2m_mean",
    "surface_pressure_mean"
]

async def fetch_weather_for_city(client: httpx.AsyncClient, city: Dict[str, Any]) -> pd.DataFrame:
    """Fetch historical weather data for a city."""
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(VARIABLES),
        "timezone": "auto"
    }
    
    try:
        response = await client.get(BASE_URL, params=params, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        
        if "daily" in data:
            df = pd.DataFrame(data["daily"])
            # Add city name for identification
            df["city_name"] = city["name"]
            return df
        else:
            print(f"  Error: No daily data returned for {city['name']}")
            return pd.DataFrame()
    except Exception as e:
        print(f"  Error fetching weather for {city['name']}: {e}")
        return pd.DataFrame()

async def main() -> None:
    os.makedirs("data", exist_ok=True)
    
    async with httpx.AsyncClient() as client:
        for city in CITIES:
            city_slug = city["name"].lower().replace(" ", "_")
            print(f"Fetching weather for {city['name']}...")
            
            df = await fetch_weather_for_city(client, city)
            
            if not df.empty:
                output_path = f"data/weather_{city_slug}.parquet"
                df.to_parquet(output_path, index=False)
                print(f"  Saved {len(df)} records to {output_path}")
            
            # Respect the 0.5-second sleep between requests
            await asyncio.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(main())
