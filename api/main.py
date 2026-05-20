import os
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api.schemas import CityInfo, HealthResponse, ForecastResponse
from api.forecast import CITIES, load_model, get_forecast

# Load environment variables at startup
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm all models on app startup."""
    for pollutant in ["pm2.5", "o3", "no2"]:
        try:
            load_model(pollutant)
        except Exception as e:
            print(f"Error pre-warming {pollutant} model: {e}")
    yield

app = FastAPI(
    title="US Air Quality Forecast API",
    description="48-hour PM2.5, O3, and NO2 forecasts for 200+ US cities.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to interactive documentation."""
    return RedirectResponse(url="/docs", status_code=302)

# CORS middleware allowing all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/forecast/{city_slug}", response_model=ForecastResponse)
async def forecast_endpoint(
    city_slug: str,
    pollutant: str = Query("pm25"),
    hours: int = Query(48, ge=1, le=48)
):
    """
    Get 48-hour air quality forecast for a specific city and pollutant.
    """
    if pollutant not in ["pm25", "o3", "no2"]:
        raise HTTPException(status_code=400, detail="Invalid pollutant. Must be pm25, o3, or no2.")
    
    try:
        # Convert pm25 to pm2.5 for internal logic if necessary, 
        # but forecast.py handles pm25 by stripping dots anyway.
        response = await get_forecast(city_slug, pollutant)
        
        # Filter forecasts if 'hours' is less than 48
        # Our get_forecast returns 24h and 48h.
        if hours <= 24:
            response.forecasts = response.forecasts[:1]
            
        return response
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cities", response_model=List[CityInfo])
async def list_cities():
    """Returns a list of all supported cities."""
    return [
        CityInfo(
            name=city["name"],
            state=city["state"],
            lat=city["lat"],
            lon=city["lon"]
        ) for city in CITIES.values()
    ]

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check and loaded models list."""
    loaded_models = []
    for p in ["pm2.5", "o3", "no2"]:
        # Check if loaded in cache without re-loading
        # Since we use lru_cache, we can just call it (it will be fast if loaded)
        try:
            load_model(p)
            loaded_models.append(p)
        except:
            pass
            
    return HealthResponse(
        status="ok",
        models_loaded=loaded_models
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
