from typing import List
from pydantic import BaseModel

class HourlyForecast(BaseModel):
    """Represents a single hour of predicted air quality data."""
    timestamp: str  # ISO 8601
    predicted: float  # g/m³ or ppb
    unit: str
    confidence_low: float
    confidence_high: float

class ForecastResponse(BaseModel):
    """The main response for a city's air quality forecast."""
    city: str
    pollutant: str
    generated_at: str  # ISO 8601
    forecasts: List[HourlyForecast]
    model_rmse: float
    model_r2: float

class CityInfo(BaseModel):
    """Metadata about a city in the system."""
    name: str
    state: str
    lat: float
    lon: float

class HealthResponse(BaseModel):
    """System health check response."""
    status: str
    models_loaded: List[str]
