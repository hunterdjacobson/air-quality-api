import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock
from api.main import app
from api.schemas import ForecastResponse, HourlyForecast

@pytest.mark.asyncio
async def test_health():
    """Test that the health endpoint returns 200 and ok status."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Mock load_model to avoid file system dependency
        with patch("api.main.load_model", return_value=MagicMock()):
            response = await ac.get("/health")
    
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "models_loaded" in response.json()

@pytest.mark.asyncio
async def test_cities():
    """Test that the cities endpoint returns at least 10 cities."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/cities")
    
    assert response.status_code == 200
    cities = response.json()
    assert isinstance(cities, list)
    assert len(cities) >= 10
    assert "name" in cities[0]
    assert "lat" in cities[0]

@pytest.mark.asyncio
async def test_forecast_valid():
    """Test a valid forecast request for New York."""
    mock_response = ForecastResponse(
        city="New York",
        pollutant="pm25",
        generated_at="2026-05-20T12:00:00",
        forecasts=[
            HourlyForecast(timestamp="2026-05-21T12:00:00", predicted=12.5, unit="µg/m³", confidence_low=10.0, confidence_high=15.0),
            HourlyForecast(timestamp="2026-05-22T12:00:00", predicted=14.2, unit="µg/m³", confidence_low=11.3, confidence_high=17.1)
        ],
        model_rmse=7.03,
        model_r2=0.07
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        with patch("api.main.get_forecast", return_value=mock_response):
            response = await ac.get("/forecast/new-york", params={"pollutant": "pm25"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["city"] == "New York"
    assert len(data["forecasts"]) == 2

@pytest.mark.asyncio
async def test_forecast_invalid_city():
    """Test that an invalid city returns 404."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/forecast/invalid-city")
    
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_forecast_invalid_pollutant():
    """Test that an invalid pollutant returns 400."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/forecast/new-york", params={"pollutant": "invalid"})
    
    assert response.status_code == 400
    assert "Invalid pollutant" in response.json()["detail"]
