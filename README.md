# US Air Quality Forecast API

A REST API serving 48-hour PM2.5, O3, and NO2 air quality forecasts for 20 major US cities using XGBoost and FastAPI.

## Live API

The API is deployed on Render and can be accessed at: [https://air-quality-api-3afx.onrender.com](https://air-quality-api-3afx.onrender.com)
Interactive docs: [https://air-quality-api-3afx.onrender.com/docs](https://air-quality-api-3afx.onrender.com/docs)
> **Note:** The free Render tier sleeps after 15 minutes of inactivity. 
> The first request may take ~50 seconds to respond while the service wakes up.

### Example Commands

**Get 48-hour PM2.5 forecast for New York:**
```bash
curl "https://air-quality-api-3afx.onrender.com/forecast/new-york?pollutant=pm25"
```

**Get a list of supported cities:**
```bash
curl "https://air-quality-api-3afx.onrender.com/cities"
```

**System health and loaded models:**
```bash
curl "https://air-quality-api-3afx.onrender.com/health"
```

## Model Performance

The following table summarizes the performance of the XGBoost models compared to a persistence baseline (predicting tomorrow will be the same as today).

| Pollutant | RMSE | Units | Improvement Over Baseline |
|-----------|------|-------|---------------------------|
| PM2.5     | 7.04 | µg/m³ | 24.4%                     |
| O3        | 7.0  | ppb   | 26.5%                     |
| NO2       | 4.03 | µg/m³ | 34.7%                     |

## Tech Stack

- **Language:** Python 3.12+
- **API Framework:** FastAPI, Uvicorn
- **Data Processing:** Pandas, Polars, Numpy
- **Machine Learning:** XGBoost, Scikit-learn, Joblib
- **Hyperparameter Tuning:** Optuna
- **Weather Data:** Open-Meteo
- **AQI Data:** EPA AQS, AirNow
- **Deployment:** Render
- **HTTP Client:** httpx (Async)

## Architecture

The system follows a modular pipeline: a data collection service gathers historical and real-time data from EPA and Open-Meteo, which is then preprocessed for offline model training using XGBoost and Optuna for optimization. At runtime, the FastAPI application fetches live weather and AQI data, builds a feature vector, and performs inference using the pre-trained models to deliver 48-hour forecasts. The entire application is deployed as a Python web service on Render for reliable access. Models were trained on 6 years of historical data (2018–2023) and evaluated on a held-out 2023 test set.

## Local Setup

Follow these steps to get the API running on your local machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hunterdjacobson/air-quality-api.git
   cd air-quality-api
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory and add your credentials:
   ```env
   EPA_API_KEY=your_epa_key_here
   EPA_EMAIL=your_email@example.com
   AIRNOW_API_KEY=your_airnow_key_here
   ```

5. **Run the API locally:**
   ```bash
   uvicorn api.main:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000`. You can access the interactive Swagger documentation at `http://127.0.0.1:8000/docs`.

6. **Run the test suite:**
   ```bash
   pytest tests/ -v
   ```

## Data Sources

This project uses data from the following open sources:
- **EPA AQS:** For high-quality historical air quality data.
- **AirNow:** For real-time AQI monitoring and observations.
- **Open-Meteo:** For free, high-resolution weather forecast data.
