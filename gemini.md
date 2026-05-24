# Air Quality Forecasting API — Project Context

## What this project is
A REST API that serves 48-hour PM2.5, O3, and NO2 air quality forecasts
for 200+ US cities. Built with Python, XGBoost, and FastAPI. Deployed
free on Render.

## Tech stack
- Python 3.12
- pandas, polars, xgboost, scikit-learn, optuna, joblib
- FastAPI + uvicorn
- httpx for async HTTP calls
- pydantic v2 for schemas
- pytest for tests

## Project structure
```air-quality-api/
├── data/
│   ├── collect_epa.py
│   ├── collect_weather.py
│   └── preprocess.py
├── models/
│   ├── train.py
│   ├── evaluate.py
│   └── saved/
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── forecast.py
├── tests/
├── .env               
├── requirements.txt
├── render.yaml
└── GEMINI.md
```

## Coding rules
- Use type hints on all functions
- Use pydantic BaseModel for all API request/response schemas
- Use httpx (not requests) for all HTTP calls — we want async
- Wind direction must always be encoded as sin/cos, never raw degrees
- Train/val/test split must be chronological (2018-2021 train, 2022 val, 2023 test)
- All model artifacts saved to models/saved/ as .pkl via joblib
- Load models once at startup with @lru_cache, never per-request
- Secrets come from os.environ, never hardcoded

## What to never do
- Never use the requests library (use httpx)
- Never shuffle time-series data before splitting
- Never hardcode API keys
- Never commit the .env file
