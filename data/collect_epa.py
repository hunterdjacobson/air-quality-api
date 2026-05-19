import os
import asyncio
import pandas as pd
import httpx
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

EPA_API_KEY = os.getenv("EPA_API_KEY")
EPA_EMAIL = os.getenv("EPA_EMAIL")

# 20 major US cities with state code, county code, and CBSA name
CITIES = [
    {"name": "New York", "state": "36", "county": "061", "cbsa": "New York-Newark-Jersey City, NY-NJ-PA"},
    {"name": "Los Angeles", "state": "06", "county": "037", "cbsa": "Los Angeles-Long Beach-Anaheim, CA"},
    {"name": "Chicago", "state": "17", "county": "031", "cbsa": "Chicago-Naperville-Elgin, IL-IN-WI"},
    {"name": "Houston", "state": "48", "county": "201", "cbsa": "Houston-The Woodlands-Sugar Land, TX"},
    {"name": "Phoenix", "state": "04", "county": "013", "cbsa": "Phoenix-Mesa-Chandler, AZ"},
    {"name": "Philadelphia", "state": "42", "county": "101", "cbsa": "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD"},
    {"name": "San Antonio", "state": "48", "county": "029", "cbsa": "San Antonio-New Braunfels, TX"},
    {"name": "Dallas", "state": "48", "county": "113", "cbsa": "Dallas-Fort Worth-Arlington, TX"},
    {"name": "San Jose", "state": "06", "county": "085", "cbsa": "San Jose-Sunnyvale-Santa Clara, CA"},
    {"name": "Austin", "state": "48", "county": "453", "cbsa": "Austin-Round Rock-Georgetown, TX"},
    {"name": "Jacksonville", "state": "12", "county": "031", "cbsa": "Jacksonville, FL"},
    {"name": "Fort Worth", "state": "48", "county": "439", "cbsa": "Dallas-Fort Worth-Arlington, TX"},
    {"name": "Columbus", "state": "39", "county": "049", "cbsa": "Columbus, OH"},
    {"name": "Charlotte", "state": "37", "county": "119", "cbsa": "Charlotte-Concord-Gastonia, NC-SC"},
    {"name": "San Francisco", "state": "06", "county": "075", "cbsa": "San Francisco-Oakland-Berkeley, CA"},
    {"name": "Indianapolis", "state": "18", "county": "097", "cbsa": "Indianapolis-Carmel-Anderson, IN"},
    {"name": "Seattle", "state": "53", "county": "033", "cbsa": "Seattle-Tacoma-Bellevue, WA"},
    {"name": "Denver", "state": "08", "county": "031", "cbsa": "Denver-Aurora-Lakewood, CO"},
    {"name": "Nashville", "state": "47", "county": "037", "cbsa": "Nashville-Davidson--Murfreesboro--Franklin, TN"},
    {"name": "Boston", "state": "25", "county": "025", "cbsa": "Boston-Cambridge-Newton, MA-NH"},
]

# Pollutant parameter codes
POLLUTANTS = {
    "88101": "PM2.5",
    "44201": "O3",
    "42602": "NO2"
}

YEARS = range(2018, 2024)
BASE_URL = "https://aqs.epa.gov/data/api/dailyData/byCounty"

async def fetch_city_pollutant_year(
    client: httpx.AsyncClient, 
    param: str, 
    city: Dict[str, str], 
    year: int, 
    max_retries: int = 3
) -> List[Dict[str, Any]]:
    """Fetch daily data for a specific pollutant, city, and year with retry logic."""
    params = {
        "email": EPA_EMAIL,
        "key": EPA_API_KEY,
        "param": param,
        "bdate": f"{year}0101",
        "edate": f"{year}1231",
        "state": city["state"],
        "county": city["county"]
    }
    
    for attempt in range(max_retries):
        try:
            response = await client.get(BASE_URL, params=params, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            if data.get("Header", [{}])[0].get("status") == "Success":
                return data.get("Data", [])
            else:
                status_msg = data.get("Header", [{}])[0].get("status", "Unknown error")
                print(f"  [Attempt {attempt+1}] API returned non-success for {city['name']} {year}: {status_msg}")
        except Exception as e:
            print(f"  [Attempt {attempt+1}] Error fetching {city['name']} {year}: {e}")
        
        if attempt < max_retries - 1:
            await asyncio.sleep(2)  # Retry delay
            
    return []

async def main() -> None:
    if not EPA_API_KEY or not EPA_EMAIL:
        print("Error: EPA_API_KEY and EPA_EMAIL must be set in environment.")
        return

    async with httpx.AsyncClient() as client:
        for param_code, param_name in POLLUTANTS.items():
            print(f"Starting collection for {param_name} ({param_code})...")
            all_data = []
            
            for city in CITIES:
                print(f"  Fetching {city['name']}...")
                for year in YEARS:
                    data = await fetch_city_pollutant_year(client, param_code, city, year)
                    if data:
                        # Enrich data with CBSA name and city name if not present
                        for entry in data:
                            entry["city_name"] = city["name"]
                            entry["cbsa_name"] = city["cbsa"]
                        all_data.extend(data)
                    
                    # Respect the 2-second sleep between requests requirement
                    await asyncio.sleep(2)
                
            if all_data:
                df = pd.DataFrame(all_data)
                output_path = f"data/raw_{param_name}.parquet"
                os.makedirs("data", exist_ok=True)
                df.to_parquet(output_path, index=False)
                print(f"Saved {len(df)} records to {output_path}")
            else:
                print(f"No data collected for {param_name}")

if __name__ == "__main__":
    asyncio.run(main())
