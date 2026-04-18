import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CITIES = {
    "Pristina": (42.6629, 21.1655), "Prizren": (42.2139, 20.7397),
    "Peja": (42.6591, 20.2883), "Gjakova": (42.3844, 20.4285),
    "Mitrovica": (42.8914, 20.8650), "Ferizaj": (42.3702, 21.1553),
    "Gjilan": (42.4635, 21.4694), "Podujeva": (42.9106, 21.1930),
    "Vushtrri": (42.8231, 20.9675), "Suhareka": (42.3586, 20.8250),
    "Rahovec": (42.3996, 20.6547), "Malisheva": (42.4822, 20.7458),
    "Skenderaj": (42.7867, 20.7897), "Kamenica": (42.5781, 21.5806),
    "Lipjan": (42.5217, 21.1294), "Dragash": (42.0267, 20.6522),
    "Istog": (42.7808, 20.4875), "Decan": (42.5478, 20.2889),
    "Kline": (42.6231, 20.5778), "Fushe Kosova": (42.6394, 21.0961),
    "Obiliq": (42.6867, 21.0703), "Drenas": (42.6211, 20.8905),
    "Shtime": (42.4331, 21.0397), "Hani i Elezit": (42.1514, 21.2961),
    "Novoberde": (42.6025, 21.4336), "Zubin Potok": (42.9156, 20.6892),
    "Zvecan": (42.9103, 20.8403)
}

def get_resilient_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

def fetch_historical_weather():
    # Set timeframe to exactly 31 days (Yields ~20,088 rows)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=31) 
    
    archive_url = "https://archive-api.open-meteo.com/v1/archive"
    session = get_resilient_session()
    all_city_dataframes = []

    logging.info(f"Fetching 1 month of data: {start_date} to {end_date}...")

    for city, (lat, lon) in CITIES.items():
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,surface_pressure,cloud_cover,wind_speed_10m,wind_direction_10m",
            "timezone": "Europe/Belgrade"
        }

        try:
            response = session.get(archive_url, params=params, timeout=15)
            response.raise_for_status() 
            data = response.json()

            hourly_data = data.get("hourly", {})
            if hourly_data:
                df_city = pd.DataFrame(hourly_data)
                df_city['city'] = city 
                all_city_dataframes.append(df_city)

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed for {city}: {e}")

    if all_city_dataframes:
        df_final = pd.concat(all_city_dataframes, ignore_index=True)

        # Cleanup
        df_final.rename(columns={'time': 'datetime'}, inplace=True)
        df_final['datetime'] = pd.to_datetime(df_final['datetime'])
        df_final["hour"] = df_final["datetime"].dt.hour
        df_final["day"] = df_final["datetime"].dt.day
        df_final["month"] = df_final["datetime"].dt.month
        df_final["year"] = df_final["datetime"].dt.year

        df_final = df_final.head(20470)

        file_name = "kosovo_historical_weather.csv"
        df_final.to_parquet(file_name, index=False)
        
        logging.info(f"✅ Success! Exact rows to {file_name}.")
    else:
        logging.error("No data extracted.")

if __name__ == "__main__":
    fetch_historical_weather()