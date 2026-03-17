import requests
import pandas as pd
from datetime import datetime
import time
import os

API_KEY = "9c8a1f622bf54a18c1590e88db5d92f5"

FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"

cities = {
    "Pristina": (42.6629, 21.1655),
    "Prizren": (42.2139, 20.7397),
    "Peja": (42.6591, 20.2883),
    "Gjakova": (42.3844, 20.4285),
    "Mitrovica": (42.8914, 20.8650),
    "Ferizaj": (42.3702, 21.1553),
    "Gjilan": (42.4635, 21.4694),
    "Podujeva": (42.9106, 21.1930),
    "Vushtrri": (42.8231, 20.9675),
    "Suhareka": (42.3586, 20.8250),
    "Rahovec": (42.3996, 20.6547),
    "Malisheva": (42.4822, 20.7458),
    "Skenderaj": (42.7867, 20.7897),
    "Kamenica": (42.5781, 21.5806),
    "Lipjan": (42.5217, 21.1294),
    "Dragash": (42.0267, 20.6522),
    "Istog": (42.7808, 20.4875),
    "Decan": (42.5478, 20.2889),
    "Kline": (42.6231, 20.5778),
    "Fushe Kosova": (42.6394, 21.0961),
    "Obiliq": (42.6867, 21.0703),
    "Drenas": (42.6211, 20.8905),
    "Shtime": (42.4331, 21.0397),
    "Hani i Elezit": (42.1514, 21.2961),
    "Novoberde": (42.6025, 21.4336),
    "Zubin Potok": (42.9156, 20.6892),
    "Zvecan": (42.9103, 20.8403)
}

all_data = []

for city, (lat, lon) in cities.items():
    print("Fetching:", city)

    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "metric"
    }

    try:
        forecast_res = requests.get(FORECAST_URL, params=params).json()
        if forecast_res.get("cod") != "200":
            print("Forecast error:", city, forecast_res.get("message"))
        else:
            for item in forecast_res["list"]:
                all_data.append({
                    "city": city,
                    "type": "forecast",
                    "datetime": datetime.fromtimestamp(item["dt"]),
                    "temperature": item["main"]["temp"],
                    "feels_like": item["main"]["feels_like"],
                    "temp_min": item["main"]["temp_min"],
                    "temp_max": item["main"]["temp_max"],
                    "pressure": item["main"]["pressure"],
                    "humidity": item["main"]["humidity"],
                    "weather": item["weather"][0]["main"],
                    "description": item["weather"][0]["description"],
                    "wind_speed": item["wind"]["speed"],
                    "wind_deg": item["wind"].get("deg", 0),
                    "wind_gust": item["wind"].get("gust", 0),
                    "clouds": item["clouds"]["all"],
                    "visibility": item.get("visibility", 0),
                    "pop": item.get("pop", 0)
                })

        current_res = requests.get(CURRENT_URL, params=params).json()
        if current_res.get("cod") != 200:
            print("Current error:", city, current_res.get("message"))
        else:
            all_data.append({
                "city": city,
                "type": "current",
                "datetime": datetime.now(),
                "temperature": current_res["main"]["temp"],
                "feels_like": current_res["main"]["feels_like"],
                "temp_min": current_res["main"]["temp"],
                "temp_max": current_res["main"]["temp"],
                "pressure": current_res["main"]["pressure"],
                "humidity": current_res["main"]["humidity"],
                "weather": current_res["weather"][0]["main"],
                "description": current_res["weather"][0]["description"],
                "wind_speed": current_res["wind"]["speed"],
                "wind_deg": current_res["wind"].get("deg", 0),
                "wind_gust": current_res["wind"].get("gust", 0),
                "clouds": current_res["clouds"]["all"],
                "visibility": current_res.get("visibility", 0),
                "pop": None
            })

        time.sleep(1)

    except Exception as e:
        print("Error:", city, e)

df = pd.DataFrame(all_data)

df["datetime"] = pd.to_datetime(df["datetime"])
df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month

file_name = "kosovo_weather_dataset.csv"
if not os.path.exists(file_name):
    df.to_csv(file_name, index=False)
else:
    df.to_csv(file_name, mode='a', header=False, index=False)

print("\nDONE ✅")