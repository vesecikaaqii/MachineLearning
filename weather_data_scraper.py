import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import re

# cities to scrape
cities = [
    "Pristina+Kosovo",
    "Prizren+Kosovo",
    "Peja+Kosovo",
    "Gjakova+Kosovo",
    "Mitrovica+Kosovo",
    "Ferizaj+Kosovo",
    "Gjilan+Kosovo",
    "Podujeva+Kosovo",
    "Vushtrri+Kosovo",
    "Suhareka+Kosovo",
    "Rahovec+Kosovo",
    "Malisheva+Kosovo",
    "Skenderaj+Kosovo",
    "Kamenica+Kosovo",
    "Lipjan+Kosovo",
    "Dragash+Kosovo",
    "Istog+Kosovo",
    "Decan+Kosovo",
    "Kline+Kosovo",
    "Fushe+Kosova+Kosovo"
]

headers = {
    "User-Agent": "Mozilla/5.0"
}

hourly_data = []
tenday_data = []


def clean_temperature(temp):
    number = re.findall(r'\d+', temp)
    return int(number[0]) if number else None


def scrape_hourly(city):

    url = f"https://weather.com/weather/hourbyhour/l/{city}"

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    hours = soup.find_all("h3")
    temps = soup.find_all("span", {"data-testid": "TemperatureValue"})

    for h, t in zip(hours, temps):

        hourly_data.append({
            "date": datetime.now().date(),
            "city": city.replace("+Kosovo",""),
            "time": h.text,
            "temperature": clean_temperature(t.text)
        })


def scrape_10day(city):

    url = f"https://weather.com/weather/tenday/l/{city}"

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    days = soup.find_all("h2")
    temps = soup.find_all("span", {"data-testid": "TemperatureValue"})

    for d, t in zip(days, temps):

        tenday_data.append({
            "date_collected": datetime.now().date(),
            "city": city.replace("+Kosovo",""),
            "day": d.text,
            "temperature": clean_temperature(t.text)
        })


# scrape all cities
for city in cities:

    print("Scraping:", city)

    try:
        scrape_hourly(city)
        scrape_10day(city)
    except:
        print("Error scraping", city)


# create dataframes
hourly_df = pd.DataFrame(hourly_data)
tenday_df = pd.DataFrame(tenday_data)


# save datasets
hourly_df.to_csv("kosovo_hourly_weather.csv", index=False)
tenday_df.to_csv("kosovo_10day_weather.csv", index=False)


print("\nScraping finished!")
print("Hourly dataset rows:", len(hourly_df))
print("10-day dataset rows:", len(tenday_df))