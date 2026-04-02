"""Historical weather data for MLB games via Open-Meteo API."""

from datetime import datetime, timezone, timedelta

import requests
import pandas as pd

from mlb.ballpark_factors import STADIUM_COORDINATES, INDOOR_STADIUMS

# Default values for indoor stadiums or missing data
INDOOR_DEFAULTS = {"temperature": 72.0, "wind_speed": 0.0}


def fetch_weather_batch(lat, lon, start_date, end_date):
    """Fetch hourly temperature and wind speed from Open-Meteo archive API.

    Returns dict of {date_str: {hour_int: {temperature, wind_speed}}}.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,wind_speed_10m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "America/New_York",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        if not resp.ok:
            return {}
        data = resp.json()
    except (requests.RequestException, ValueError):
        return {}

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    winds = hourly.get("wind_speed_10m", [])

    result = {}
    for t, temp, wind in zip(times, temps, winds):
        # Format: "2025-06-15T19:00"
        date_str = t[:10]
        hour = int(t[11:13])
        if date_str not in result:
            result[date_str] = {}
        result[date_str][hour] = {
            "temperature": temp if temp is not None else 72.0,
            "wind_speed": wind if wind is not None else 5.0,
        }
    return result


def fetch_game_weather(venue_name, date_str, game_time_utc=None):
    """Fetch weather for a single game. Returns {temperature, wind_speed}."""
    if venue_name in INDOOR_STADIUMS:
        return INDOOR_DEFAULTS.copy()

    coords = STADIUM_COORDINATES.get(venue_name)
    if not coords:
        return INDOOR_DEFAULTS.copy()

    lat, lon = coords
    weather = fetch_weather_batch(lat, lon, date_str, date_str)
    day_data = weather.get(date_str, {})

    # Convert UTC game time to Eastern (Open-Meteo returns Eastern data)
    target_hour = 19  # default: 7 PM ET
    if game_time_utc:
        try:
            utc_hour = int(str(game_time_utc)[:2])
            month = int(date_str[5:7]) if len(date_str) >= 7 else 7
            offset = 4 if 3 <= month <= 10 else 5
            target_hour = (utc_hour - offset) % 24
        except (ValueError, IndexError):
            pass

    if target_hour in day_data:
        return day_data[target_hour]
    # Fallback: average of available hours
    if day_data:
        avg_temp = sum(d["temperature"] for d in day_data.values()) / len(day_data)
        avg_wind = sum(d["wind_speed"] for d in day_data.values()) / len(day_data)
        return {"temperature": avg_temp, "wind_speed": avg_wind}
    return INDOOR_DEFAULTS.copy()


def add_weather_features(df):
    """Add temperature and wind_speed columns to a DataFrame of games.

    Batches API calls per venue to minimize requests. Indoor stadiums
    get default values without API calls.
    """
    print("   -> Fetching weather data...")
    df = df.copy()
    df["temperature"] = float("nan")
    df["wind_speed"] = float("nan")

    if "venue_name" not in df.columns or "date" not in df.columns:
        df["temperature"] = INDOOR_DEFAULTS["temperature"]
        df["wind_speed"] = INDOOR_DEFAULTS["wind_speed"]
        return df

    # Set indoor stadiums to defaults immediately
    if "venue_indoor" in df.columns:
        indoor_mask = df["venue_indoor"] == 1
    else:
        indoor_mask = df["venue_name"].isin(INDOOR_STADIUMS)
    df.loc[indoor_mask, "temperature"] = INDOOR_DEFAULTS["temperature"]
    df.loc[indoor_mask, "wind_speed"] = INDOOR_DEFAULTS["wind_speed"]

    # Batch by venue for outdoor stadiums
    outdoor = df[~indoor_mask & df["temperature"].isna()].copy()
    if outdoor.empty:
        print("      No outdoor games need weather data.")
        return df

    venues = outdoor["venue_name"].unique()
    fetched = 0
    for venue in venues:
        coords = STADIUM_COORDINATES.get(venue)
        if not coords:
            df.loc[df["venue_name"] == venue, "temperature"] = INDOOR_DEFAULTS["temperature"]
            df.loc[df["venue_name"] == venue, "wind_speed"] = INDOOR_DEFAULTS["wind_speed"]
            continue

        venue_rows = outdoor[outdoor["venue_name"] == venue]
        dates = sorted(venue_rows["date"].unique())
        if not dates:
            continue

        lat, lon = coords
        weather_data = fetch_weather_batch(lat, lon, dates[0], dates[-1])
        fetched += 1

        for idx, row in venue_rows.iterrows():
            date_str = str(row["date"])[:10]
            game_time = str(row.get("game_time", ""))
            # game_time is UTC (e.g. "23:00"); convert to Eastern (UTC-4/-5)
            # since Open-Meteo data is in America/New_York
            target_hour = 19  # default: 7 PM ET
            if game_time and len(game_time) >= 2:
                try:
                    utc_hour = int(game_time[:2])
                    # Approximate Eastern offset: -4 during DST (Apr-Oct), -5 otherwise
                    month = int(date_str[5:7]) if len(date_str) >= 7 else 7
                    offset = 4 if 3 <= month <= 10 else 5
                    target_hour = (utc_hour - offset) % 24
                except ValueError:
                    pass

            day_data = weather_data.get(date_str, {})
            if target_hour in day_data:
                df.at[idx, "temperature"] = day_data[target_hour]["temperature"]
                df.at[idx, "wind_speed"] = day_data[target_hour]["wind_speed"]
            elif day_data:
                avg_temp = sum(d["temperature"] for d in day_data.values()) / len(day_data)
                avg_wind = sum(d["wind_speed"] for d in day_data.values()) / len(day_data)
                df.at[idx, "temperature"] = avg_temp
                df.at[idx, "wind_speed"] = avg_wind
            else:
                df.at[idx, "temperature"] = INDOOR_DEFAULTS["temperature"]
                df.at[idx, "wind_speed"] = INDOOR_DEFAULTS["wind_speed"]

    # Fill any remaining NaN
    df["temperature"] = df["temperature"].fillna(INDOOR_DEFAULTS["temperature"])
    df["wind_speed"] = df["wind_speed"].fillna(INDOOR_DEFAULTS["wind_speed"])

    matched = (df["temperature"] != INDOOR_DEFAULTS["temperature"]).sum()
    print(f"      Weather data: {fetched} venue batches, {matched}/{len(df)} rows with real data")
    return df
