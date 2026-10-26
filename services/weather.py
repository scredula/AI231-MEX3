"""Weather via Open-Meteo (https://open-meteo.dev) — free, no API key, works worldwide.

get_weather(location): geocodes the location, returns current conditions + today's
high/low in a compact summary suitable for spoken answers.
"""
import httpx

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
    81: "rain showers", 82: "violent rain showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with hail",
}


class WeatherError(Exception):
    pass


async def get_weather(location: str, units: str = "metric") -> str:
    """Current weather + today's forecast for a place name. No API key required."""
    if not location or len(location) > 100:
        raise WeatherError("location must be 1..100 characters")
    unit_params = ({"temperature_unit": "fahrenheit", "wind_speed_unit": "mph"}
                   if units == "imperial" else {})
    async with httpx.AsyncClient(timeout=30) as client:
        geo = await client.get(GEOCODE_URL, params={"name": location, "count": 1, "language": "en"})
        if geo.status_code != 200:
            raise WeatherError(f"geocoding failed ({geo.status_code})")
        results = geo.json().get("results") or []
        if not results:
            return f"error: unknown location '{location}'"
        place = results[0]
        name = place.get("name", location)
        country = place.get("country", "")
        r = await client.get(FORECAST_URL, params={
            "latitude": place["latitude"], "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,weather_code",
            "forecast_days": 1,
            **unit_params,
        })
        if r.status_code != 200:
            raise WeatherError(f"forecast failed ({r.status_code})")
        d = r.json()
        cur = d.get("current", {})
        daily = d.get("daily", {})
        temp = cur.get("temperature_2m")
        code = cur.get("weather_code")
        desc = WMO.get(code, "unknown conditions")
        hum = cur.get("relative_humidity_2m")
        wind = cur.get("wind_speed_10m")
        unit = "°F" if units == "imperial" else "°C"
        parts = [f"Weather in {name}{', ' + country if country else ''}: {temp}{unit}, {desc}"]
        if hum is not None:
            parts.append(f"humidity {hum}%")
        if wind is not None:
            parts.append(f"wind {wind} {'mph' if units == 'imperial' else 'km/h'}")
        tmax, tmin = daily.get("temperature_2m_max", [None])[0], daily.get("temperature_2m_min", [None])[0]
        if tmax is not None and tmin is not None:
            parts.append(f"today {tmin}{unit} to {tmax}{unit}")
        return ". ".join(parts) + "."