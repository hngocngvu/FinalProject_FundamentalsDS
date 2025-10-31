import requests
import pandas as pd
import numpy as np
import holidays


def fetch_eia_data(api_key, region_code, start_date, end_date, data_type) -> pd.DataFrame:
    """
    Function to get data from EIA.
    """
    # Config endpoint and facet based on data_type
    endpoint_config = {
        'demand': {
            "endpoint": "https://api.eia.gov/v2/electricity/rto/region-data/data/",
            "facets": {"type": "D"}
        },
        'price': {
            "endpoint": "https://api.eia.gov/v2/electricity/rto/region-data/data/",
            "facets": {"type": "LMP"}
        },
        'wind': {
            "endpoint": "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/",
            "facets": {"fueltype": "WND"}
        },
        'solar': {
            "endpoint": "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/",
            "facets": {"fueltype": "SUN"}
        }
    }

    if data_type not in endpoint_config:
        print(f"Error: Invalid data_type '{data_type}'")
        return pd.DataFrame()

    config = endpoint_config[data_type]
    api_endpoint = config["endpoint"]
    facet_filters = config["facets"]

    base_params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": region_code,
        "start": start_date,
        "end": end_date,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000
    }

    for key, value in facet_filters.items():
        base_params[f"facets[{key}][]"] = value

    all_records = []
    offset = 0
    while True:
        params = base_params.copy()
        params["offset"] = offset
        try:
            r = requests.get(api_endpoint, params=params)
            r.raise_for_status()
            data = r.json()["response"]["data"]
            if not data:
                break
            all_records.extend(data)
            offset += len(data)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching EIA {data_type} data for {region_code}: {e}")
            return pd.DataFrame()

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["datetime"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors='coerce')

    rename_map = {
        'demand': 'demand_MW',
        'price': 'price_USD_per_MWh',
        'wind': 'wind_gen_MW',
        'solar': 'solar_gen_MW'
    }
    return df[["datetime", "value"]].rename(columns={"value": rename_map[data_type]}).dropna()


def fetch_weather(lat, lon, start_date, end_date) -> pd.DataFrame:
    """Get historical weather data from Open-Meteo."""
    params = {
        "latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m"
    }
    try:
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data["hourly"])
        df["datetime"] = pd.to_datetime(df["time"])
        return df.rename(columns={
            "time": "time_str",
            "temperature_2m": "temp_celsius",
            "relative_humidity_2m": "humidity_percent",
        })
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return pd.DataFrame()