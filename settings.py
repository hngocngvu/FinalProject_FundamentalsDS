import os
from dotenv import load_dotenv

def load_api_key():
    # 1️⃣ Try to load from .env in local environment
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(base_dir, '..', '.env')

    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)

    api_key = os.getenv("API_KEY")

    if not api_key:
        print("API_KEY not found. Please set it manually if running on Google Colab:")
        print('   ➜ import os; os.environ["API_KEY"] = "YOUR_KEY_HERE"')

    return api_key

api_key = load_api_key()

CONFIG = {
    "api_key": {api_key},
    "start_date": "2021-01-01",
    "end_date": "2024-12-31",
    "regions": {
        "ERCOT": {"code": "ERCO", "lat": 29.76, "lon": -95.36},
        "CAISO": {"code": "CISO", "lat": 34.05, "lon": -118.25},
    },
    "features_for_model": [
        # Cyclical & Time Features
        'hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos',
        'is_weekend', 'is_holiday',

        # Weather Features
        'heat_index_celsius', 'temp_x_hour_sin',

        # Demand Features
        'demand_MW_lag_24h',
        'demand_MW_lag_168h',
        'demand_MW_rolling_mean_24h',
        'demand_MW_rolling_std_24h',
        'demand_ewma_24h',

        # Temperature Features
        'temp_celsius_lag_24h',
        'temp_celsius_rolling_mean_24h',
        'temp_celsius_rolling_std_24h',

        # Net Demand Features
        'net_demand_MW',
        'net_demand_MW_lag_24h',
        'net_demand_MW_rolling_mean_24h',
        'net_demand_MW_rolling_std_24h',

        # Price Features
        'price_USD_per_MWh',
        'price_USD_per_MWh_lag_24h',
        'price_USD_per_MWh_rolling_mean_24h',
    ]
}