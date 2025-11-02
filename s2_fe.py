import pandas as pd
import numpy as np
import holidays

def _calculate_heat_index(temp_c, humidity):
    """Calculate the Heat Index (feels hot)."""
    temp_f = temp_c * 9/5 + 32
    hi = -42.379 + 2.04901523 * temp_f + 10.14333127 * humidity \
         - 0.22475541 * temp_f * humidity - 6.83783e-3 * temp_f**2 \
         - 5.481717e-2 * humidity**2 + 1.22874e-3 * temp_f**2 * humidity \
         + 8.5282e-4 * temp_f * humidity**2 - 1.99e-6 * temp_f**2 * humidity**2
    return (hi - 32) * 5/9

def create_features(df: pd.DataFrame, features_for_model: list) -> pd.DataFrame:
    """
    Generates features from aggregated data.
    This function will not fail if columns are missing.
    """
    df = df.sort_values("datetime").copy()

    # 1. & 2. Time-based & Cyclical features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    us_holidays = holidays.US()
    df['is_holiday'] = df['datetime'].dt.date.astype('datetime64[ns]').isin(us_holidays).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.0)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.0)

    # 3. Weather-based features
    df['heat_index_celsius'] = _calculate_heat_index(df['temp_celsius'], df['humidity_percent'])

    # 4. Renewable Energy & Net Demand features
    if 'wind_gen_MW' not in df.columns: df['wind_gen_MW'] = 0
    if 'solar_gen_MW' not in df.columns: df['solar_gen_MW'] = 0
    df['wind_gen_MW'] = df['wind_gen_MW'].fillna(0)
    df['solar_gen_MW'] = df['solar_gen_MW'].fillna(0)
    df['net_demand_MW'] = df['demand_MW'] - df['wind_gen_MW'] - df['solar_gen_MW']

    # 5. Interaction features
    df['temp_x_hour_sin'] = df['temp_celsius'] * df['hour_sin']

    # 6. Lag and Rolling features
    targets = ['demand_MW', 'temp_celsius', 'humidity_percent', 'price_USD_per_MWh', 'net_demand_MW']
    for target in targets:
        if target not in df.columns:
            continue
        for lag in [24, 168]:
            df[f'{target}_lag_{lag}h'] = df[target].shift(lag)
        df[f'{target}_rolling_mean_24h'] = df[target].rolling(window=24, min_periods=1).mean()
        df[f'{target}_rolling_std_24h'] = df[target].rolling(window=24, min_periods=1).std().fillna(0)

    df['demand_ewma_24h'] = df['demand_MW'].ewm(span=24, adjust=False).mean()

    print(f"  [FE] Before cleaning: df has {len(df)} rows.")

    # Print NaN info safely
    if 'price_USD_per_MWh_lag_168h' in df.columns:
        print(f"  [FE] NaNs in price_lag_168h: {df['price_USD_per_MWh_lag_168h'].isna().sum()}")
    if 'demand_MW_lag_168h' in df.columns:
        print(f"  [FE] NaNs in demand_lag_168h: {df['demand_MW_lag_168h'].isna().sum()}")

    # 7. Cleaning data
    # Identify the columns that are actually available for cleaning and modeling
    available_cols_for_model = [col for col in features_for_model if col in df.columns]

    # Only drop rows with NaN in the columns we will actually use
    # This prevents data loss if only the price column is missing
    df_cleaned = df.dropna(subset=available_cols_for_model).reset_index(drop=True)

    print(f"  [FE] After cleaning: df has {len(df_cleaned)} rows.")
    return df_cleaned