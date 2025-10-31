from settings import CONFIG
from s1_extract_data import fetch_eia_data, fetch_weather
from s2_fe import create_features
import pandas as pd 


def save_data_pipeline():
    """Pipeline to fetch, merge, create features, and save data for all regions."""
    
    all_final_dfs = []  # reset mỗi lần chạy
    
    for region_name, region_info in CONFIG["regions"].items():
        print(f"\nProcessing region: {region_name}")
        region_code = region_info["code"]
        lat = region_info["lat"]
        lon = region_info["lon"]

        # Fetch data
        demand_df = fetch_eia_data(CONFIG["api_key"], region_code, CONFIG["start_date"], CONFIG["end_date"], 'demand')
        weather_df = fetch_weather(lat, lon, CONFIG["start_date"], CONFIG["end_date"])

        # Merge
        merged_df = pd.merge(demand_df, weather_df, on="datetime", how="left")
        print(f"  [Merge] Merged df has {len(merged_df)} rows.")

        # FE
        final_df = create_features(merged_df, CONFIG["features_for_model"])
        final_df['region'] = region_name
        all_final_dfs.append(final_df)

    # Combine all regions
    combined_final_df = pd.concat(all_final_dfs, ignore_index=True)
    print(f"\nCombined final df has {len(combined_final_df)} rows.")

    # Save file
    combined_final_df.to_csv("final_dataset.csv", index=False)
    return combined_final_df
