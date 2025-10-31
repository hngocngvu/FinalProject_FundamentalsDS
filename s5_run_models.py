import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from config_models import run_lof, run_isolation_forest, tune_dbscan_hyperparameters
from settings import CONFIG
from s3_save_data import save_data_pipeline

df = save_data_pipeline()
contamination_rate = 0.01  # 1% anomalies
features_for_anomaly = [f for f in CONFIG["features_for_model"] if 'price' not in f and f != 'net_demand_MW']
features_for_anomaly = [f for f in features_for_anomaly if f in df.columns]

print(f"Loaded dataset with {len(df)} rows.")

def run_models():
    # Remove price related features and leakage features
    features_for_anomaly = [f for f in CONFIG["features_for_model"] if 'price' not in f and f != 'net_demand_MW']
    features_for_anomaly = [f for f in features_for_anomaly if f in df.columns]

    print(f"Using {len(features_for_anomaly)} features for anomaly detection.")
    print(f"Features: {features_for_anomaly}")

    # Hyperparameter tuning for DBSCAN per region (this part is already correct)
    best_dbscan_params = {}
    for region in df['region'].unique():
        region_df = df[df['region'] == region]
        best_params = tune_dbscan_hyperparameters(region_df, region, features_for_anomaly, n_trials=50)
        best_dbscan_params[region] = best_params

    contamination_rate = 0.01  # 1% anomalies

    # Initialize the anomaly columns to 0 first
    df['lof_anomaly'] = 0
    df['isolation_forest_anomaly'] = 0
    df['dbscan_anomaly'] = 0

    # Loop through each region to run all models
    for region in df['region'].unique():
        print(f"\n--- Running all models for region: {region} ---")

        # Create a boolean mask to identify the rows for the current region
        region_mask = df['region'] == region

        # Get the features for just this region
        region_features = df.loc[region_mask, features_for_anomaly]

        # 1. Run Local Outlier Factor for the region
        lof_predictions = run_lof(region_features, features_for_anomaly, contamination=contamination_rate)
        df.loc[region_mask, 'lof_anomaly'] = lof_predictions
        print(f"  [Model] Local Outlier Factor found {df.loc[region_mask, 'lof_anomaly'].sum()} outliers.")

        # 2. Run DBSCAN for the region using its tuned hyperparameters
        params = best_dbscan_params[region]
        print(f"  [Model] Running DBSCAN with params: {params}")
        scaler = StandardScaler()
        region_features_scaled = scaler.fit_transform(region_features)
        model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'], n_jobs=-1)
        predictions = model.fit_predict(region_features_scaled)
        df.loc[region_mask, 'dbscan_anomaly'] = [1 if x == -1 else 0 for x in predictions]
        print(f"  [Model] DBSCAN found {df.loc[region_mask, 'dbscan_anomaly'].sum()} outliers.")

        # 3. Run Isolation Forest for the region
        iso_forest_predictions = run_isolation_forest(region_features, features_for_anomaly, contamination=contamination_rate)
        df.loc[region_mask, 'isolation_forest_anomaly'] = iso_forest_predictions
        print(f"  [Model] Isolation Forest found {df.loc[region_mask, 'isolation_forest_anomaly'].sum()} outliers.")


    print("\n--- Anomaly detection completed for all models and all regions. ---")
    print(f"Total LOF Anomalies: {df['lof_anomaly'].sum()}")
    print(f"Total DBSCAN Anomalies: {df['dbscan_anomaly'].sum()}")
    print(f"Total Isolation Forest Anomalies: {df['isolation_forest_anomaly'].sum()}")