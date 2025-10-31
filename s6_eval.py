import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
from s5_run_models import features_for_anomaly, contamination_rate
from s3_save_data import save_data_pipeline

df = save_data_pipeline()


model_cols = ['lof_anomaly', 'dbscan_anomaly', 'isolation_forest_anomaly']

def run_eval():
    # Overall anomaly counts
    print("\nAnomaly counts by region and model:")
    for region in df['region'].unique():
        region_df = df[df['region'] == region]
        print(f"\nRegion: {region}")
        for col in model_cols:
            count = region_df[col].sum()
            print(f"  {col}: {count} anomalies")
            percent = (count / len(region_df)) * 100
            print(f"    Percentage: {percent:.4f}%")

    # Generate simple ensemble score
    df['ensemble_score_simple'] = df[model_cols].sum(axis=1)
    print("\nSimple aggregate anomaly score distribution:")
    print(df['ensemble_score_simple'].value_counts().sort_index())

    # Advanced Ensemble
    print("\n--- Advanced Ensemble Anomaly Detection ---")

    # We trust LOF and Isolation Forest more, so we weight them higher
    weights = {
        'lof_anomaly': 0.4,
        'dbscan_anomaly': 0.2,
        'isolation_forest_anomaly': 0.4
    }
    print("Weights for ensemble:", weights)

    # Weight anomaly scores
    df['ensemble_weighted_score'] = (
        df['lof_anomaly'] * weights['lof_anomaly'] +
        df['dbscan_anomaly'] * weights['dbscan_anomaly'] +
        df['isolation_forest_anomaly'] * weights['isolation_forest_anomaly']
    )

    # Define threshold to decide which is the final anomaly
    # We consider a point an anomaly if at least ONE of the reliable models
    # (LOF or Isolation Forest) flags it. Since their weight is 0.4, any score >= 0.4 indicates at least one flagged it.
    anomaly_threshold = 0.4
    print(f"Anomaly threshold set at: {anomaly_threshold}")
    df['ensemble_final_anomaly'] = (df['ensemble_weighted_score'] >= anomaly_threshold).astype(int)

    # Compare results
    print("\n --- Compare the number of anomaly points ---")
    print(f"  - LOF: {df['lof_anomaly'].sum()}")
    print(f"  - Isolation Forest: {df['isolation_forest_anomaly'].sum()}")
    print(f"  - DBSCAN: {df['dbscan_anomaly'].sum()}")
    print(f"  - Simple Ensemble (>=1 vote): {(df['ensemble_score_simple'] >= 1).sum()}")
    print(f"  - Weighted Ensemble (final): {df['ensemble_final_anomaly'].sum()}")

    # Analysis SHAP for Isolation Forest
    import shap

    # Dictionaries to hold the results for each region
    all_shap_values = {}
    all_features_df = {}
    all_explainers = {}

    # Loop through each region to perform a separate SHAP analysis
    for region in df['region'].unique():
        print(f"\n--- Running SHAP Analysis for region: {region} ---")

        # 1. Isolate the data for the current region
        region_df = df[df['region'] == region].copy()

        # 2. Find the anomalies detected by the region-specific Isolation Forest
        anomalies_df = region_df[region_df['isolation_forest_anomaly'] == 1].copy()
        features_df = anomalies_df[features_for_anomaly]

        # Only run SHAP if there are anomalies detected for this region
        if not anomalies_df.empty:
            # 3. Create and retrain an Isolation Forest model ONLY on this region's data
            # This ensures the explainer uses the same logic that found the anomalies
            isolation_model = IsolationForest(n_estimators=200, contamination=contamination_rate, random_state=42)
            isolation_model.fit(region_df[features_for_anomaly])

            # 4. Create an explainer using the region-specific model and background data
            explainer = shap.Explainer(isolation_model, region_df[features_for_anomaly])

            # 5. Calculate SHAP values for the anomalies found in this region
            shap_values = explainer(features_df)

            # 6. Store the results in our dictionaries
            all_shap_values[region] = shap_values
            all_features_df[region] = features_df
            all_explainers[region] = explainer
            print(f"  SHAP analysis complete for {len(features_df)} anomalies in {region}.")
        else:
            print(f"  No anomalies detected by Isolation Forest in {region}; skipping SHAP analysis.")