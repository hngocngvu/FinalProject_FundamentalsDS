import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from s6_eval import model_cols, all_shap_values, all_explainers, all_features_df, get_anomaly_df, df
from IPython.display import display
import os

df= get_anomaly_df(df)
print("--- Start deep analysis of anomalies ---")

def deep_analyze_anomalies():
    # Loop through each region to perform a separate deep analysis
    for region in df['region'].unique():
        print(f"\n--- Starting deep analysis of anomalies for {region} ---")

        # Check if we have SHAP results for this region. This is the only check needed.
        if region not in all_shap_values:
            print(f"Skipping deep analysis for {region} as no SHAP values were calculated.")
            continue

        # --- Step 1: Investigate the most reliable outliers FOR THIS REGION ---
        region_df = df[df['region'] == region]
        high_confidence_anomalies = region_df[region_df['ensemble_score_simple'] >= 3].copy() # ALL 3 models agree

        print(f"[Investigation] Found {len(high_confidence_anomalies)} high confidence anomalies in {region}.")

        if high_confidence_anomalies.empty:
            print("No high-confidence anomalies to display or plot.")
            continue # Skip to the next region if there are none

        print("Showing the top 5 high confidence anomalies:")
        display(high_confidence_anomalies.sort_values(by='ensemble_weighted_score', ascending=False).head())

        # Select top 5 anomalies for detailed analysis
        top_5_anomalies = high_confidence_anomalies.sort_values(by='ensemble_weighted_score', ascending=False).head(5)
        top_5_indices = top_5_anomalies.index.tolist()
        print(f"\nTop 5 anomaly indices for {region}: {top_5_indices}")

        # --- Step 2: Use the CORRECT region-specific SHAP data ---
        # Load the pre-calculated, region-specific explainer and values
        explainer = all_explainers[region]
        shap_values_for_region = all_shap_values[region]
        features_df_for_region = all_features_df[region]

        # Loop through each of the top 5 anomalies
        for i, event_index in enumerate(top_5_indices, 1):
            print(f"\n--- SHAP Waterfall Plot for anomaly #{i} at index {event_index} in {region} ---")

            try:
                # Find the position of our anomaly within the REGION-SPECIFIC features dataframe
                idx_in_region_shap = features_df_for_region.index.get_loc(event_index)

                # Build the explanation object using the REGION-SPECIFIC data
                explanation_object = shap.Explanation(
                    values=shap_values_for_region.values[idx_in_region_shap, :],
                    base_values=explainer.expected_value,
                    data=features_df_for_region.iloc[idx_in_region_shap, :],
                    feature_names=features_df_for_region.columns.tolist()
                )

                # Create and save the waterfall plot
                plt.figure(figsize=(12, 8), dpi=200)
                shap.waterfall_plot(explanation_object, show=False)
                plt.tight_layout()
                plt.savefig(f"anomaly_{i}_{region}_index_{event_index}_shap_waterfall.svg", bbox_inches='tight', format='svg')
                plt.show()
                plt.close()

            except KeyError:
                print(f"Error: Index {event_index} not found in the {region} SHAP dataset. This is unexpected but can happen.")

    # --- Step 3: Explore feature relationships with SHAP Dependence Plot (Per Region) ---

    # Loop through each region to create a separate set of plots
    for region in df['region'].unique():
        print(f"\n--- Creating SHAP Dependence Plots for {region} ---")

        # The correct way to check if we have data to plot for this region
        if region not in all_shap_values:
            print(f"Skipping dependence plots for {region} as no SHAP values were calculated.")
            continue

        # Retrieve the correct, region-specific SHAP values from our dictionary
        # The SHAP Explanation object is the modern way to handle this data
        shap_values_for_region = all_shap_values[region]

        # --- Feature 1: Temperature (heat_index_celsius) for this region ---
        print(f"  Plotting relationship for Heat Index in {region}...")

        shap.plots.scatter(
            shap_values_for_region[:, "heat_index_celsius"],
            color=shap_values_for_region[:, "demand_MW_rolling_mean_24h"],
            show=False
        )

        # Get the current figure and save it with a region-specific name
        fig = plt.gcf()
        fig.tight_layout()
        output_filename = f'shap_dependence_heat_index_{region}.svg'
        print(f"  Saving plot to: {output_filename}")
        fig.savefig(output_filename, bbox_inches='tight', format='svg')

        # Show and then close the figure to prepare for the next plot
        plt.show()
        plt.close(fig)

        # --- Feature 2: Demand Volatility for this region ---
        print(f"  Plotting relationship for Demand Volatility in {region}...")

        shap.plots.scatter(
            shap_values_for_region[:, "demand_MW_rolling_std_24h"],
            color=shap_values_for_region[:, "temp_celsius_lag_24h"],
            show=False
        )

        # Get the current figure and save it with a region-specific name
        fig = plt.gcf()
        fig.tight_layout()
        output_filename_2 = f'shap_dependence_demand_std_{region}.svg'
        print(f"  Saving plot to: {output_filename_2}")
        fig.savefig(output_filename_2, bbox_inches='tight', format='svg')

        # Show and close the final plot for this region
        plt.show()
        plt.close(fig)

if __name__ == "__main__":
    deep_analyze_anomalies()