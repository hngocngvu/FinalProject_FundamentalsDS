import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd
from s5_run_models import model_cols, all_shap_values, all_features_df
from s3_save_data import save_data_pipeline

df = save_data_pipeline()
def run_shap():
    def plot_anomalies_by_region(df: pd.DataFrame, model_cols: list, output_filename: str):
        """Plot the demand and outliers for each region."""
        regions = df['region'].unique()
        n_regions = len(regions)

        fig, axes = plt.subplots(n_regions, 1, figsize=(20, 8 * n_regions), sharex=True)
        if n_regions == 1: axes = [axes]

        colors = ['red', 'purple', 'green', 'orange']
        markers = ['o', 'X', 'P', 's']

        for i, region in enumerate(regions):
            ax = axes[i]
            region_df = df[df['region'] == region]

            sns.lineplot(x='datetime', y='demand_MW', data=region_df, ax=ax, color='lightblue', label='Demand', zorder=1)

            for idx, col in enumerate(model_cols):
                anomalies = region_df[region_df[col] == 1]
                ax.scatter(anomalies['datetime'], anomalies['demand_MW'],
                        color=colors[idx % len(colors)],
                        s=50,
                        label=f'Anomaly ({col})',
                        marker=markers[idx % len(markers)],
                        zorder=2)

            ax.set_title(f'Anomaly Detection in Demand: {region}', fontsize=16)
            ax.set_ylabel('Demand (MW)')
            ax.legend()
            ax.grid(True)

        plt.xlabel('Time', fontsize=12)
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=2))
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.savefig(output_filename, format='png', bbox_inches='tight')
        plt.show()

    def plot_shap_summary(shap_values, features_df: pd.DataFrame, output_filename):
        """Plot SHAP summary."""
        if shap_values is None: return

        print("Displaying SHAP summary plot for anomalies...")
        fig = plt.figure(figsize=(12, 8), dpi=200)
        shap.summary_plot(shap_values, features_df, plot_type="bar", show=False)
        fig = plt.gcf()
        fig.set_size_inches(12, 6)
        plt.tight_layout()
        plt.savefig(output_filename, format='png', bbox_inches='tight')
        plt.show()

    # Run visualizations
    plot_anomalies_by_region(df, model_cols, output_filename='anomalies_by_region.png')

    plot_df = df.copy()
    plot_df.rename(columns={'ensemble_final_anomaly': 'Final Anomaly (Weighted)'}, inplace=True)
    plot_anomalies_by_region(plot_df, ['Final Anomaly (Weighted)'], output_filename='final_anomalies_by_region.png')

    for region in df['region'].unique():
        if region in all_shap_values:
            print(f"\n--- SHAP Summary Plot for {region} ---")

            # Get the specific SHAP values and features for this region
            region_shap_values = all_shap_values[region]
            region_features_df = all_features_df[region]

            # Create the plot with a region-specific filename
            plot_shap_summary(
                region_shap_values,
                region_features_df,
                f'shap_summary_{region}.png'
            )
        else:
            print(f"\n--- No SHAP values to plot for {region} ---")

if __name__ == "__main__":
    run_shap()
