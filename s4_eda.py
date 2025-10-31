import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, zscore
from IPython.display import display
from s3_save_data import df

print(f"Loaded dataset with {len(df)} rows.")

def eda():


    # 1. Distribution Analysis using a KDE Plot (Best for comparing with hue)
    plt.figure(figsize=(12, 6), dpi=200)
    sns.kdeplot(data=df, x='demand_MW', hue='region', fill=True, common_norm=False)
    plt.title('Distribution of demand_MW by Region')
    plt.xlabel('Demand (MW)')
    plt.ylabel('Density')
    plt.savefig('distribution_of_demand_MW.png', format='png', bbox_inches='tight')
    plt.show()


    # 2. Box plot to identify outliers by region
    plt.figure(figsize=(12, 6), dpi=200)
    sns.boxplot(data=df, x='demand_MW', y='region')
    plt.title('Box Plot of demand_MW by Region')
    plt.xlabel('Demand (MW)')
    plt.savefig('box_plot_of_demand_MW.png', format='png', bbox_inches='tight')
    plt.show()

    # 3. Time series plot with hue
    plt.figure(figsize=(18, 6), dpi=200)
    sns.lineplot(data=df, x='datetime', y='demand_MW', hue='region')
    plt.title('Time Series of demand_MW by Region')
    plt.xlabel('Time')
    plt.ylabel('Demand (MW)')
    plt.legend(title='Region')
    plt.grid(True)
    # This will format the x-axis to show each quarter
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    plt.savefig('time_series_of_demand_MW.png', format='png', bbox_inches='tight')
    plt.show()


    # 4. Statistical Tests (Calculated per region for more accurate insights)
    for region in df['region'].unique():
        print(f"\n--- Statistical Analysis for {region} ---")
        region_df = df[df['region'] == region]

        # Skewness and Kurtosis
        skewness = skew(region_df['demand_MW'].dropna())
        kurt = kurtosis(region_df['demand_MW'].dropna())
        print(f"Skewness of demand_MW: {skewness:.2f}")
        print(f"Kurtosis of demand_MW: {kurt:.2f}")

        # IQR method for outlier detection
        Q1 = region_df['demand_MW'].quantile(0.25)
        Q3 = region_df['demand_MW'].quantile(0.75)
        IQR = Q3 - Q1
        outliers_iqr = region_df[(region_df['demand_MW'] < (Q1 - 1.5 * IQR)) | (region_df['demand_MW'] > (Q3 - 1.5 * IQR))]
        print(f"Number of outliers detected by IQR method: {len(outliers_iqr)} ({len(outliers_iqr)/len(region_df)*100:.2f}%)")

    # Create side-by-side boxplots for each region
    plt.figure(figsize=(18, 8), dpi=200)
    sns.boxplot(data=df, x='hour', y='demand_MW', hue='region')
    plt.title('Demand Distribution by Hour of the Day for Each Region')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Demand (MW)')
    plt.grid(True)
    plt.legend(title='Region')
    plt.savefig('demand_distribution_by_hour_of_the_day.png', format='png', bbox_inches='tight')
    plt.show()

    # Determine which hours have the most outliers, calculated per region
    print("Hourly outlier counts by region:")
    for region in df['region'].unique():
        print(f"\n--- Region: {region} ---")
        region_df = df[df['region'] == region]
        hourly_outliers = []
        for hour in range(24):
            hourly_data = region_df[region_df['hour'] == hour]['demand_MW']
            Q1 = hourly_data.quantile(0.25)
            Q3 = hourly_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            num_outliers = hourly_data[(hourly_data < lower_bound) | (hourly_data > upper_bound)].count()
            hourly_outliers.append({'hour': hour, 'outliers': num_outliers})

        outlier_counts = pd.DataFrame(hourly_outliers)
        # Display the top 5 hours with the most outliers for the current region
        print(outlier_counts.sort_values(by='outliers', ascending=False).head(5))

    # Scatter plot of temperature vs. demand, colored by region
    plt.figure(figsize=(12, 7), dpi=200)
    sns.scatterplot(data=df, x='temp_celsius', y='demand_MW', hue='region', alpha=0.6)
    plt.title('Temperature vs. Demand by Region')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Demand (MW)')
    plt.grid(True)
    plt.legend(title='Region')
    plt.savefig('temperature_vs_demand.png', format='png', bbox_inches='tight')
    plt.show()

    # Analyze extreme temperatures and demand separately for each region
    for region in df['region'].unique():
        print(f"\n--- Analysis for {region} ---")
        region_df = df[df['region'] == region]

        # Identify extreme temperatures using percentiles for this region
        lower_threshold = region_df['temp_celsius'].quantile(0.05)
        upper_threshold = region_df['temp_celsius'].quantile(0.95)

        print(f"5th percentile (extreme cold): {lower_threshold:.2f}°C")
        print(f"95th percentile (extreme heat): {upper_threshold:.2f}°C")

        # Analyze demand during extreme temperatures for this region
        extreme_cold_demand = region_df[region_df['temp_celsius'] <= lower_threshold]['demand_MW'].mean()
        extreme_heat_demand = region_df[region_df['temp_celsius'] >= upper_threshold]['demand_MW'].mean()
        normal_demand = region_df[(region_df['temp_celsius'] > lower_threshold) & (region_df['temp_celsius'] < upper_threshold)]['demand_MW'].mean()

        print(f"Average demand during extreme cold: {extreme_cold_demand:.2f} MW")
        print(f"Average demand during extreme heat: {extreme_heat_demand:.2f} MW")
        print(f"Average demand during normal temperatures: {normal_demand:.2f} MW")

    # Scatter plot to compare demand with its 24-hour lag, colored by region
    plt.figure(figsize=(10, 10), dpi=200)
    sns.scatterplot(data=df, x='demand_MW_lag_24h', y='demand_MW', hue='region', alpha=0.5)
    # Add a reference line for no change
    min_val = df[['demand_MW', 'demand_MW_lag_24h']].min().min()
    max_val = df[['demand_MW', 'demand_MW_lag_24h']].max().max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='No Change')
    plt.title('Demand vs. 24-Hour Lagged Demand by Region')
    plt.xlabel('Demand (MW) 24 hours ago')
    plt.ylabel('Current Demand (MW)')
    plt.grid(True)
    plt.legend(title='Region')
    plt.savefig('demand_vs_24-hour_lagged_demand.png', format='png', bbox_inches='tight')
    plt.show()

    # Calculate the difference to find sudden changes
    df['demand_change_24h'] = df['demand_MW'] - df['demand_MW_lag_24h']

    # Find the largest sudden increases and decreases for each region
    for region in df['region'].unique():
        print(f"\n--- Sudden Demand Changes for {region} ---")
        region_df = df[df['region'] == region].copy()

        top_5_increases = region_df.sort_values(by='demand_change_24h', ascending=False).head(5)
        top_5_decreases = region_df.sort_values(by='demand_change_24h', ascending=True).head(5)

        print("Top 5 largest sudden increases in demand (over 24 hours):")
        display(top_5_increases[['datetime', 'demand_MW', 'demand_MW_lag_24h', 'demand_change_24h']])

        print("\nTop 5 largest sudden decreases in demand (over 24 hours):")
        display(top_5_decreases[['datetime', 'demand_MW', 'demand_MW_lag_24h', 'demand_change_24h']])

    from scipy.stats import ttest_ind

    # Violin plot to compare demand distribution, split by region
    plt.figure(figsize=(12, 7), dpi=200)
    sns.violinplot(data=df, x='is_weekend', y='demand_MW', hue='region')
    plt.title('Demand Distribution: Weekday vs. Weekend by Region')
    plt.xticks([0, 1], ['Weekday', 'Weekend'])
    plt.xlabel('')
    plt.ylabel('Demand (MW)')
    plt.legend(title='Region')
    plt.savefig('demand_distribution:_weekday_vs_weekend.png', format='png', bbox_inches='tight')
    plt.show()

    # Perform an independent t-test for each region to see if the difference is significant
    for region in df['region'].unique():
        print(f"\n--- T-test for {region} ---")
        region_df = df[df['region'] == region]

        # Separate weekday and weekend data for the current region
        weekday_demand = region_df[region_df['is_weekend'] == 0]['demand_MW'].dropna()
        weekend_demand = region_df[region_df['is_weekend'] == 1]['demand_MW'].dropna()

        # Perform Welch's t-test
        t_stat, p_value = ttest_ind(weekday_demand, weekend_demand, equal_var=False)

        print(f"T-test results for comparing weekday and weekend demand in {region}:")
        print(f"  T-statistic: {t_stat:.2f}")
        print(f"  P-value: {p_value: .4f}")

        if p_value < 0.05:
            print("  => The difference in mean demand between weekdays and weekends is statistically significant.")
        else:
            print("  => There is no statistically significant difference in mean demand between weekdays and weekends.")

if __name__ == "__main__":
    eda()