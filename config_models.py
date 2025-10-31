import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import optuna
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

def run_lof(df: pd.DataFrame, features: list, contamination=0.01) -> pd.Series:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])

    model = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=False)
    predictions = model.fit_predict(scaled_features)
    # Convert -1 (outlier) to 1, and 1 (inlier) to 0
    return pd.Series(predictions, index=df.index).apply(lambda x: 1 if x == -1 else 0)

def tune_dbscan_hyperparameters(df: pd.DataFrame, region_name: str, features: list, n_trials: int = 50) -> dict:
    """
    Use Optuna to find the best hyperparameters for DBSCAN on a specific region.
    """
    print(f"\n--- Start hyperparameter tuning for DBSCAN in region {region_name} ---")

    # 1. Normalize features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[features])

    # 2. Define the objective function for Optuna
    # Silhouette Score measure how well clusters are separated. Higher is better.
    def objective(trial):
        eps = trial.suggest_float('eps', 0.5, 5.0, log=True) # Find eps in log scale
        min_samples = trial.suggest_int('min_samples', 5, 150)

        # Run DBSCAN with suggested parameters
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=2)
        labels = model.fit_predict(data_scaled)

        # Handle case where DBSCAN finds no clusters (all noise or single cluster)
        # Silhouette Score requires at least 2 clusters to compute.
        if len(set(labels)) < 2:
            return -1.0  # Return worst score so Optuna knows this is a bad choice

        # Calculate and return Silhouette Score
        score = silhouette_score(data_scaled, labels, sample_size=10000, random_state=42)
        return score

    # 3. Run Optuna study
    # 'direction="maximize"' since we want silhouette score to be as high as possible
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"--- Complete fine-tuning for {region_name} ---")
    print(f"  Best Silhouette Score: {study.best_value:.4f}")
    print(f"  The best parameters: {study.best_params}")

    return study.best_params

def run_dbscan(df: pd.DataFrame, features: list, eps=1.2, min_samples=5) -> pd.Series:
    """
    NOTICE: DBSCAN is very sensitive to hyperparameters (eps, min_samples).
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])

    model = DBSCAN(eps=eps, min_samples=min_samples)
    predictions = model.fit_predict(scaled_features)
    # Convert -1 (noise/outlier) to 1, and others to 0
    return pd.Series(predictions, index=df.index).apply(lambda x: 1 if x == -1 else 0)

from sklearn.ensemble import IsolationForest

def run_isolation_forest(df: pd.DataFrame, features: list, contamination=0.01) -> pd.Series:
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    predictions = model.fit_predict(df[features])
    return pd.Series(predictions, index=df.index).apply(lambda x: 1 if x == -1 else 0)