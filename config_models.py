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

def tune_dbscan_hyperparameters(
    df: pd.DataFrame, region_name: str, features: list, n_trials: int = 50, random_seed: int = 42
) -> dict:
    """
    Use Optuna to find the best DBSCAN hyperparameters for a specific region.
    Returns dict: {'eps': ..., 'min_samples': ...}
    """
    print(f"\n--- Start hyperparameter tuning for DBSCAN in region {region_name} ---")

    # 1️⃣ Scale features for this region
    region_features = df[features]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(region_features)

    # 2️⃣ Define Optuna objective function
    def objective(trial):
        eps = trial.suggest_float("eps", 0.5, 5.0, log=True)
        min_samples = trial.suggest_int("min_samples", 5, 150)

        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1)  # n_jobs=1 for deterministic
        labels = model.fit_predict(data_scaled)

        # Silhouette score requires at least 2 clusters
        if len(set(labels)) < 2 or all(l == -1 for l in labels):
            return -1.0  # worst score for Optuna

        # Compute silhouette score
        score = silhouette_score(data_scaled, labels, sample_size=min(10000, len(labels)), random_state=random_seed)
        return score

    # 3️⃣ Run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"--- Complete fine-tuning for {region_name} ---")
    print(f"  Best Silhouette Score: {study.best_value:.4f}")
    print(f"  Best parameters: {study.best_params}")

    return study.best_params

def run_dbscan(df: pd.DataFrame, features: list, eps=1.2, min_samples=5) -> pd.Series:
    """
    NOTICE: DBSCAN is very sensitive to hyperparameters (eps, min_samples).
    """
    # Scale region features
    region_features = df[features]  # fill missing values
    scaler = StandardScaler()
    region_scaled = scaler.fit_transform(region_features)
    
    # Fit DBSCAN
    model = DBSCAN(eps=eps, min_samples=min_samples)
    predictions = model.fit_predict(region_scaled)
    # Convert -1 (noise/outlier) to 1, and others to 0
    return pd.Series(predictions, index=df.index).apply(lambda x: 1 if x == -1 else 0)

from sklearn.ensemble import IsolationForest

def run_isolation_forest(df: pd.DataFrame, features: list, contamination=0.01) -> pd.Series:
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    predictions = model.fit_predict(df[features])
    return pd.Series(predictions, index=df.index).apply(lambda x: 1 if x == -1 else 0)