# FinalProject_FundamentalsDS
Detecting Anomalies in Regional
Electricity Usage: Case Study of California and Texas

Description: This study examines electricity demand anomalies in two U.S. regions: ERCOT (Texas) and
CAISO (California) by applying three unsupervised machine learning models: Local Outlier
Factor (LOF), DBSCAN, and Isolation Forest. A total of nineteen engineered features, including temporal, meteorological, and rolling statistical indicators, were utilized to identify abnormal demand patterns from 2021 to 2024. Hyperparameter tuning was conducted using
Optuna for DBSCAN, and SHAP analysis was employed to interpret the models’ decision
mechanisms.

Members: Nguyen Vu Hong Ngoc, Hoang Khanh Dong, Pham Quang Vinh

Instruction:

## Install required libraries for this project
```bash
pip install -r requirements.txt
```

## Get your api key from EIA
[https://www.eia.gov/opendata/register.php](https://www.eia.gov/opendata/register.php)

Run the file get_api_key.py to generate the .env file

```bash
python get_api_key.py
```

Finally, open and update your API key inside the .env file:

```bash
nano ../.env
```
Press CTRL + O to save and CTRL + X to exit nano.

## Run the pipeline

Run the file pipeline.py to crawl data, create additional features for the dataset, run the models and evaluate them
```bash
python pipeline.py
```
If you want to view exploratory and explainability plots:

```bash
python s4_eda.py
python s7_shap_analysis.py
python s8_examine.py
```