# Exploratory Data Analysis Report

## 1. Target Distribution (`click`)
Class imbalance tells us if the dataset is skewed towards one class.

- **Non-clicks (0)**: 83.59%
- **Clicks (1)**: 16.41%

> **Insight**: A highly imbalanced dataset means a model predicting '0' all the time could have high accuracy. This is why we optimize for Precision and LogLoss.

## 2. Missing Values
No missing values detected (Note: Categorical datasets often encode missing as '-1' or 'unknown').

## 3. Feature Cardinality
Cardinality is the count of unique values. High cardinality features (e.g., `device_ip` with thousands of unique values) often require Target Encoding or Hashing, or should be dropped in the baseline to prevent overfitting.

| Feature | Unique Values | Type |
|---------|---------------|------|
| click | 2 | Context/Target |
| hour | 4 | Context/Target |
| C1 | 7 | Categorical (Low/Mid) |
| banner_pos | 6 | Categorical (Low/Mid) |
| site_id | 1704 | High Cardinality ID |
| site_domain | 1586 | High Cardinality ID |
| site_category | 21 | Categorical (Low/Mid) |
| app_id | 1641 | High Cardinality ID |
| app_domain | 122 | Categorical (Low/Mid) |
| app_category | 20 | Categorical (Low/Mid) |
| device_id | 41413 | High Cardinality ID |
| device_ip | 171304 | High Cardinality ID |
| device_model | 3967 | High Cardinality ID |
| device_type | 4 | Categorical (Low/Mid) |
| device_conn_type | 4 | Categorical (Low/Mid) |
| C14 | 540 | Categorical (Low/Mid) |
| C15 | 8 | Categorical (Low/Mid) |
| C16 | 9 | Categorical (Low/Mid) |
| C17 | 154 | Categorical (Low/Mid) |
| C18 | 4 | Categorical (Low/Mid) |
| C19 | 40 | Categorical (Low/Mid) |
| C20 | 154 | Categorical (Low/Mid) |
| C21 | 34 | Categorical (Low/Mid) |
