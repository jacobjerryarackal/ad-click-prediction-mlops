# Phase 3: Data Pipeline Concepts

This document explains the MLOps engineering principles applied in the data ingestion and preprocessing steps of the Ad Click Prediction system.

## 1. Fail-Fast Data Validation (`steps/load_data.py`)
**The Concept**: Garbage data silently corrupts machine learning models. If a data source drops the target column, or if a bug causes 100% of a feature to become `Null`, a normal script will still happily train a useless model.

**The Solution**: We implemented a "fail-fast" pure Python validation check immediately after loading the data. If the dataset does not meet strict schema requirements (like containing the `click` and `hour` columns, or not being empty), the pipeline crashes intentionally. This prevents bad data from ever reaching the model.

## 2. Preventing Data Leakage (`steps/split_data.py`)
**The Concept**: "Data Leakage" happens when the model gets hints about the test set during training. For example, if we use a random `train_test_split` on ad-click data, the model might see User A's behavior at 5:00 PM in the training set, and predict their behavior at 4:00 PM in the test set. It is effectively looking into the future to predict the past. 

**The Solution**: We implemented a strict **Chronological Split**. We sort the data by the `hour` column, and assign the first 80% of rows to the training set, and the final 20% to the testing set. This perfectly mimics production, where we train on the past to predict the future.

## 3. Preventing Training-Serving Skew (`steps/preprocessing.py`)
**The Concept**: "Training-Serving Skew" is the #1 cause of silent model failure in production. It occurs when the Python code used to transform data during *training* differs slightly from the code used to transform data when a user hits the *live prediction API*. 

**The Solution**: Instead of using manual Pandas transformations (`df['col'].fillna(...)`), we bundle all transformations (One-Hot Encoding, dropping high-cardinality columns) into an `sklearn.compose.ColumnTransformer`. 

In the future, this preprocessor object will be saved as an artifact. When the real-time API receives a single JSON request from an ad-server, it passes it through this exact same preprocessor artifact, guaranteeing 100% identical transformations.

## 4. Separation of Concerns
We separated Data Splitting from Preprocessing. In ZenML, each step should have one responsibility. This allows ZenML to cache the "Split Data" artifact independently. If we want to try a new preprocessing method tomorrow, ZenML won't have to re-split the data—it will just load the cached split and run the new preprocessor, saving time and compute.