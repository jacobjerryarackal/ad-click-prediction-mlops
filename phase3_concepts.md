# Phase 3: Implementation Concepts

This document explains the MLOps engineering principles applied in the implementation phase of the Ad Click Prediction system.

## 1. Fail-Fast Data Validation (`steps/load_data.py`)
**The Concept**: Garbage data silently corrupts machine learning models. If a data source drops the target column, or if a bug causes 100% of a feature to become `Null`, a normal script will still happily train a useless model.

**The Solution**: We implemented a "fail-fast" pure Python validation check immediately after loading the data. If the dataset does not meet strict schema requirements, the pipeline crashes intentionally. This prevents bad data from ever reaching the model.

## 2. Preventing Data Leakage (`steps/split_data.py`)
**The Concept**: "Data Leakage" happens when the model gets hints about the test set during training. For example, if we use a random `train_test_split` on ad-click data, the model might see User A's behavior at 5:00 PM in the training set, and predict their behavior at 4:00 PM in the test set. 

**The Solution**: We implemented a strict **Chronological Split**. We sort the data by the `hour` column, and assign the first 80% of rows to the training set, and the final 20% to the testing set. This mimics production, where we train on the past to predict the future.

## 3. Preventing Training-Serving Skew (`steps/preprocessing.py`)
**The Concept**: "Training-Serving Skew" is the #1 cause of silent model failure in production. It occurs when the Python code used to transform data during *training* differs slightly from the code used to transform data when a user hits the *live prediction API*. 

**The Solution**: Instead of using manual Pandas transformations, we bundle all transformations (One-Hot Encoding, dropping high-cardinality columns) into an `sklearn.compose.ColumnTransformer`. When the real-time API receives a request, it passes it through this exact same preprocessor artifact, guaranteeing 100% identical transformations.

## 4. Separation of Concerns
We separated Data Splitting from Preprocessing. In ZenML, each step should have one responsibility. This allows ZenML to cache the "Split Data" artifact independently. 

## 5. The Baseline Model & Reproducibility (`steps/train_model.py`)
**The Concept**: Never start with a complex model (like XGBoost or Neural Networks). You need a performance "floor" to measure against. Additionally, models must be strictly reproducible. 

**The Solution**: We start our training pipeline with a simple `LogisticRegression`. We also pin the `random_state` so the model converges identically every time it runs on the same data. By passing `class_weight='balanced'`, we natively handle the class imbalance common in ad-click datasets. This acts as our Baseline Model.

## 6. The DAG Pipeline (`pipelines/training_pipeline.py`)
**The Concept**: Notebooks execute top-to-bottom and hold hidden state in memory. Production ML requires a Directed Acyclic Graph (DAG) where outputs of one step explicitly become inputs to the next.

**The Solution**: We use ZenML's `@pipeline` decorator to explicitly wire our isolated steps together:
`Load & Validate` -> `Split` -> `Preprocess` -> `Train Model`
This guarantees execution order and allows the orchestrator to track the lineage of every artifact.

## 7. Honest Evaluation (`steps/evaluate_model.py`)
**The Concept**: A model with 99% accuracy is useless if it predicts "not clicked" for everyone in an imbalanced dataset. We must measure metrics that tie directly to business costs.

**The Solution**: We evaluate **Precision** (to minimize false positives and wasted ad spend), **LogLoss** (to ensure probabilities are well-calibrated for the bidding engine), and **AUC** (to measure general ranking quality). We strictly evaluate on the `X_test` holdout set that was never seen during preprocessing fitting or model training.

## 8. Experiment Tracking & Model Registry (MVP v2)
**The Concept**: If you run a script and print the metrics to the terminal, those results are lost forever. When you try to reproduce the best model next week, you won't remember which hyperparameters or data split you used.

**The Solution**: We integrated **MLflow** and ZenML's **Model Control Plane (MCP)**. 
1. By attaching `@pipeline(model=Model(...))`, ZenML acts as a bookshelf. Every pipeline run automatically creates a new "Model Version" on the shelf.
2. By using `@step(experiment_tracker="mlflow_tracker")` and `mlflow.sklearn.autolog()`, every hyperparameter (like `class_weight='balanced'`) is automatically recorded without writing manual logging code.
3. By using `log_metadata()`, our evaluation metrics (Precision, LogLoss, AUC) are permanently attached to the specific model version in the registry. 

Now, every model is perfectly reproducible and comparable.

## 9. The Candidate Model: XGBoost
**The Concept**: Linear models (like Logistic Regression) struggle to capture complex, non-linear interactions in tabular data (e.g., clicking behavior varying wildly depending on the interaction between `device_type` and `hour`).

**The Solution**: We introduce **XGBoost**, an advanced gradient boosting ensemble that builds decision trees sequentially to correct prior errors. It is the industry standard for tabular data.

**Handling Imbalance**: Instead of `class_weight`, XGBoost uses `scale_pos_weight`. We dynamically calculate this as `count(negative_samples) / count(positive_samples)` to ensure the trees don't ignore the minority "clicked" class.