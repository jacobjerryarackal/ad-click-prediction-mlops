# Architecture: Ad Click Prediction

## MLOps Pipeline Overview
1. **Data Ingestion** → 2. **Data Validation** → 3. **Feature Engineering** → 4. **Model Training** → 5. **Model Evaluation** → 6. **Model Registry** → 7. **Deployment** → 8. **Monitoring** → 9. **Drift Detection** → 10. **Retraining Trigger**

> **Note:** For deep details on production scaling (10M requests/day), latency budgets (<50ms), and deployment architecture (Redis, BentoML, Shadow Deployments), please see the **System Design Document**.

## Data Plan
- **Ingestion**: Read from `/data/dataset.csv`. Use the entire dataset for robustness.
- **Versioning**: Track datasets automatically via ZenML Artifacts.
- **Validation**: Evidently data quality and schema checks to fail fast on invalid inputs (missing columns, unexpected null rates).

## Feature Engineering Plan
- **Time-Based Split**: Sort by the `hour` column to split chronologically. Do NOT use a random split to avoid future-to-past data leakage.
- **Preprocessing Bundling**: `sklearn.Pipeline` will wrap imputation, encoding, and scaling to guarantee identical transformations in training and serving, preventing training-serving skew.
- **Categorical Handling**: High-cardinality ID features (`device_ip`, `device_id`) will be dropped in the MVP baseline to reduce dimensionality and overfitting. Low-cardinality categoricals will be One-Hot Encoded.

## Training & Evaluation Plan
- **Baseline Model**: Logistic Regression.
- **Candidate Models**: LightGBM or XGBoost (high performance for tabular/categorical data).
- **Experiment Tracking**: MLflow.
- **Evaluation Strategy**: Holdout test set evaluating **Precision** (primary), **LogLoss**, and **AUC**.

## Deployment Plan
- **Type**: Both **Batch Inference** and **Real-Time API**.
- **Strategy**: Direct deployment where Staging models are promoted to Production if evaluation metrics surpass the existing Production model.
- **Rollback**: Quickly revert by updating MLflow Registry aliases.

## Monitoring & Drift Plan
- **Drift Detection**: Evidently reports to monitor data drift on incoming features versus the training baseline.
- **Alerting**: Logging and alerts upon breaching drift thresholds.

## Versioning & Governance
- **Model Registry**: MLflow Model Registry for versioning.
- **Audit Trail**: Metadata capturing git commit, training data version, metrics, and parameters for each logged model.

## ZenML Stack Specification
| Component | Choice | Why |
|-----------|--------|-----|
| Orchestrator | Local | Simple and efficient starting point; easily swapped to cloud later. |
| Artifact Store | Local | Local directory to store intermediate outputs. |
| Experiment Tracker | MLflow | Industry standard, tracks metrics, parameters, and models natively. |
| Model Registry | MLflow | Manages lifecycle states (Staging vs. Production). |
| Data Validator | Evidently | Generates automated data quality and drift profiles. |
| Model Deployer | MLflow | Simplifies serving the model as a real-time API or in batch jobs. |

## Pipeline Decomposition
- **Training Pipeline**: Ingest → Validate → Preprocess & Train → Evaluate → Register Model
- **Batch Inference Pipeline**: Load Model → Preprocess Input → Predict → Save Predictions
- **Real-Time API**: Endpoint serving the registered MLflow production model

## Project Structure
- `core/`: Pure Python logic (preprocessing, validation) independent of framework logic
- `steps/`: ZenML framework steps wrapping the `core/` functions
- `pipelines/`: ZenML pipeline definitions
- `tests/`: Tests for logic

## MVP Scope
1. **MVP v1**: Set up basic `core/` structure, ingest data, and train Logistic Regression.
2. **MVP v2**: Introduce LightGBM/XGBoost, log to MLflow, and save in Model Registry.
3. **MVP v3**: Deploy real-time API and batch script.
