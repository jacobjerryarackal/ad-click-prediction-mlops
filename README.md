# Ad Click Prediction MLOps

An enterprise-grade MLOps pipeline for predicting Ad Clicks, built with **ZenML, MLflow, Evidently, XGBoost, and FastAPI**.

## Project Overview
This repository implements a complete machine learning lifecycle for an Ad-Tech real-time bidding use case. It demonstrates how to transition from raw data to a hardened, automated, and self-monitoring ML system that generates sub-50ms predictions to power ad exchanges.

## Key Features
* **Robust Pipelines:** Orchestrated via ZenML, including chronologically split data to prevent leakage.
* **Advanced Feature Engineering:** Utilizing Target Encoding inside `sklearn.Pipeline` to handle high-cardinality features (like IP addresses) without training-serving skew.
* **Experiment Tracking & Model Registry:** Powered by MLflow to track XGBoost hyperparameters, log metrics (Precision, AUC, LogLoss), and version models.
* **Continual Learning & Drift Detection:** Evidently monitors data distributions, triggering automatic retraining pipelines when covariate shift is detected.
* **Real-Time & Batch Serving:** 
  * A highly optimized `FastAPI` service with Pydantic-enforced Data Contracts.
  * A batch inference pipeline for large-scale offline scoring.
* **Agent-Ready Quality Gates:** Enforced `pytest`, `mypy`, `ruff`, and `pre-commit` hooks for an anti-slop engineering environment.

## System Architecture & Documentation
The conceptual design, decisions, and system constraints are fully documented:
* `architecture.md` - MLOps Pipeline and Data Design
* `system_design.md` - Infrastructure, Scaling, and Latency Strategies
* `mlops_implementation_concepts.md` - Core ML Engineering principles applied
* `mlops_production_concepts.md` - Post-deployment monitoring and retraining logic
* `agent-workflow.md` - CI/CD and AI Agent contribution guidelines

## How to Run

### 1. Training & Registry
```bash
python run_pipeline.py
```

### 2. Batch Inference
```bash
python run_inference.py
```

### 3. Real-Time API & Streamlit UI
**Terminal 1 (Backend):**
```bash
python serve_api.py
```
**Terminal 2 (Frontend):**
```bash
streamlit run streamlit_app.py
```

### 4. Automated Retraining (Drift Check)
```bash
python run_continual_learning.py
```
