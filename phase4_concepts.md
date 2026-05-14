# Phase 4: Production & Monitoring Concepts

This document explains the MLOps engineering principles applied *after* a model is shipped to production.

## 1. Silent Failure and Model Decay
**The Concept**: Unlike traditional software, ML models do not throw errors when the world changes; they just make bad predictions. If ad formats change, or a new demographic starts visiting the website, the distribution of features fed into the model will shift. This is called **Covariate Shift** (Data Drift).

**The Solution**: We must actively monitor the statistical distributions of our incoming production data and compare it against the baseline data the model was trained on.

## 2. Statistical Drift Detection with Evidently (`steps/drift_steps.py`)
**The Concept**: You cannot just eyeball millions of rows of data to see if it changed. You need mathematical proof.

**The Solution**: We use **Evidently**. For every single feature, Evidently runs specific statistical tests:
- For numerical features (like `C14`), it uses tests like **Wasserstein distance** or **Kolmogorov-Smirnov**.
- For categorical features (like `device_type`), it uses tests like **Chi-Square**.
If the statistical distribution of the feature in the "Current" data differs significantly from the "Reference" data, the feature is marked as "Drifted". If too many features drift, an alert should be fired to trigger model retraining.

## 3. Automated Retraining (Continual Learning)
**The Concept**: When drift is detected, human intervention is too slow. The system should automatically self-heal by retraining the model on the most recent data.

**The Solution**: We build a "Continual Learning" script (`run_continual_learning.py`) that acts as an orchestrator. It runs the drift pipeline, dynamically inspects the Evidently report, and if `dataset_drift` is `True`, it programmatically triggers the `ad_click_training_pipeline()`. This creates a closed-loop MLOps system.