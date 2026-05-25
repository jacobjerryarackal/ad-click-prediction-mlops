# Problem Statement: Ad Click Prediction

## Business Context
Predict whether a user will click on an online ad. Accurate click prediction helps improve ad targeting, reduces wasted ad spend (by not showing ads to low-intent users), increases campaign ROI, and improves overall marketing performance for digital marketing teams, advertising platforms, and e-commerce companies.

## ML Formulation
- **Problem type**: Binary Classification
- **Target variable**: `click` (0 = Not Clicked, 1 = Clicked)
- **Primary metric**: **Precision** — because false positives (showing ads to low-intent users) result in wasted ad spend.
- **Guardrail metrics**:
  - **LogLoss**: For probability calibration (critical for expected value calculation and bidding).
  - **AUC**: For general ranking quality.
- **Current baseline**: No current baseline (the first simple model will establish it).

## Data Summary
- **Rows**: ~500,000+ (entire dataset to be used)
- **Features**: Includes categorical/ID features (`hour`, `C1`, `banner_pos`, `site_id`, `site_domain`, `site_category`, `app_id`, `app_domain`, `app_category`, `device_id`, `device_ip`, `device_model`, `device_type`, `device_conn_type`, and anonymized features `C14`-`C21`).
- **Label availability**: Yes (`click` column).
- **Known issues**: High cardinality in categorical columns (e.g., `device_ip`, `site_domain`), likely class imbalance.

## Constraints
- **Latency**: Both Batch processing and Real-time API predictions.
- **Interpretability**: Explain everything simply but deeply.
- **Production**: Optimize for production readiness.

## Framework
- **Orchestration**: ZenML

## Success Criteria
A production-grade ML system deployed with training, evaluation, and inference pipelines, with monitoring capability to prevent silent failures.
