# Production System Design: Ad Click Predictor

This document outlines the infrastructure and system design required to scale the Ad Click Predictor to production ad-tech volumes.

## 1. Requirements & Constraints
- **Traffic Scale**: 10,000,000 requests per day.
- **Throughput**: ~115 average QPS (Queries Per Second), ~350 peak QPS.
- **Latency Budget**: < 50ms at P99 (Strict ad-exchange requirement).
- **Availability**: 99.99% (~52 minutes of allowed downtime per year).
- **Feature Freshness**: Target encodings can be updated via a nightly batch (real-time streaming computation is not required).

## 2. High-Level Architecture
The system is divided into two distinct paths: the Real-Time Serving Path (critical path) and the Asynchronous Training Path.

### A. Real-Time Serving Path
1. **Ad Exchange / Client**: Sends user context JSON to the API Gateway.
2. **API Gateway / Load Balancer**: Distributes incoming traffic across multiple availability zones.
3. **Serving Fleet (BentoML / Triton)**: A scalable cluster of stateless microservices wrapping the XGBoost model.
4. **Online Feature Store (Redis)**: An ultra-fast, in-memory key-value store holding the precomputed Target Encodings (historical click rates) for high-cardinality features like `device_ip` and `site_domain`. Total storage requirement is < 50MB.

### B. Nightly Retraining Path (Continual Learning)
1. **Orchestrator**: GitHub Actions or Airflow triggers the ZenML pipelines nightly.
2. **Drift Pipeline**: Runs `evidently` to compare yesterday's traffic distributions against the training baseline.
3. **Training Pipeline**: Recomputes Target Encodings and fits a new XGBoost model.
4. **Registry**: Saves the artifacts to the MLflow Model Registry.

---

## 3. Deep Dives & Hard Problems

### Option A: Fallback Strategy & Caching (Resiliency)
To achieve 99.99% availability, the system must survive partial failures:
- **Redis Failure**: If the Redis feature store goes down, the API will fail to fetch target encodings. 
  - *Mitigation*: We maintain an in-memory LRU (Least Recently Used) cache inside the serving pods containing the top 10,000 most active IPs. If Redis times out, the API reads from local memory or uses a safe global default.
- **Out of Vocabulary (OOV)**: If a brand new `device_ip` appears, Redis will return null.
  - *Mitigation*: We implement a Hierarchical Fallback. If `device_ip` is unknown, we check the user's `site_domain` average. If that is also unknown, we fallback to the global CTR (Click-Through Rate) prior.

### Option B: Serving Optimization (Latency)
To guarantee < 50ms latency at P99 under peak load of 350 QPS:
- **Payload Optimization**: The API strips heavy Pandas DataFrames from the critical path, mapping the JSON directly to raw Numpy arrays or DMatrix objects before passing to XGBoost.
- **Dynamic Batching**: Using a serving engine like BentoML, the server waits a micro-window (e.g., 5ms) to group concurrent HTTP requests. It passes a batch of 10 requests to XGBoost simultaneously, maximizing CPU cache utilization and increasing throughput by 5x without breaching the 50ms SLA.

### Option C: Deployment Strategy (Zero-Downtime)
Pushing new models nightly introduces massive risk.
- **Shadow Deployment**: New models are first deployed in "Shadow Mode". Live traffic is duplicated to the new model, but its predictions are discarded. We monitor its latency and ensure it doesn't crash on real-world payloads.
- **Canary Rollout**: Once Shadow Testing passes, the Load Balancer routes 1% of live traffic to the new model, monitoring for a spike in HTTP 500 errors. It gradually ramps to 10%, 50%, and 100%.
- **Rolling Updates**: Kubernetes replaces the old pods one by one. Connections are gracefully drained so zero in-flight requests are dropped.

---

## 4. Monitoring & Observability
An ML system is only as good as its telemetry.
- **Infrastructure Metrics**: Prometheus and Grafana monitor API Latency, QPS, Error Rates, and CPU/Memory utilization.
- **Prediction Logging**: Every JSON payload and the resulting model probability is written asynchronously to an event stream (e.g., Kafka) and dumped into a Data Warehouse (Snowflake/BigQuery).
- **Drift Alerting**: The nightly ZenML drift pipeline analyzes the data warehouse logs and fires a Slack alert if `dataset_drift == True`.