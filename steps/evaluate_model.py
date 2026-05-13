import pandas as pd
from sklearn.base import ClassifierMixin
from typing import Tuple
from typing_extensions import Annotated
from zenml import step, log_metadata
from zenml.logger import get_logger
from core.evaluation import evaluate_model

logger = get_logger(__name__)

@step(experiment_tracker="mlflow_tracker")
def evaluate(
    model: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[
    Annotated[float, "precision"],
    Annotated[float, "logloss"],
    Annotated[float, "auc"]
]:
    """
    ZenML Step: Evaluates the model on the test set.
    """
    logger.info("Starting model evaluation...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Log metrics to ZenML Model Control Plane and MLflow
    log_metadata(metadata=metrics, infer_model=True)
    
    logger.info(f"Test Metrics -> Precision: {metrics['precision']:.4f} | LogLoss: {metrics['logloss']:.4f} | AUC: {metrics['auc']:.4f}")
    return metrics["precision"], metrics["logloss"], metrics["auc"]