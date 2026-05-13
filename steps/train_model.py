import pandas as pd
import mlflow
from sklearn.base import ClassifierMixin
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger
from core.training import train_baseline_model

logger = get_logger(__name__)

@step(experiment_tracker="mlflow_tracker")
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Annotated[ClassifierMixin, "baseline_model"]:
    """
    ZenML Step: Trains the baseline ML model and logs to MLflow.
    """
    logger.info("Starting baseline model training...")
    
    # Enable MLflow autologging for scikit-learn
    mlflow.sklearn.autolog()
    
    model = train_baseline_model(X_train, y_train)
    logger.info("Model training complete.")
    return model