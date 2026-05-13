import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.base import ClassifierMixin
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger
from core.training import train_baseline_model, train_xgboost_model

logger = get_logger(__name__)

@step(experiment_tracker="mlflow_tracker")
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "xgboost",
) -> Annotated[ClassifierMixin, "trained_model"]:
    """
    ZenML Step: Trains the ML model and logs to MLflow.
    """
    logger.info(f"Starting {model_type} model training...")
    
    # Enable MLflow autologging for scikit-learn
    mlflow.sklearn.autolog()
    
    if model_type == "xgboost":
        mlflow.xgboost.autolog()
        model = train_xgboost_model(X_train, y_train)
    else:
        model = train_baseline_model(X_train, y_train)
        
    logger.info("Model training complete.")
    return model