import pandas as pd
from sklearn.base import ClassifierMixin
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger
from core.training import train_baseline_model

logger = get_logger(__name__)

@step
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Annotated[ClassifierMixin, "baseline_model"]:
    """
    ZenML Step: Trains the baseline ML model.
    """
    logger.info("Starting baseline model training...")
    model = train_baseline_model(X_train, y_train)
    logger.info("Model training complete.")
    return model