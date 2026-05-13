import pandas as pd
import mlflow
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client
from sklearn.base import ClassifierMixin
from core.training import train_baseline_model

# We must enable the MLflow experiment tracker for this step.
# The active stack already has it registered as 'mlflow_tracker'.
try:
    experiment_tracker = Client().active_stack.experiment_tracker
except Exception as e:
    experiment_tracker = None

@step(experiment_tracker=experiment_tracker.name if experiment_tracker else None)
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Annotated[ClassifierMixin, "baseline_model"]:
    """
    ZenML Step: Trains the model and logs it to MLflow.
    
    Concept: Experiment Tracking
    By wrapping the core training logic in an MLflow autolog context, 
    we automatically save the model's hyper-parameters (like max_iter, class_weight), 
    training metrics, and the actual model binary artifact.
    This guarantees reproducibility and allows us to easily compare different models later.
    """
    print("Starting MLflow tracking...")
    
    # Enable automatic logging for scikit-learn runs
    mlflow.sklearn.autolog()
    
    # Train the model (it will be automatically logged by MLflow inside train_baseline_model)
    model = train_baseline_model(X_train, y_train)
    
    return model
