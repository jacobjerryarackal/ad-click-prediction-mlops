import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from typing import Tuple
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger
from zenml.client import Client
from core.inference import generate_predictions

logger = get_logger(__name__)

@step
def load_inference_data(
    data_path: str = "data/dataset.csv"
) -> Annotated[pd.DataFrame, "inference_data"]:
    """
    ZenML Step: Loads data for inference. Drops columns that shouldn't be passed to the model.
    """
    logger.info(f"Loading inference data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Simulate unseen data by dropping target and split features
    cols_to_drop = ["click", "id", "hour"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
    logger.info(f"Loaded {len(df)} rows for batch inference.")
    return df

@step
def load_model_artifacts() -> Tuple[
    Annotated[ClassifierMixin, "trained_model"],
    Annotated[ColumnTransformer, "preprocessor"]
]:
    """
    ZenML Step: Fetches the latest model and preprocessor artifacts from the ZenML Model Control Plane.
    """
    logger.info("Connecting to ZenML Model Registry...")
    client = Client()
    model_version = client.get_model_version("ad_click_predictor", "latest")
    
    logger.info(f"Loading artifacts from version: {model_version.version}")
    model = model_version.get_artifact("trained_model").load()
    preprocessor = model_version.get_artifact("preprocessor").load()
    
    return model, preprocessor

@step
def predict_batch(
    model: ClassifierMixin,
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
) -> Annotated[pd.DataFrame, "predictions"]:
    logger.info("Starting batch prediction...")
    predictions = generate_predictions(model, preprocessor, df)
    return predictions