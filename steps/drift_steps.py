import pandas as pd
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger
from core.monitoring import generate_drift_report

logger = get_logger(__name__)

@step
def detect_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> Annotated[dict, "drift_report"]:
    """
    ZenML Step: Runs Evidently drift detection and logs high-level results.
    """
    logger.info("Running Evidently data drift detection...")
    drift_report = generate_drift_report(reference_df, current_df)
    
    # Extract basic info for logging
    dataset_drift = drift_report["metrics"][0]["result"]["dataset_drift"]
    drifted_features = drift_report["metrics"][0]["result"]["number_of_drifted_columns"]
    
    logger.info(f"Dataset Drift Detected: {dataset_drift}")
    logger.info(f"Number of Drifted Features: {drifted_features}")
    
    return drift_report