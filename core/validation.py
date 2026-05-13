import logging
import pandas as pd

logger = logging.getLogger(__name__)

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validates the Ad Click Prediction dataset.
    Raises ValueError if validation fails.
    """
    if "click" not in df.columns:
        raise ValueError("Critical Validation Failure: Target column 'click' is missing from the dataset.")
        
    if "hour" not in df.columns:
        raise ValueError("Critical Validation Failure: Time column 'hour' is missing from the dataset.")
        
    # Check for empty dataframe
    if df.empty:
        raise ValueError("Critical Validation Failure: Dataset is empty.")
        
    # Check for completely null columns
    null_percentages = df.isnull().mean()
    completely_null_cols = null_percentages[null_percentages == 1.0].index.tolist()
    if completely_null_cols:
        raise ValueError(f"Validation Failure: The following columns are completely null: {completely_null_cols}")
        
    logger.info(f"Data validation passed. Data shape: {df.shape}")
