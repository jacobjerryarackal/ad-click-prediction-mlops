import os
import pandas as pd
from typing_extensions import Annotated
from zenml.logger import get_logger
from zenml import step
from core.validation import validate_dataset

logger = get_logger(__name__)

@step
def load_and_validate_data(
    data_path: str = "data/dataset.csv",
) -> Annotated[pd.DataFrame, "raw_dataset"]:
    """
    Reads the CSV, runs schema validation, and returns a Pandas DataFrame.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at path: {data_path}")
        
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Run pure Python validation logic
    validate_dataset(df)
    
    return df
