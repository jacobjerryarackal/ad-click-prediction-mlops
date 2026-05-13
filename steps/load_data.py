import os
import pandas as pd
from typing_extensions import Annotated
from zenml import step
from core.validation import validate_dataset

@step
def load_and_validate_data(
    data_path: str = "data/dataset.csv",
) -> Annotated[pd.DataFrame, "raw_dataset"]:
    """
    Reads the CSV, runs schema validation, and returns a Pandas DataFrame.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at path: {data_path}")
        
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Run pure Python validation logic
    validate_dataset(df)
    
    return df
