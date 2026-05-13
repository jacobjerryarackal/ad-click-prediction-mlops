import logging
import pandas as pd
from typing import Tuple

logger = logging.getLogger(__name__)

def split_data_chronological(df: pd.DataFrame, target_col: str = "click", test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data chronologically based on the 'hour' column.
    """
    logger.info("Sorting data chronologically...")
    # Sort data by time
    df_sorted = df.sort_values(by="hour").reset_index(drop=True)
    
    # Calculate split index
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    # Drop ID, target, and hour (used only for splitting, not a direct feature)
    X_train = train_df.drop(columns=[target_col, "id", "hour"], errors="ignore") 
    y_train = train_df[target_col]
    
    X_test = test_df.drop(columns=[target_col, "id", "hour"], errors="ignore")
    y_test = test_df[target_col]
    
    logger.info(f"Chronological split complete. Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test