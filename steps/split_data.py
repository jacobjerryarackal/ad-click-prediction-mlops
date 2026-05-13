import pandas as pd
from typing import Tuple
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger
from core.splitting import split_data_chronological

logger = get_logger(__name__)

@step
def split_dataset(
    df: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train_raw"],
    Annotated[pd.DataFrame, "X_test_raw"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    ZenML Step: Splits the dataset into train and test sets chronologically.
    """
    logger.info("Starting chronological data split step...")
    X_train, X_test, y_train, y_test = split_data_chronological(df)
    return X_train, X_test, y_train, y_test