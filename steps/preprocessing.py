import pandas as pd
from typing import Tuple
from typing_extensions import Annotated
from sklearn.compose import ColumnTransformer
from zenml import step
from zenml.logger import get_logger
from core.advanced_preprocessing import get_advanced_preprocessor

logger = get_logger(__name__)

@step
def preprocess_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train_processed"],
    Annotated[pd.DataFrame, "X_test_processed"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
    Annotated[ColumnTransformer, "preprocessor"],
]:
    """
    ZenML Step: Fits the sklearn pipeline on training data, and transforms both train and test sets.
    """
    logger.info("Building and fitting preprocessor...")
    preprocessor = get_advanced_preprocessor(X_train)
    
    # Fit on training data AND target labels to securely calculate Target Encodings
    X_train_processed_array = preprocessor.fit_transform(X_train, y_train)
    
    # Transform test data (never fit on test data!)
    X_test_processed_array = preprocessor.transform(X_test)
    
    # Convert back to DataFrame for better tracking/debugging 
    # (get_feature_names_out requires sklearn 1.2+)
    feature_names = preprocessor.get_feature_names_out()
    
    X_train_processed = pd.DataFrame(X_train_processed_array, columns=feature_names)
    X_test_processed = pd.DataFrame(X_test_processed_array, columns=feature_names)
    
    logger.info(f"Preprocessing complete. Training features shape: {X_train_processed.shape}")
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor
