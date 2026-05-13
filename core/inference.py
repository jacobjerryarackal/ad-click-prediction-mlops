import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
import logging

logger = logging.getLogger(__name__)

def generate_predictions(
    model: ClassifierMixin,
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
    user_ids: pd.Series = None
) -> pd.DataFrame:
    """
    Applies the fitted preprocessor and model to generate predictions.
    """
    logger.info("Transforming new data with the saved preprocessor...")
    # Apply the exact same transformation as training
    X_processed_array = preprocessor.transform(df)
    
    feature_names = preprocessor.get_feature_names_out()
    X_processed = pd.DataFrame(X_processed_array, columns=feature_names)
    
    logger.info("Generating click probabilities...")
    # Get probability of class 1 (click)
    probabilities = model.predict_proba(X_processed)[:, 1]
    
    # Return a DataFrame with predictions
    result_df = pd.DataFrame({"click_probability": probabilities})
    
    # Attach the user IDs back to the predictions if provided
    if user_ids is not None and not user_ids.empty:
        result_df.insert(0, "id", user_ids.values)
        
    return result_df