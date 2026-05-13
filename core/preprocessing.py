import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple

def split_data_chronological(df: pd.DataFrame, target_col: str = "click", test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data chronologically based on the 'hour' column.
    
    Concept: Data Leakage
    If we split randomly, future clicks might be in the training set and past clicks in the test set.
    The model would 'cheat' by learning future trends to predict the past.
    By sorting chronologically, we simulate real-world production where we train on the past to predict the future.
    """
    print("Sorting data chronologically...")
    # Sort data by time
    df_sorted = df.sort_values(by="hour").reset_index(drop=True)
    
    # Calculate split index
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    # Drop ID, target, and hour (used only for splitting, not a direct feature)
    X_train = train_df.drop(columns=[target_col, "id", "hour"]) 
    y_train = train_df[target_col]
    
    X_test = test_df.drop(columns=[target_col, "id", "hour"])
    y_test = test_df[target_col]
    
    return X_train, X_test, y_train, y_test

def get_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Builds an sklearn ColumnTransformer to preprocess the features.
    
    Concept: Training-Serving Skew
    By bundling the OneHotEncoder into a Pipeline/ColumnTransformer, we ensure that
    when a new row of data arrives in production, it undergoes the EXACT same transformation
    as the training data. If we processed data manually with pandas, we might miss a new categorical value
    in production and crash the system.
    """
    # Identify high cardinality columns to drop for MVP
    # Heuristic: Drop anything with > 1000 unique values in training
    high_cardinality_cols = [col for col in X_train.columns if X_train[col].nunique() > 1000]
    
    # The remaining categorical columns will be One-Hot Encoded
    low_cardinality_cols = [col for col in X_train.columns if col not in high_cardinality_cols]
    
    # Define the transformer
    # handle_unknown='ignore' ensures that if a new category appears in the test set or in production,
    # the encoder doesn't crash, it just sets all one-hot columns to 0.
    categorical_transformer = Pipeline(steps=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, low_cardinality_cols),
            ("drop_high_card", "drop", high_cardinality_cols)
        ],
        remainder="passthrough" # Keep any other columns (numeric) as they are
    )
    
    return preprocessor
