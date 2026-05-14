import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder

def get_advanced_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Creates a ColumnTransformer that applies OneHotEncoding to low-cardinality features
    and TargetEncoding to high-cardinality features.
    """
    # Define high cardinality features that were previously dropped
    high_cardinality_cols = ["site_id", "site_domain", "app_id", "device_id", "device_ip", "device_model"]
    target_encode_cols = [c for c in high_cardinality_cols if c in X.columns]
    
    # Categorical columns (low cardinality)
    categorical_cols = [c for c in X.select_dtypes(include=['object', 'category']).columns if c not in target_encode_cols]
    
    # Numeric columns
    numeric_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

    # Build transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    target_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        # TargetEncoder intelligently smooths categories with very few samples
        ('target_encoder', TargetEncoder(target_type="binary"))
    ])

    transformers = []
    if numeric_cols:
        transformers.append(('num', numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(('cat', categorical_transformer, categorical_cols))
    if target_encode_cols:
        transformers.append(('te', target_transformer, target_encode_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return preprocessor