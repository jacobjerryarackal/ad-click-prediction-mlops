import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin

def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
    """
    Trains a Logistic Regression model as the baseline.
    
    Concept: Baseline Model
    We start with a simple, interpretable model before trying complex ones (like LightGBM or XGBoost).
    This establishes a minimum performance threshold. If a complex model doesn't significantly beat
    the baseline, we shouldn't use it because it adds unnecessary engineering complexity and latency.
    """
    print("Training baseline Logistic Regression model...")
    
    # max_iter=1000 ensures the solver converges on large datasets.
    # class_weight='balanced' adjusts weights inversely proportional to class frequencies.
    # This helps our model pay more attention to the minority class (clicks, which are only 16%).
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    
    model.fit(X_train, y_train)
    print("Model training complete.")
    
    return model
