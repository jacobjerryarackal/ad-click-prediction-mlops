import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin
import logging

logger = logging.getLogger(__name__)

def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
    """
    Trains a baseline Logistic Regression model.
    Uses class_weight='balanced' to handle the imbalanced nature of ad clicks.
    """
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    return model