import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin
from xgboost import XGBClassifier
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

def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
    """
    Trains an XGBoost candidate model.
    Uses scale_pos_weight to handle the imbalanced nature of ad clicks.
    """
    # Calculate ratio of negative to positive samples to handle class imbalance
    neg_class_count = (y_train == 0).sum()
    pos_class_count = (y_train == 1).sum()
    scale_weight = neg_class_count / pos_class_count if pos_class_count > 0 else 1.0
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model