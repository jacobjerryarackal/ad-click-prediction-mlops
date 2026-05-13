import logging
import pandas as pd
from sklearn.metrics import precision_score, log_loss, roc_auc_score
from sklearn.base import ClassifierMixin
from typing import Dict

logger = logging.getLogger(__name__)

def evaluate_model(model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluates the model using business-aligned metrics: Precision, LogLoss, and AUC.
    """
    logger.info("Predicting on test set...")
    y_pred = model.predict(X_test)
    
    # We need predict_proba for LogLoss and AUC
    y_proba = model.predict_proba(X_test)[:, 1] 
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    ll = log_loss(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    
    return {
        "precision": precision,
        "logloss": ll,
        "auc": auc
    }