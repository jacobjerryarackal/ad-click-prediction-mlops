import pandas as pd
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset
import logging

logger = logging.getLogger(__name__)

def generate_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, sample_size: int = 10000) -> dict:
    """
    Generates a data drift report using Evidently.
    Compares the current production data against the reference (training) baseline.
    """
    logger.info("Building Evidently DataDriftPreset report...")
    
    # Sample data to speed up statistical calculations
    if sample_size:
        if len(reference_df) > sample_size:
            reference_df = reference_df.sample(n=sample_size, random_state=42)
        if len(current_df) > sample_size:
            current_df = current_df.sample(n=sample_size, random_state=42)
            
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    return report.as_dict()