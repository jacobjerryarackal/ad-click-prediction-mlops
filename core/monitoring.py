import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import logging

logger = logging.getLogger(__name__)

def generate_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    """
    Generates a data drift report using Evidently.
    Compares the current production data against the reference (training) baseline.
    """
    logger.info("Building Evidently DataDriftPreset report...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    return report.as_dict()