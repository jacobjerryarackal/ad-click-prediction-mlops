from zenml import pipeline
from steps.load_data import load_and_validate_data
from steps.split_data import split_dataset
from steps.drift_steps import detect_data_drift

@pipeline
def ad_click_drift_pipeline():
    """
    ZenML Pipeline: Compares a reference dataset to a current dataset to detect data drift.
    For demonstration, we split the raw data and compare the older 80% to the newer 20%.
    """
    # 1. Load data
    df = load_and_validate_data()
    
    # 2. Split chronologically (Past 80% vs Present 20%)
    X_train_raw, X_test_raw, _, _ = split_dataset(df)
    
    # 3. Detect drift
    detect_data_drift(reference_df=X_train_raw, current_df=X_test_raw)