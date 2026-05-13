from zenml import pipeline, Model
from steps.load_data import load_and_validate_data
from steps.split_data import split_dataset
from steps.preprocessing import preprocess_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate

@pipeline(
    model=Model(
        name="ad_click_predictor",
        description="Predicts whether a user will click an online ad",
        tags=["candidate", "xgboost", "tabular"],
    )
)
def ad_click_training_pipeline():
    """
    ZenML Pipeline: Defines the end-to-end training workflow.
    """
    # 1. Load and validate the data
    df = load_and_validate_data()
    
    # 2. Split the data chronologically (to prevent leakage)
    X_train_raw, X_test_raw, y_train, y_test = split_dataset(df)
    
    # 3. Preprocess features (bundle transformations to prevent skew)
    X_train_processed, X_test_processed, y_train, y_test = preprocess_data(
        X_train=X_train_raw,
        X_test=X_test_raw,
        y_train=y_train,
        y_test=y_test
    )
    
    # 4. Train model
    model = train_model(
        X_train=X_train_processed, 
        y_train=y_train,
        model_type="xgboost"
    )
    
    # 5. Evaluate model
    precision, logloss, auc = evaluate(
        model=model,
        X_test=X_test_processed,
        y_test=y_test
    )