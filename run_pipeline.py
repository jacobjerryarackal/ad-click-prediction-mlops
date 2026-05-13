from zenml import pipeline
from steps.load_data import load_and_validate_data
from steps.preprocessing import preprocess_data
from steps.training import train_model

@pipeline
def ad_click_training_pipeline():
    """Pipeline with data ingestion, preprocessing, and training."""
    df = load_and_validate_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)

if __name__ == "__main__":
    print("Running pipeline...")
    ad_click_training_pipeline()
