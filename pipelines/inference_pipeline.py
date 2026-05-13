from zenml import pipeline
from steps.inference_steps import load_inference_data, load_model_artifacts, predict_batch

@pipeline
def ad_click_batch_inference_pipeline():
    """
    ZenML Pipeline: Loads new data, fetches the registered model and preprocessor, and generates predictions.
    """
    # 1. Load the new, unseen data
    df = load_inference_data()
    
    # 2. Load the production model and preprocessor from the Model Registry
    model, preprocessor = load_model_artifacts()
    
    # 3. Generate predictions
    predict_batch(
        model=model,
        preprocessor=preprocessor,
        df=df
    )