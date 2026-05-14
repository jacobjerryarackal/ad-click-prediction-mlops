from pipelines.drift_pipeline import ad_click_drift_pipeline
from pipelines.training_pipeline import ad_click_training_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print("1. Running Data Drift Pipeline...")
    ad_click_drift_pipeline()
    
    print("\n2. Analyzing Drift Results...")
    client = Client()
    # Get the most recent run of the drift pipeline
    run = client.get_pipeline("ad_click_drift_pipeline").last_run
    
    # Load the drift report artifact
    try:
        drift_report = run.steps["detect_data_drift"].output.load()
    except AttributeError:
        # Fallback for newer ZenML versions
        drift_report = run.steps["detect_data_drift"].outputs["drift_report"].load()
        
    is_drifted = drift_report["metrics"][0]["result"]["dataset_drift"]
    
    if is_drifted:
        print("🚨 DRIFT DETECTED! 🚨")
        print("Triggering automated model retraining to self-heal...")
        ad_click_training_pipeline()
    else:
        print("✅ No drift detected. The production model is still healthy.")