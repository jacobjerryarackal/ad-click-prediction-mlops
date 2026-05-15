import os
import joblib
from zenml.client import Client

def export_artifacts():
    print("Fetching the latest training pipeline run from ZenML...")
    client = Client()
    
    try:
        run = client.get_pipeline("ad_click_training_pipeline").last_run
        print(f"Found run: {run.name}")
    except Exception:
        print("Could not find the pipeline. Have you run `python run_pipeline.py` yet?")
        return
        
    os.makedirs("models", exist_ok=True)
    
    model = None
    preprocessor = None
    
    # Dynamically search through the pipeline steps for the model and preprocessor
    for step_name, step in run.steps.items():
        print(f"Checking step: '{step_name}'...")
        
        outputs = {}
        if hasattr(step, "outputs") and step.outputs:
            outputs = step.outputs
        elif hasattr(step, "output") and step.output:
            outputs = {"output": step.output}
            
        for output_name, artifact_view in outputs.items():
            try:
                # Handle newer ZenML versions where outputs are lists of artifacts
                if isinstance(artifact_view, list):
                    if not artifact_view:
                        continue
                    artifact_view = artifact_view[0]
                    
                obj = artifact_view.load()
                if hasattr(obj, "predict"):
                    model = obj
                    print(f"  ✅ Found model in output '{output_name}'!")
                elif hasattr(obj, "transform") and not hasattr(obj, "predict"):
                    preprocessor = obj
                    print(f"  ✅ Found preprocessor in output '{output_name}'!")
            except Exception as e:
                print(f"  ⚠️ Could not load output '{output_name}': {e}")
            
    if model:
        joblib.dump(model, "models/model.pkl")
    if preprocessor:
        joblib.dump(preprocessor, "models/preprocessor.pkl")
        
    if model and preprocessor:
        print("\n🚀 Successfully exported models/model.pkl and models/preprocessor.pkl!")
    else:
        print("\n❌ Missing artifacts. Model found:", bool(model), "| Preprocessor found:", bool(preprocessor))

if __name__ == "__main__":
    export_artifacts()