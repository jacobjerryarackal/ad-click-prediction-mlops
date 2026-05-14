from fastapi import FastAPI
import pandas as pd
import uvicorn
from pydantic import BaseModel
from zenml.client import Client
from core.inference import generate_predictions
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Ad Click Predictor API", description="Real-time inference endpoint for ad clicks.")

# Global variables to hold artifacts in memory
model = None
preprocessor = None

class AdClickPayload(BaseModel):
    C1: int
    banner_pos: int
    site_id: str
    site_domain: str
    site_category: str
    app_id: str
    app_domain: str
    app_category: str
    device_id: str
    device_ip: str
    device_model: str
    device_type: int
    device_conn_type: int
    C14: int
    C15: int
    C16: int
    C17: int
    C18: int
    C19: int
    C20: int
    C21: int

@app.on_event("startup")
def load_artifacts():
    """Loads the model and preprocessor from the ZenML Model Registry into memory at startup."""
    global model, preprocessor
    logger.info("Connecting to ZenML Model Registry...")
    client = Client()
    model_version = client.get_model_version("ad_click_predictor", "latest")
    
    logger.info(f"Downloading artifacts from version: {model_version.name}...")
    model = model_version.get_artifact("trained_model").load()
    preprocessor = model_version.get_artifact("preprocessor").load()
    logger.info("API is ready to accept requests!")

@app.post("/predict")
def predict(payload: AdClickPayload) -> Dict[str, float]:
    """Accepts a JSON payload (raw user features), transforms it, and returns a click probability."""
    # Convert the validated Pydantic object into a Pandas DataFrame
    df = pd.DataFrame([payload.model_dump()])  
    predictions = generate_predictions(model, preprocessor, df)
    return {"click_probability": float(predictions["click_probability"].iloc[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)