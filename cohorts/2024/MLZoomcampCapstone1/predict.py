import numpy as np
from typing import Dict
import bentoml
from pydantic import BaseModel


# Define the input schema
class HeartDiseaseFeatures(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

    class Config:
        json_schema_extra = {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1,
            }
        }


model_runner = bentoml.sklearn.get("heart_disease_classifier:latest").to_runner()

svc = bentoml.Service(name="heart_disease_predictor", runners=[model_runner])


@svc.api(
    input=bentoml.io.JSON.from_sample(
        HeartDiseaseFeatures.Config.json_schema_extra["example"]
    ),
    output=bentoml.io.JSON(),
)
async def predict(input_data: Dict) -> Dict:
    """Predict heart disease risk based on input features."""
    try:
        model = bentoml.sklearn.get("heart_disease_classifier:latest")
        scaler = model.custom_objects["scaler"]
        feature_names = model.custom_objects["feature_names"]

        # Convert input features to array using correct feature order
        feature_values = [input_data[name] for name in feature_names]

        # Scale features
        scaled_features = scaler.transform([feature_values])

        # Make prediction
        prediction = await model_runner.predict.async_run(scaled_features)
        probabilities = await model_runner.predict_proba.async_run(scaled_features)

        return {
            "prediction": int(prediction[0]),
            "probability": float(probabilities[0][1]),
            "status": "success",
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}
