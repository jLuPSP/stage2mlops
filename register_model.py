import shutil
import json
import os

os.makedirs("registry", exist_ok=True)

metadata = {
    "model_path": "models/model.pkl",
    "version": "v1",
    "metrics": {
        "f1_score": 0.93
    }
}

with open("registry/metadata.json", "w") as f:
    json.dump(metadata, f)

shutil.copy("models/model.pkl", "registry/model_v1.pkl")
print("Model registered.")

# app.py
from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("models/model.pkl")

class IrisInput(BaseModel):
    features: list

@app.post("/predict")
def predict(input: IrisInput):
    pred = model.predict([input.features])
    return {"prediction": int(pred[0])}
