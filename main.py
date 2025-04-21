 from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load your model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.post("/predict")
def predict(data: dict):
    # Process input data and make prediction
    # ...
    return {"prediction": "result"}
