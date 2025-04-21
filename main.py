from fastapi import FastAPI, Form
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict/")
def predict(
    v1: float = Form(...), v2: float = Form(...), v3: float = Form(...), v4: float = Form(...),
    v5: float = Form(...), v6: float = Form(...), v7: float = Form(...), v8: float = Form(...),
    v9: float = Form(...), v10: float = Form(...), v11: float = Form(...), v12: float = Form(...),
    v13: float = Form(...), v14: float = Form(...), v15: float = Form(...), v16: float = Form(...),
    v17: float = Form(...), v18: float = Form(...), v19: float = Form(...), v20: float = Form(...),
    v21: float = Form(...), v22: float = Form(...), v23: float = Form(...), v24: float = Form(...),
    v25: float = Form(...), v26: float = Form(...), v27: float = Form(...), v28: float = Form(...),
    amount: float = Form(...), time: float = Form(...)
):
    features = np.array([[time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                          v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                          v21, v22, v23, v24, v25, v26, v27, v28, amount]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)
    return {"prediction": int(prediction[0])}
