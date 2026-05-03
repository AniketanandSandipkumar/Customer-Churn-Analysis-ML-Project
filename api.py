from fastapi import FastAPI
import pickle
import pandas as pd
from src.preprocessing import create_features

app = FastAPI()

model = pickle.load(open("models/churn_model_latest.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "Churn API Running 🚀"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    df = create_features(df)

    prob = model.predict_proba(df)[0][1]

    return {"churn_probability": float(prob)}
