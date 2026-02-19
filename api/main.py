from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib


#api 
app = FastAPI()

model = joblib.load("/media/bishaldas/Apps/Loan_defult_predition/model/rf_model_Loan_default_Prediction (1).pkl")
#define input formate
class UserInput(BaseModel):
    Employed: float
    Bank_Balance: float
    Annual_Salary: float

@app.get("/")
def home():
    return {"message ":"Api running successfully"}

@app.post("/predict")
def predict(data:UserInput):
    input_array = np.array([[
        data.Employed,
        data.Bank_Balance,
        data.Annual_Salary
    ]])

    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array)

    return {
        "prediction": int(prediction[0]),
        "confidence": float(max(probability[0]))
    }