from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("car_price_pipeline.pkl")

class CarInput(BaseModel):
    mileage: float
    age: int
    colour: str
    make: str
    condition: str

@app.get("/")
def home():
    return {"message": "Car price prediction API is running"}

@app.post("/predict")
def predict_price(car: CarInput):
    input_df = pd.DataFrame([{
        "mileage": car.mileage,
        "age": car.age,
        "colour": car.colour,
        "make": car.make,
        "condition": car.condition
    }])

    prediction = model.predict(input_df)[0]

    return {
        "predicted_price": float(prediction)
    }