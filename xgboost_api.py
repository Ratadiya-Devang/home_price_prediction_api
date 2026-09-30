import pandas as pd
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


api = FastAPI()


api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = joblib.load("xgboostregression.pkl")


class PredictionRequest(BaseModel):
    size: float = Field(..., gt=0, description="House size in square feet")
    bedrooms: int = Field(..., gt=0, description="Number of bedrooms")
    age: float = Field(..., gt=0, description="House age in years")
    distance: float = Field(..., gt=0, description="Distance from city in kilometers")


@api.get("/")
def test():
    return {
        "msg": "Home price prediction API is running"
    }


@api.post("/prediction")
def predict(data: PredictionRequest):

    new_data = pd.DataFrame({
        "size": [data.size],
        "bedrooms": [data.bedrooms],
        "age": [data.age],
        "distance": [data.distance]
    })

    prediction = model.predict(new_data)

    price = int(prediction[0])

    return {
        "price": price
    }