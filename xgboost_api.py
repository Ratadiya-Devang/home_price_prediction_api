import pandas as pd 
from fastapi import FastAPI
from prediction import houseprediction
import joblib



api = FastAPI()
model = joblib.load("xgboostregression.pkl")

@api.get("/")
def test():
    return {"msg":"your model successfully tested"}

@api.post("/prediction")
def predict(size:float,bedrooms:int,age:float,distance:float):
    
    new_data = pd.DataFrame({
        "size":[size],
        "bedrooms":[bedrooms],
        "age":[age],
        "distance":[distance]
    })

    pre = model.predict(new_data)

    return houseprediction(pre)
