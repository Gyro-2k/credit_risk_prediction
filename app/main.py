from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

my_app = FastAPI()

class Applicant(BaseModel):
    age: int
    sex: str
    job:int
    housing: str
    savings_account: str
    checking_account: str
    credit_amount: int
    duration: int
    purpose: str
    
    # default values
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 35,
                    "sex": "male",
                    "job": 2,
                    "housing": "own",
                    "savings_account": "little",
                    "checking_account": "moderate",
                    "credit_amount": 5000,
                    "duration": 24,
                    "purpose": "car"
                }
            ]
        }
    }


# my trained model
my_model = joblib.load("xgb_model.pkl")

@my_app.post("/predict")
def predict(applicant: Applicant):
    input_df = pd.DataFrame([applicant.dict()])
    probalility = my_model.predict_proba(input_df)[0][1]
    prediction = my_model.predict(input_df)[0]

    return {

        # round to 3 decimal places
        "default probability": round(float(probalility), 3),
        "prediction": "bad (likely default)" if prediction == 1 else "good (likely repay)"
    }

@my_app.get("/")
def home():
    return {"message": "Loan Default API made by gyro"}