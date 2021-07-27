import os
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel

from starter.train_model import online_inference

app = FastAPI()

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class RowData(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system('rm -Rf .dvc/cache')
    os.system('rm -Rf .dvc/tmp/lock')
    os.system('dvc config core.hardlink_lock true')
    # os.system("dvc pull")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -Rf .dvc .apt/usr/lib/dvc")


@app.get("/")
def home():
    return {"Hello": "Welcome to project 3!"}


@app.post('/inference')
async def predict_income(inputrow: RowData):
    row_dict = jsonable_encoder(inputrow)
    model_path = 'model/rf_model_20210727-113146'
    prediction = online_inference(row_dict, model_path, CAT_FEATURES)

    return {"income class": prediction}


"""
sample_post_request = {
  "age": 32,
  "workclass": "Private",
  "fnlwgt": 205019,
  "education": "Assoc-acdm",
  "education_num": 12,
  "marital_status": "Never-married",
  "occupation": "Sales",
  "relationship": "Not-in-family",
  "race": "Black",
  "sex": "Male",
  "capital_gain": 0,
  "capital_loss": 0,
  "hours_per_week": 50,
  "native_country": "United-States"
}
"""
