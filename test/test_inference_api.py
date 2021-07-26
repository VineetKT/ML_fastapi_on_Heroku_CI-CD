from fastapi.testclient import TestClient
from inference_api import app

client = TestClient(app)


def test_home():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json() == {"Hello": "Welcome to project 3!"}


def test_1_predict_income():
    row1 = {
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
    r = client.post('/inference', json=row1)
    assert r.status_code == 200
    assert r.json() == {"income class": '<=50K'}


def test_2_predict_income():
    row2 = {
        "age": 47,
        "workclass": "Private",
        "fnlwgt": 51835,
        "education": "Prof-school",
        "education_num": 15,
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 1902,
        "hours_per_week": 60,
        "native_country": "Honduras"
    }
    r = client.post('/inference', json=row2)
    assert r.status_code == 200
    assert r.json() == {"income class": '>50K'}
