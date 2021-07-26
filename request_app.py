import requests

row = {
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

# https://git.heroku.com/quiet-bayou-25995.git
heroku_endpoint = 'https://quiet-bayou-25995.herokuapp.com/'
localhost_endpoint = 'http://127.0.0.1:8000/'

response = requests.get(localhost_endpoint)
# response = requests.post(heroku_endpoint, json=row)

print(response.status_code)
print(response.json())
