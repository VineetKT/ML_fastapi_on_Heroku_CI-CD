# Overview

Deploying a Machine Learning Model on Heroku with FastAPI

Training a Random Forest classification model to predict the income category of a person based on other personal informations.

1. Created unit tests to monitor the _model performance on various slices_ of the data.
2. Then, _deployed the model using the FastAPI_ package and create API tests.
3. Both the slice-validation and the API tests were incorporated into a _CI/CD framework using GitHub Actions_.
4. UCI census datasets was used to experience updating the _dataset and model in git and DVC_.

## Repositories

[Link to the project repo](https://github.com/VineetKT/project3-ML-FastAPI-Heroku)

## Data

The training data is the census data available at the UCI library. It is the adult.income data from the data folder.
<br />
Link: [UCI Census Data](https://archive.ics.uci.edu/ml/datasets/census+income)

This data versioning is tracked through `DVC` using `AWS S3` bucket as remote storage.

## Model

A basic Random Forest classifier imported from scikit-learn library and fit onto the census data
Model parameters are(other than default):
{
"random_state": 8,
"max_depth": 16,
"n_estimators":128
}
<br />
<br />
Refer to the - [model card](https://github.com/VineetKT/project3-ML-FastAPI-Heroku/blob/main/model_card_template.md)

The model versioning was tracked using `dvc`.
<br />
Also, the performance of the model was evaluated on a slice of data ([code](https://github.com/VineetKT/project3-ML-FastAPI-Heroku/blob/main/test_slice.py)). The results are stored in the `slice_output.txt` for slices done on _education_ and _race_.
<br />
<br />
To run the model trainer, evaluation code ([link](https://github.com/VineetKT/project3-ML-FastAPI-Heroku/blob/main/main.py)):

```
python main.py
```

## Github action

**Continuous integration** was incorporated in the project using the _Github actions_. The action was completed only if the `pytest` and `flake8` linter tests passed on the project without any error.

## Unit tests

Unit tests were written for the model training and the inference API features. The tests are done using the `pytest` library via command:

```
pytest test/ -vv
```

## API Creation

A `FastAPI` framework was developed for the inference API using the input type-hints example from `pydantic` library
The API main file is the [inference_api.py](https://github.com/VineetKT/project3-ML-FastAPI-Heroku/blob/main/inference_api.py)

## API Deployment

The API was deployed on `Heroku` using the main branch of the current GitHub repository with `Continuous Delivery` enabled.
