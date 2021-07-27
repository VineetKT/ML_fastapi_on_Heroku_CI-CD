# Script to train machine learning model.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from .ml.data import process_data
from .ml.model import compute_model_metrics, inference, train_model
import joblib


def get_data(data_path):
    # Add code to load in the data.
    input_df = pd.read_csv(data_path, index_col=None)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train_data, test_data = train_test_split(
        input_df, test_size=0.20, random_state=8, shuffle=True
    )

    return train_data, test_data


def trainer(train_data, model_path, cat_features, label_column='income'):
    # Proces the train data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train_data, categorical_features=cat_features, label=label_column, training=True
    )

    # Train and save a model.
    model = train_model(X_train, y_train)

    # Save the model in `model_path`
    joblib.dump((model, encoder, lb), model_path)


def batch_inference(test_data, model_path, cat_features, label_column='income'):
    # load the model from `model_path`
    model, encoder, lb = joblib.load(model_path)

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test_data,
        categorical_features=cat_features,
        label=label_column,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # evaluate model
    preds = inference(model=model, X=X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print('Precision:\t', precision)
    print('Recall:\t', recall)
    print('F-beta score:\t', fbeta)

    return precision, recall, fbeta


def online_inference(row_dict, model_path, cat_features):
    # load the model from `model_path`
    model, encoder, lb = joblib.load(model_path)

    row_transformed = list()
    X_categorical = list()
    X_continuous = list()

    for key, value in row_dict.items():
        mod_key = key.replace('_', '-')
        if mod_key in cat_features:
            X_categorical.append(value)
        else:
            X_continuous.append(value)

    y_cat = encoder.transform([X_categorical])
    y_conts = np.asarray([X_continuous])

    row_transformed = np.concatenate([y_conts, y_cat], axis=1)

    # get inference from model
    preds = inference(model=model, X=row_transformed)

    return '>50K' if preds[0] else '<=50K'
