# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

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


def model_inference(test_data, model_path, cat_features, label_column='income'):
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
