# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Add the necessary imports for the starter code.
from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference, train_model
import joblib


def create_data_slice(data_path, col_to_slice, value_to_replace=None):

    # Add code to load in the data.
    if value_to_replace:
        input_df = pd.read_csv(data_path, index_col=None)
        input_df[col_to_slice] = input_df[col_to_slice].apply(
            lambda x: str(value_to_replace)
        )

    else:
        input_df = pd.read_csv(data_path, index_col=None)
        input_df[col_to_slice] = input_df[col_to_slice].apply(
            lambda x: input_df[col_to_slice][0]
        )

    return input_df


def test_data_slice(data):

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # load the model.
    model, encoder, lb = joblib.load('model/log_clf_v1.pkl')

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label='income',
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


if __name__ == '__main__':
    col_to_slice = 'education'
    value_to_replace = 'HS-grad'  # ['Bachelors, 'Masters', 'HS-grad']

    print("performance on sliced column\t", col_to_slice, value_to_replace)
    sliced_data = create_data_slice(
        'data/cleaned_data.csv', col_to_slice, value_to_replace)
    precision, recall, fbeta = test_data_slice(sliced_data)

    with open('slice_output.txt', 'a') as f:
        result = f"\n{'-'*50}\nperformance on sliced column -- {col_to_slice} -- {value_to_replace}\n{'-'*50} \
            \nPrecision:\t{precision}\nRecall:\t{recall}\nF-beta score:\t{fbeta}\n"
        f.write(result)
