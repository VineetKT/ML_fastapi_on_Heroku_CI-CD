# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Add the necessary imports for the starter code.
from .ml.data import process_data
from .ml.model import compute_model_metrics, inference, train_model


def trainer(data_path):

    # Add code to load in the data.
    input_df = pd.read_csv(data_path, index_col=None)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(input_df, test_size=0.20)

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

    # Proces the test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="income", training=True
    )

    # Train and save a model.
    model = train_model(X_train, y_train)

    # Testing and evaluation
    X_test, y_test, encoder, lb = process_data(
        test,
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

    return model
