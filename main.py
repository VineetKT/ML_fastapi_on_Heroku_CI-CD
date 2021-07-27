"""Put the code for your API here.
"""

from starter.train_model import trainer, get_data, batch_inference
import time


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

if __name__ == '__main__':
    data_path = 'data/cleaned_data.csv'
    # model_path = 'model/rf_model_20210725-180330'
    model_path = f'model/rf_model_{time.strftime("%Y%m%d-%H%M%S")}'
    print(model_path)

    train_data, test_data = get_data(data_path)
    trainer(train_data, model_path, CAT_FEATURES)
    precision, recall, f_beta = batch_inference(
        test_data, model_path, CAT_FEATURES)
