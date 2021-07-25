"""Put the code for your API here.
"""
import joblib

from starter.train_model import trainer


if __name__ == '__main__':
    model = trainer('data/cleaned_data.csv')
    _ = joblib.dump(model, 'model/log_clf_v1.pkl')
