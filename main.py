"""Put the code for your API here.
"""
from starter.train_model import trainer


def add(x, y):
    return x+y


if __name__ == '__main__':
    print(add(8, 23))
    trainer('data/cleaned_data.csv')
