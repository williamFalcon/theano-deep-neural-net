
import sys
import os
from dnn.dnn import MLP
import pandas as pd
import numpy as np


def load_dataset():
    data_path = os.path.dirname(os.path.realpath(__file__)) + '/db/train.csv'
    train_df = pd.read_csv(data_path)

    # extract the x and y for the models
    # format for the nn
    X = train_df.values[:, 1:] / np.float32(256)
    Y = train_df.values[:, 0].astype(np.int32)

    val_size = 10000
    tng_end = (.80 * len(X)) - val_size
    val_end = tng_end + val_size

    # tng data 70%
    X_train = X[0:tng_end]
    y_train = Y[0:tng_end]

    # validation
    # 20%
    X_val = X[tng_end: val_end]
    y_val = Y[tng_end: val_end]

    # test accuracy of classifier
    # 10%
    X_test = X[val_end:]
    y_test = Y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def run_demo():
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # images are 28*28 pixels. Shown as a vector of 28*28 length
    nn = MLP(input_dim_count=28*28, output_size=10)
    nn.fit(X_train, y_train, X_val, y_val, X_test, y_test, epochs=50)

    # predict on first 5 of test
    x_preds = X_test[0:5]
    ans = nn.predict(x_preds)

    # print prediction results
    print(ans)
    print(y_test[0:5])


if __name__ == '__main__':
    run_demo()