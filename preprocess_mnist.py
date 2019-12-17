import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def main():

    # Load MNIST
    inputs, targets = fetch_openml('mnist_784', version=1, return_X_y=True)

    # Split into train and test
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, train_size=60000, test_size=10000)

    # Replace labels 0-4 with 0 and labels 5-9 with 1
    targets_train = targets_train.astype(int)
    targets_test = targets_test.astype(int)

    targets_train[targets_train >= 0 && targets_train <= 4] = 0
    targets_train[targets_train >= 5 && targets_train <= 9] = 1



if __name__ == "__main__":
    main()
