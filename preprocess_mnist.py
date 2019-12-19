import os
import argparse

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(8)


def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument("save_path", type=str, help="Path to save preprocessed data to.")
    argparser.add_argument("n_labeled", type=int, help="Number of points to be labeled.")
    argparser.add_argument("n_unlabeled", type=int, help="Number of points to be unlabeled.")
    argparser.add_argument("n_test", type=int, help="Number of test points.")
    args = argparser.parse_args()

    # If save path does not exist, create it
    directory = os.path.dirname(args.save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Load MNIST
    inputs, targets = fetch_openml('mnist_784', version=1, return_X_y=True)

    # Split into train and test
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, train_size=60000, test_size=10000)

    # Normalize
    scaler = StandardScaler()
    train_inputs = scaler.fit_transform(train_inputs)
    test_inputs = scaler.transform(test_inputs)

    train_targets = train_targets.astype(int)
    test_targets = test_targets.astype(int)

    # Split samples into each class
    train_inputs_by_class = [[] for i in range(10)]
    for i in range(train_inputs.shape[0]):
        target = train_targets[i]
        train_inputs_by_class[target].append(train_inputs[i, :].tolist())

    test_inputs_by_class = [[] for i in range(10)]
    for i in range(test_inputs.shape[0]):
        target = test_targets[i]
        test_inputs_by_class[target].append(test_inputs[i, :].tolist())

    # Take equal number of each class for train, unlabeled, and test
    train_inputs = []
    unlabeled_inputs = []
    test_inputs = []
    for c in range(10):

        train_inputs_by_class[c] = np.array(train_inputs_by_class[c])
        test_inputs_by_class[c] = np.array(test_inputs_by_class[c])

        # train
        random_indices = np.random.choice(train_inputs_by_class[c].shape[0], int(args.n_labeled / 10), replace=False)
        train_inputs.append(train_inputs_by_class[c][random_indices, :])
        mask = np.ones(train_inputs_by_class[c].shape[0], dtype=bool)
        mask[random_indices] = False
        train_inputs_by_class[c] = train_inputs_by_class[c][mask]

        # unlabeled
        random_indices = np.random.choice(train_inputs_by_class[c].shape[0], int(args.n_unlabeled / 10), replace=False)
        unlabeled_inputs.append(train_inputs_by_class[c][random_indices, :])
        mask = np.ones(train_inputs_by_class[c].shape[0], dtype=bool)
        mask[random_indices] = False
        train_inputs_by_class[c] = train_inputs_by_class[c][mask]

        # test
        random_indices = np.random.choice(test_inputs_by_class[c].shape[0], int(args.n_test / 10), replace=False)
        test_inputs.append(test_inputs_by_class[c][random_indices, :])
        mask = np.ones(test_inputs_by_class[c].shape[0], dtype=bool)
        mask[random_indices] = False
        test_inputs_by_class[c] = test_inputs_by_class[c][mask]

    train_inputs = np.array(train_inputs)
    unlabeled_inputs = np.array(unlabeled_inputs)
    test_inputs = np.array(test_inputs)

    # Split into classes
    train_classA = train_inputs[0:5, :, :]
    train_classB = train_inputs[5:10, :, :]
    train_classA = train_classA.reshape(train_classA.shape[0] * train_classA.shape[1], train_classA.shape[2])
    train_classB = train_classB.reshape(train_classB.shape[0] * train_classB.shape[1], train_classB.shape[2])

    train_unlabeled = unlabeled_inputs.reshape(unlabeled_inputs.shape[0] * unlabeled_inputs.shape[1], unlabeled_inputs.shape[2])

    test_classA = test_inputs[0:5, :, :]
    test_classB = test_inputs[5:10, :, :]
    test_classA = test_classA.reshape(test_classA.shape[0] * test_classA.shape[1], test_classA.shape[2])
    test_classB = test_classB.reshape(test_classB.shape[0] * test_classB.shape[1], test_classB.shape[2])

    # Save data
    train_classA_save_path = os.path.join(args.save_path, "train_classA.npy")
    train_classB_save_path = os.path.join(args.save_path, "train_classB.npy")
    np.save(train_classA_save_path, train_classA)
    np.save(train_classB_save_path, train_classB)

    print("Saved {} training inputs of class A in {}".format(train_classA.shape[0], train_classA_save_path))
    print("Saved {} training inputs of class B in {}".format(train_classB.shape[0], train_classB_save_path))

    train_unlabeled_save_path = os.path.join(args.save_path, "train_unlabeled.npy")
    np.save(train_unlabeled_save_path, train_unlabeled)

    print("Saved {} unlabeled training inputs in {}".format(train_unlabeled.shape[0], train_unlabeled_save_path))

    test_classA_save_path = os.path.join(args.save_path, "test_classA.npy")
    test_classB_save_path = os.path.join(args.save_path, "test_classB.npy")
    np.save(test_classA_save_path, test_classA)
    np.save(test_classB_save_path, test_classB)

    print("Saved {} test inputs of class A in {}".format(test_classA.shape[0], test_classA_save_path))
    print("Saved {} test inputs of class B in {}".format(test_classB.shape[0], test_classB_save_path))


if __name__ == "__main__":
    main()
