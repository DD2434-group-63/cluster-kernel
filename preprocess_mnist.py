import os
import argparse

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import scipy.sparse as ss

np.random.seed(8)


def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument("save_path", type=str, help="Path to save preprocessed data to.")
    argparser.add_argument("n_unlabeled", type=int, help="Number of points to be unlabeled.")
    argparser.add_argument("n_labeled", type=int, help="Number of points to be labeled.")
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

    # Replace labels 0-4 with 0 and labels 5-9 with 1
    train_targets = train_targets.astype(int)
    test_targets = test_targets.astype(int)

    train_zero_mask = np.logical_and(train_targets >= 0, train_targets <= 4)
    train_targets[train_zero_mask] = -1
    train_targets[np.logical_not(train_zero_mask)] = 1

    test_zero_mask = np.logical_and(test_targets >= 0, test_targets <= 4)
    test_targets[test_zero_mask] = -1
    test_targets[np.logical_not(test_zero_mask)] = 1

    # Extract inputs of each class, for labeled and unlabeled
    train_classA = train_inputs[train_targets == -1, :]
    train_classB = train_inputs[train_targets == 1, :]

    # Unlabeled
    random_indices = np.random.choice(train_classA.shape[0], int(args.n_unlabeled / 2), replace=False)
    train_unlabeled_classA = train_classA[random_indices, :]
    mask = np.ones(train_classA.shape[0], dtype=bool)
    mask[random_indices] = False
    train_classA = train_classA[mask]

    random_indices = np.random.choice(train_classB.shape[0], int(args.n_unlabeled / 2), replace=False)
    train_unlabeled_classB = train_classB[random_indices, :]
    mask = np.ones(train_classB.shape[0], dtype=bool)
    mask[random_indices] = False
    train_classB = train_classB[mask]

    train_unlabeled = np.concatenate((train_unlabeled_classA, train_unlabeled_classB))

    # Labeled
    random_indices = np.random.choice(train_classA.shape[0], int(args.n_labeled / 2), replace=False)
    train_classA = train_classA[random_indices, :]
    random_indices = np.random.choice(train_classB.shape[0], int(args.n_labeled / 2), replace=False)
    train_classB = train_classB[random_indices, :]

    # Get rid of excess test points
    test_classA = test_inputs[test_targets == -1, :]
    test_classB = test_inputs[test_targets == 1, :]
    random_indices = np.random.choice(test_classA.shape[0], int(args.n_test / 2), replace=False)
    test_classA = test_classA[random_indices, :]
    random_indices = np.random.choice(test_classB.shape[0], int(args.n_test / 2), replace=False)
    test_classB = test_classB[random_indices, :]

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
