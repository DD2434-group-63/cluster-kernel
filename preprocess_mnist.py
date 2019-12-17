import os
import argparse

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument("save_path", type=str, help="Path to save preprocessed data to.")
    argparser.add_argument("--n_unlabeled", type=int, default=None, help="Number of points to be unlabeled. Remaining points will be labeled.")
    argparser.add_argument("--n_labeled", type=int, default=None, help="Number of points to be labeled. Remaining points will be unlabeled.")
    args = argparser.parse_args()

    assert args.n_labeled is not None or args.n_unlabeled is not None, "One of n_labeled or n_unlabeled must be specified."
    assert args.n_labeled is None or args.n_unlabeled is None, "ONly one of n_labeled or n_unlabeled can be specified."

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

    # Randomly select specified number of unlabeled samples
    if args.n_labeled is not None:
        args.n_unlabeled = train_inputs.shape[0] - args.n_labeled
    random_indices = np.random.choice(train_inputs.shape[0], args.n_unlabeled, replace=False)
    train_unlabeled = train_inputs[random_indices, :]
    mask = np.ones(train_inputs.shape[0], dtype=bool)
    mask[random_indices] = False
    train_inputs = train_inputs[mask]
    train_targets = train_targets[mask]

    # Extract inputs of each class
    train_classA = train_inputs[train_targets == -1, :]
    train_classB = train_inputs[train_targets == 1, :]

    test_classA = test_inputs[test_targets == -1, :]
    test_classB = test_inputs[test_targets == 1, :]

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
