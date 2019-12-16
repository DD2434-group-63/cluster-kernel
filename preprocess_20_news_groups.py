import os
import argparse

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

WINDOWS_IND = 4
MAC_IND = 5

if __name__ == "__main__":

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

    # Fetch data
    train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    train_inputs = train_data.data
    train_targets = train_data.target

    test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    test_inputs = test_data.data
    test_targets = test_data.target

    # Extract features
    count_vec = CountVectorizer(max_features=7511)
    train_inputs = count_vec.fit_transform(train_inputs)
    test_inputs = count_vec.fit_transform(test_inputs)

    # Remove samples that are not of class mac or windows
    samples_to_use_mask = np.logical_or(train_targets == WINDOWS_IND, train_targets == MAC_IND)
    train_inputs = train_inputs[samples_to_use_mask, :]
    train_targets = train_targets[samples_to_use_mask]

    samples_to_use_mask = np.logical_or(test_targets == WINDOWS_IND, test_targets == MAC_IND)
    test_inputs = test_inputs[samples_to_use_mask, :]
    test_targets = test_targets[samples_to_use_mask]

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
    train_classA = train_inputs[train_targets == WINDOWS_IND, :]
    train_classB = train_inputs[train_targets == MAC_IND, :]

    test_classA = test_inputs[test_targets == WINDOWS_IND, :]
    test_classB = test_inputs[test_targets == MAC_IND, :]

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
