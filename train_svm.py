import argparse
import os

import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics.pairwise import rbf_kernel

from kernel_functions import *

np.random.seed(8)

# Hyperparameters
C = 3.0
kernel = "rbf"
GAMMA = 1 / (2 * 5 ** 2)


def compute_accuracy(predictions, targets):
    """
    Computes the accuracy of predictions in relation to targets.
    """
    return predictions[predictions == targets].size / predictions.size


def compute_K_test(K_tilde_labeled, K_labeled, train_inputs, test_inputs, GAMMA):
    """
    Compute kernel of test points as linear combination of training points.
    """
    V = rbf_kernel(test_inputs, train_inputs, GAMMA=GAMMA)
    K_tilde_test = np.matmul(K_tilde_labeled, np.matmul(np.linalg.inv(K_labeled), V.T)).T

    return K_tilde_test


def main():

    # Argument parsing
    argparser = argparse.ArgumentParser()
    argparser.add_argument("load_path", type=str, help="Path to load data from.")
    argparser.add_argument("type_kernel", type=str, help="Specify the type of kernel used in SVM. "
                                                         "('normal', 'basic_cluster, ''linear','step','lin_step','poly', 'poly_step')")
    argparser.add_argument("--test_points_at_train", type=int, default=1, help="Specify whether to use test points at training time (1) or not (0)")
    args = argparser.parse_args()

    # Load data
    train_classA_path = os.path.join(args.load_path, "train_classA.npy")
    train_classB_path = os.path.join(args.load_path, "train_classB.npy")
    train_unlabeled_path = os.path.join(args.load_path, "train_unlabeled.npy")
    test_classA_path = os.path.join(args.load_path, "test_classA.npy")
    test_classB_path = os.path.join(args.load_path, "test_classB.npy")

    try:
        train_classA = np.load(train_classA_path, allow_pickle=True).item().todense()
        train_classB = np.load(train_classB_path, allow_pickle=True).item().todense()
        train_unlabeled = np.load(train_unlabeled_path, allow_pickle=True).item().todense()
        test_classA = np.load(test_classA_path, allow_pickle=True).item().todense()
        test_classB = np.load(test_classB_path, allow_pickle=True).item().todense()
    except:
        train_classA = np.load(train_classA_path)
        train_classB = np.load(train_classB_path)
        train_unlabeled = np.load(train_unlabeled_path)
        test_classA = np.load(test_classA_path)
        test_classB = np.load(test_classB_path)

    # Stack training samples
    train_inputs = np.concatenate((train_classA, train_classB))
    train_targets = np.concatenate((np.ones(train_classA.shape[0]), -np.ones(train_classB.shape[0])))
    N = train_inputs.shape[0]
    random_perm = np.random.permutation(N)
    train_inputs = train_inputs[random_perm, :]
    train_targets = train_targets[random_perm]

    # Stack test samples
    test_inputs = np.concatenate((test_classA, test_classB))
    test_targets = np.concatenate((np.ones(test_classA.shape[0]), -np.ones(test_classB.shape[0])))
    N_test = test_inputs.shape[0]
    random_perm = np.random.permutation(N_test)
    test_inputs = test_inputs[random_perm, :]
    test_targets = test_targets[random_perm]

    # Run SVM
    if args.type_kernel == "normal":  # normal version
        svm = SVC(C=C, kernel="rbf")
        svm.fit(train_inputs, train_targets)

    elif args.type_kernel == "basic_cluster":  # basic cluster kernel version
        svm = SVC(C=C, kernel="rbf")
        train_inputs_modified, test_inputs_modified = clustring_kernel(train_inputs, train_unlabeled[0:40, :], test_inputs, GAMMA, k=2)
        svm.fit(train_inputs_modified, train_targets)

    else:  # extended cluster kernel version

        if args.test_points_at_train == 1:
            K_tilde_labeled, K_test = cluster_kernel_extension(train_inputs, train_unlabeled[0:40, :], test_inputs, GAMMA, args.type_kernel, 1)
        else:
            K_tilde_labeled, K_labeled = cluster_kernel_extension(train_inputs, train_unlabeled[0:40, :], None, GAMMA, args.type_kernel, 1)

        svm = SVC(C=C, kernel="precomputed")
        svm.fit(K_tilde_labeled.T, train_targets.T)

    print("train_input shape:", train_inputs.shape)
    print("test_input shape:", test_inputs.shape)

    # Test SVM
    if args.type_kernel == "normal":
        test_predictions = svm.predict(test_inputs)
    elif args.type_kernel == "basic_cluster":
        test_predictions = svm.predict(test_inputs_modified)
    else:
        if args.test_points_at_train == 0:
            K_test = compute_K_test(K_tilde_labeled, K_labeled, train_inputs, test_inputs, GAMMA)
        test_predictions = svm.predict(K_test)

    # performance
    accuracy = compute_accuracy(test_predictions, test_targets)

    f1_score = metrics.f1_score(test_targets, test_predictions, average='macro')
    print("Accuracy = ", accuracy)
    print("F1 score = ", f1_score)


if __name__ == "__main__":
    main()
