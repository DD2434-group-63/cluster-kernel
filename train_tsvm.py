import argparse
import os

import numpy as np
from tsvm import SKTSVM
from sklearn import metrics


def compute_accuracy(predictions, targets):
    """
    Computes the accuracy of predictions in relation to targets.
    """
    return predictions[predictions == targets].size / predictions.size


def main():

    # Argument parsing
    argparser = argparse.ArgumentParser()
    argparser.add_argument("load_path", type=str, help="Path to load data from.")
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

    # Stack training samples (labeled and unlabeled)
    # Labeled samples get labels 0 and 1, unlabeled gets label -1
    train_inputs = np.concatenate((train_classA, train_classB, train_unlabeled))
    train_targets = np.concatenate((np.ones(train_classA.shape[0], dtype=np.int8), np.zeros(train_classB.shape[0], dtype=np.int8), -np.ones(train_unlabeled.shape[0], dtype=np.int8)))
    N = train_inputs.shape[0]
    random_perm = np.random.permutation(N)
    train_inputs = train_inputs[random_perm, :]
    train_targets = train_targets[random_perm]

    # Stack test samples
    # Labels are 0 and 1
    test_inputs = np.concatenate((test_classA, test_classB))
    test_targets = np.concatenate((np.ones(test_classA.shape[0], dtype=np.int8), np.zeros(test_classB.shape[0], dtype=np.int8)))
    N_test = test_inputs.shape[0]
    random_perm = np.random.permutation(N_test)
    test_inputs = test_inputs[random_perm, :]
    test_targets = test_targets[random_perm]

    # Run transductive svm
    tsvm = SKTSVM(kernel="rbf", probability=False)
    tsvm.fit(train_inputs, train_targets)

    # Test trained tsvm
    test_predictions = tsvm.predict(test_inputs)
    accuracy = compute_accuracy(test_predictions, test_targets)
    f1_score = metrics.f1_score(test_targets, test_predictions, average='macro')
    print("Accuracy = ", accuracy)
    print("F1 score = ", f1_score)


if __name__ == "__main__":
    main()
