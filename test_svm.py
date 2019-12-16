import argparse
import os

import numpy as np

from svm_new import SVM


def main():

    # Argument parsing
    argparser = argparse.ArgumentParser()
    argparser.add_argument("load_path", type=str, help="Path to load data from.")
    args = argparser.parse_args()

    # Load test data
    test_classA_path = os.path.join(args.load_path, "test_classA.npy")
    test_classB_path = os.path.join(args.load_path, "test_classB.npy")
    try:
        test_classA = np.load(test_classA_path, allow_pickle=True).item().todense()
        test_classB = np.load(test_classB_path, allow_pickle=True).item().todense()
    except:
        test_classA = np.load(test_classA_path)
        test_classB = np.load(test_classB_path)


if __name__ == "__main__":
    main()
