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
    args = argparser.parse_args()

    # If save path does not exist, create it
    directory = os.path.dirname(args.save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Fetch data
    full_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    inputs = full_data.data
    targets = full_data.target

    # Extract features
    count_vec = CountVectorizer(max_features=7511)
    inputs = count_vec.fit_transform(inputs)

    # Remove samples that are not of class mac or windows
    samples_to_use_mask = np.logical_or(targets == WINDOWS_IND, targets == MAC_IND)
    inputs = inputs[samples_to_use_mask, :]
    targets = targets[samples_to_use_mask]

    # Change targets to -1 and 1
    targets[targets == WINDOWS_IND] = -1
    targets[targets == MAC_IND] = 1

    # Save data
    input_save_path = os.path.join(args.save_path, "inputs.npy")
    target_save_path = os.path.join(args.save_path, "targets.npy")
    np.save(input_save_path, inputs)
    np.save(target_save_path, targets)
    print("Saved inputs in {}".format(input_save_path))
    print("Saved targets in {}".format(target_save_path))
