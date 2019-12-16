import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path
from os import path
import random
import h5py
from functools import reduce
#print(os.listdir("input"))

def hdf5(path, data_key = "data", target_key = "target", flatten = True):
    """
        loads data from hdf5:
        - hdf5 should have 'train' and 'test' groups
        - each group should have 'data' and 'target' dataset or spcify the key
        - flatten means to flatten images N * (C * H * W) as N * D array
    """
    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        X_tr = train.get(data_key)[:]
        y_tr = train.get(target_key)[:]
        test = hf.get('test')
        X_te = test.get(data_key)[:]
        y_te = test.get(target_key)[:]
        if flatten:
            X_tr = X_tr.reshape(X_tr.shape[0], reduce(lambda a, b: a * b, X_tr.shape[1:]))
            X_te = X_te.reshape(X_te.shape[0], reduce(lambda a, b: a * b, X_te.shape[1:]))
    return X_tr, y_tr, X_te, y_te


X_tr, y_tr, X_te, y_te = hdf5("preprocess_USPS/input/usps.h5")
#print(X_tr.shape) 7291x256
#print(X_te.shape) 2007

def visualize():
    """
    plot some samples
    """
    num_samples = 10
    num_classes = len(set(y_tr))

    classes = set(y_tr)
    num_classes = len(classes)
    fig, ax = plt.subplots(num_samples, num_classes, sharex = True, sharey = True, figsize=(num_classes, num_samples))

    for label in range(num_classes):
        class_idxs = np.where(y_tr == label)
        for i, idx in enumerate(np.random.randint(0, class_idxs[0].shape[0], num_samples)):
            ax[i, label].imshow(X_tr[class_idxs[0][idx]].reshape([16, 16]), 'gray')
            ax[i, label].set_axis_off()
    plt.show()


def divide_data():
    """
    select 2000 samples for training and divide them into two classes
    classA: digits 0 to 4
    classB: digits 5 to 9
    the rest of the training set remains unlabeled

    label also the entire test set
    """
    index = random.sample(range(len(X_tr)), 10)
    train_classA = [] # 1
    train_classB = [] # -1
    train_unlabeled = []
    test_classA = []
    test_classB = []
    # training set
    for i in range(len(y_tr)):
        if i in index:
            if 0 <= y_tr[i] <= 4:
                train_classA.append(X_tr[i])
                # control
                """
                fig, ax = plt.subplots(1, 1, sharex = True, sharey = True)
                ax.imshow(X_tr[i].reshape([16, 16]), 'gray')
                plt.show()
                print(y_tr[i])
                """
            else:
                train_classB.append(X_tr[i])
        else:
            train_unlabeled.append(X_tr[i])
    # test set
    for i in range(len(y_te)):
        if 0 <= y_te[i] <= 4:
            test_classA.append(X_te[i])
        else:
            test_classB.append(X_te[i])

    return np.asarray(train_classA), np.asarray(train_classB), np.asarray(train_unlabeled), np.asarray(test_classA), np.asarray(test_classB)


def save_data(path):
    """
    save data sets as numpy arries
    """
    train_classA, train_classB, train_unlabeled, test_classA, test_classB = divide_data() #2000x256
    np.save(path+'train_classA.npy', train_classA)
    np.save(path+'train_classB.npy', train_classB)
    np.save(path+'train_unlabeled.npy', train_unlabeled)
    np.save(path+'test_classA.npy', test_classA)
    np.save(path+'test_classB.npy', test_classB)


def main(path):
    if os.path.isdir('./'+path):
        pass
    else:
        os.mkdir(path)
    visualize()
    save_data(path+'/')
    return


if __name__ == '__main__':
    main('preprocess_USPS')
