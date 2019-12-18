import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path
from os import path
import random
import h5py
from functools import reduce
import itertools
#print(os.listdir("input"))

np.random.seed(8)
random.seed(8)

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


X_tr, y_tr, X_te, y_te = hdf5("data/preprocess_USPS/input/usps.h5")

classes = np.empty((10, 0)).tolist()
for i in range(len(y_tr)):
    if y_tr[i] == 0 and len(classes[0]) < 200:
        classes[0].append(X_tr[i])
    elif y_tr[i] == 1 and len(classes[1]) < 200:
        classes[1].append(X_tr[i])
    elif y_tr[i] == 2 and len(classes[2]) < 200:
        classes[2].append(X_tr[i])
    elif y_tr[i] == 3 and len(classes[3]) < 200:
        classes[3].append(X_tr[i])
    elif y_tr[i] == 4 and len(classes[4]) < 200:
        classes[4].append(X_tr[i])
    elif y_tr[i] == 5 and len(classes[5]) < 200:
        classes[5].append(X_tr[i])
    elif y_tr[i] == 6 and len(classes[6]) < 200:
        classes[6].append(X_tr[i])
    elif y_tr[i] == 7 and len(classes[7]) < 200:
        classes[7].append(X_tr[i])
    elif y_tr[i] == 8 and len(classes[8]) < 200:
        classes[8].append(X_tr[i])
    elif y_tr[i] == 9 and len(classes[9]) < 200:
        classes[9].append(X_tr[i])


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
    select first 40 samples for training and divide them into two classes
    classA: digits 0 to 4
    classB: digits 5 to 9
    the rest of the training set remains unlabeled

    label also the entire test set
    """
    train_dataA = classes[0]+classes[1]+classes[2]+classes[3]+classes[4]
    train_dataB = classes[5]+classes[6]+classes[7]+classes[8]+classes[9]

    train_dataA = np.asarray(train_dataA)
    train_dataA = np.reshape(train_dataA, (1000, 256))
    train_dataB = np.asarray(train_dataB)
    train_dataB = np.reshape(train_dataB, (1000, 256))

    train_classA = np.vstack([train_dataA[0::200, :],train_dataA[1::200, :],train_dataA[2::200, :],train_dataA[3::200, :]]) # 1
    train_classB = np.vstack([train_dataB[0::200, :],train_dataB[1::200, :],train_dataB[2::200, :],train_dataB[3::200, :]])# -1
    train_unlabeled = np.vstack([train_dataA[20:1000,:],train_dataB[20:1000,:]])
    test_classA = []
    test_classB = []
    # training set

    # control
    """
    for i in range(len(train_classB)):
        plt.imshow(train_classB[i].reshape([16, 16]), 'gray')
        plt.show()
    """

    """
    for i in range(len(y_tr)):
        if i in index:
            if 0 <= y_tr[i] <= 4:
                train_classA.append(X_tr[i])
            else:
                train_classB.append(X_tr[i])
        else:
            train_unlabeled.append(X_tr[i])
    """
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
    #visualize()
    save_data(path+'/')
    return


if __name__ == '__main__':
    main('data/preprocess_USPS')
