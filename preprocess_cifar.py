import numpy as np
import matplotlib.pyplot as plt
import pickle
from functools import reduce


def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)


    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    # call select_data
    train_classA, train_classB = select_data(cifar_train_data, cifar_train_filenames, cifar_train_labels,cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names)

    #return cifar_train_data, cifar_train_filenames, cifar_train_labels, \
    #    cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names
    return train_classA, train_classB

def select_data(cifar_train_data, cifar_train_filenames, cifar_train_labels,cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names):
    """
    we have                                 we want
    Train data:  (50000, 32, 32, 3)         Train data:  (500, 1024)
    Train filenames:  (50000,)              Train filenames:  (500,)
    Train labels:  (50000,)                 Train labels:  (500,)
    Test data:  (10000, 32, 32, 3)          Test data:  (100, 1024)
    Test filenames:  (10000,)               Test filenames:  (100,)
    Test labels:  (10000,)                  Test labels:  (100,)
    Label names:  (10,)                     Label names:  (2,)
    """
    classA = []
    classB = []
    # use only class 1 and class 2
    for i in range(len(cifar_train_labels)):
        if cifar_train_labels[i] == 0 and len(classA) < 300:
            classA.append(rgb2gray(cifar_train_data[i]))
        elif cifar_train_labels[i] == 1 and len(classB) < 300:
            classB.append(rgb2gray(cifar_train_data[i]))
        elif len(classA)==300 and len(classB)==300:
            break
    classA = np.asarray(classA)
    classB = np.asarray(classB)

    # flatten
    # the shape of the training data is now 600x1024, 600 samples and 1024 pixels
    # 300 sample from each class
    classA = classA.reshape(classA.shape[0], reduce(lambda a, b: a * b, classA.shape[1:]))
    classB = classB.reshape(classB.shape[0], reduce(lambda a, b: a * b, classB.shape[1:]))

    train_classA = classA[:50,:]
    train_classB = classB[:50,:]
    train_unlabeled_classA = classA[50:250,:]
    train_unlabeled_classB = classA[50:250,:]
    train_unlabeled = np.vstack([train_unlabeled_classA, train_unlabeled_classB])
    np.random.shuffle(train_unlabeled)
    test_classA = classA[250:,:]
    test_classB = classB[250:,:]

    path = "data/preprocess_cifar/"
    np.save(path+'train_classA.npy', train_classA)
    np.save(path+'train_classB.npy', train_classB)
    np.save(path+'train_unlabeled.npy', train_unlabeled)
    np.save(path+'test_classA.npy', test_classA)
    np.save(path+'test_classB.npy', test_classB)

    print(train_classA.shape)
    print(train_unlabeled.shape)
    print(test_classA.shape)

    return train_classA, train_classB

def rgb2gray(rgb):
    """
    from (32, 32, 3) to (1, 32x32)
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


if __name__ == "__main__":
    cifar_10_dir = 'data/preprocess_cifar/cifar-10-batches-py'
    train_classA, train_classB = load_cifar_10_data(cifar_10_dir)


    # control class
    train_class = train_classA
    num_plot = 5
    f, ax = plt.subplots(num_plot, num_plot)
    for m in range(num_plot):
        for n in range(num_plot):
            idx = np.random.randint(0, train_class.shape[0])
            ax[m, n].imshow(train_class[idx].reshape(32,32))
            ax[m, n].get_xaxis().set_visible(False)
            ax[m, n].get_yaxis().set_visible(False)
    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0)
    plt.show()
