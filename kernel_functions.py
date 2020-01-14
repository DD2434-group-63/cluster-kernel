import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import sys

np.random.seed(8)

"""
def kernel_linear(x1, x2):
    return np.dot(x1, x2.T)


def kernel_quadratic(x1, x2):
    return (np.dot(x1, x2.T) ** 2)


def kernel_poly(x1, x2, p):
    return np.power(np.dot(x1, x2) + 1, p)


def kernel_rbf(x1, x2, sigma):
    return np.exp(-np.linalg.norm(x1 - x2)**2) / (2 * sigma**2)
"""
# -------------- -------------- -------------- --------------
# The Transfer functions for Lambda


def kernel_linear(Lambda):
    return Lambda


def kernel_step(Lambda, k):
    Lambda_ind = Lambda.argsort()[-k:][::-1]
    lambda_cut = Lambda[Lambda_ind[k - 1]]

    Lambda_tilde = np.zeros(Lambda.shape)
    for i in range(len(Lambda)):
        if Lambda[i] >= lambda_cut:
            Lambda_tilde[i] = 1
        else:
            Lambda_tilde[i] = 0
    return Lambda_tilde


def kernel_linear_step(Lambda, k):
    Lambda_ind = Lambda.argsort()[-k:][::-1]
    lambda_cut = Lambda[Lambda_ind[k - 1]]

    Lambda_tilde = np.zeros(Lambda.shape)
    for i in range(len(Lambda)):
        if Lambda[i] >= lambda_cut:
            Lambda_tilde[i] = Lambda[i]
        else:
            Lambda_tilde[i] = 0
    return Lambda_tilde


def kernel_poly(Lambda, t):

    Lambda_tilde = np.power(Lambda, t)
    return Lambda_tilde


def kernel_poly_step(Lambda, r, p, q):
    # Getting the index of max values in array
    Lambda_ind = Lambda.argsort()[-r:][::-1]
    Lambda_tilde = Lambda
    for i in range(len(Lambda)):
        if i in Lambda_ind:
            Lambda_tilde[i] = np.power(Lambda[i], 1/p)
        else:
            Lambda_tilde[i] = np.power(Lambda[i], q)
    return Lambda_tilde


def clustring_kernel(X_train, X_unlabeled, X_test, gamma, k):
    """
    :param X: The complete data point
    :param gamma: RBF parameter
    :param k: number of eigenvector
    :return: The kernel phi(X)
    """

    # Concatenate data
    X = np.concatenate((X_train, X_test, X_unlabeled))

    # affitnity matrix
    # Set diagonal to zero
    K = rbf_kernel(X)
    np.fill_diagonal(K, 0)

    # D matrix
    sqrt_D = np.zeros(K.shape)
    for i, row in enumerate(K):
        row_sum = np.sum(row)
        sqrt_D[i, i] = np.power(row_sum, -0.5)

    # matrix L
    L = np.matmul(sqrt_D, np.matmul(K, sqrt_D))

    _, V = np.linalg.eigh(L)
    # take only k eigenvectors
    V = V[:, 0:k]
    S = []

    for i in range(len(V)):
        s = 0
        for j in range(k):
            s += np.square(V[i, j])
        S.append(s)

        for j in range(k):
            V[i, j] = V[i, j] / np.sqrt(S[i])

    V_train, V_test = V[0:X_train.shape[0], :], V[X_train.shape[0]:X_train.shape[0] + X_test.shape[0], :]

    return V_train, V_test


def cluster_kernel_extension(X_labeled, X_unlabeled, X_test, gamma, kernel, t):

    """
    :param X_labeled:  The labaled data points
    :param X_unlabeled: The unlabeled data points
    :param X_test: The test data points
    :param gamma: The RBF parameters
    :param kernel: The specific transfer function kernel to calculate the lambda tilde
    :return: The The kernel phi(X)

    """
    if X_test is not None:
        X = np.concatenate((X_labeled, X_test, X_unlabeled))
    else:
        X = np.concatenate((X_labeled, X_unlabeled))

    print("Cluster kernel start")

    K = rbf_kernel(X, gamma=gamma)

    # Set diagonal to one
    np.fill_diagonal(K, 1)

    # D matrix
    sqrt_D = np.zeros(K.shape)
    for i, row in enumerate(K):
        row_sum = np.sum(row)
        sqrt_D[i, i] = np.power(row_sum, (-0.5))
    # matrix L
    print("Computing L")
    L = np.matmul(np.matmul(sqrt_D, K), sqrt_D)

    # eigendecomposition of L
    # Lambda: eigenvalues, U: eigenvectors
    print("Computing eig")
    Lambda, U = np.linalg.eigh(L)

    # call kernel
    if kernel == "linear":
        Lambda_tilde = kernel_linear(Lambda)
    elif kernel == "step":
        Lambda_tilde = kernel_step(Lambda, k=50)
    elif kernel == "lin_step":
        Lambda_tilde = kernel_linear_step(Lambda, k=50)
    elif kernel == "poly":
        Lambda_tilde = kernel_poly(Lambda, t=2)
    elif kernel == "poly_step":
        Lambda_tilde = kernel_poly_step(Lambda, r=10, p=2, q=2)

    print("number cluster k :", np.count_nonzero(Lambda_tilde))
    L_tilde = np.matmul(U * Lambda_tilde, U.T)

    # compute D_tilde
    D_tilde = np.zeros(L_tilde.shape)
    np.fill_diagonal(D_tilde, 1 / (L_tilde.diagonal() + sys.float_info.min))

    # compute K_tilde
    sqrt_D_tilde = np.zeros(D_tilde.shape)
    np.fill_diagonal(sqrt_D_tilde, np.power(D_tilde.diagonal(), 0.5))
    K_tilde = np.dot(sqrt_D_tilde, L_tilde).dot(sqrt_D_tilde)

    # Filter K_tilde just for the labeled data set.
    K_tilde_labeled = K_tilde[0:X_labeled.shape[0], 0:X_labeled.shape[0]]

    if X_test is not None:
        K_tilde_test = K_tilde[X_labeled.shape[0]:X_labeled.shape[0] + X_test.shape[0], 0:X_labeled.shape[0]]

    print("passed the cluster kernels")

    if X_test is not None:
        return K_tilde_labeled, K_tilde_test
    else:
        return K_tilde_labeled, K[0:X_labeled.shape[0], 0:X_labeled.shape[0]]
