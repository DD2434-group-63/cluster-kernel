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


def kernel_linear_step(Lambda):
    Lambda_ind = Lambda.argsort()[-2:][::-1]
    lambda_cut = Lambda[Lambda_ind[1]]

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
    Lambda[::-1].sort()
    Lambda_tilde = np.zeros(Lambda.shape)
    for i in range(len(Lambda)):
        if i <= r:
            Lambda_tilde[i] = np.power(Lambda_tilde[i], p)
        else:
            Lambda_tilde[i] = np.power(Lambda_tilde[i], q)
    return Lambda_tilde


def clustring_kernel(X, gamma, k):
    """
    :param X: The complete data point
    :param gamma: RBF parameter
    :param k: number of eigenvector
    :return: The kernel phi(X)
    """

    # affitnity matrix
    # Set diagonal to zero
    print("passed 1")
    K = rbf_kernel(X)
    print("passedd 2")
    np.fill_diagonal(K, 0)

    # D matrix
    D = np.zeros(K.shape, int)
    for i, row in enumerate(K):
        row_sum = np.sum(row)
        D[i, i] = row_sum

    # matrix L
    sqrt_D = np.power(D, (-0.5))
    L = np.dot(sqrt_D, K).dot(sqrt_D)

    _, V = np.linalg.eig(L)
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

    return V


def cluster_kernel_extension(X_labeled, X_unlabeled, X_test, gamma, kernel, t):

    """
    :param X_labeled:  The labaled data points
    :param X_unlabeled: The unlabeled data points
    :param X_test: The test data points
    :param gamma: The RBF parameters
    :param kernel: The specific transfer function kernel to calculate the lambda tilde
    :return: The The kernel phi(X)

    """
    X = np.concatenate((X_labeled, X_test, X_unlabeled))

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
    Lambda, U = np.linalg.eig(L)

    # call kernel
    if kernel == "linear":
        Lambda_tilde = kernel_linear(Lambda)
    elif kernel == "step":
        Lambda_tilde = kernel_step(Lambda, k=50)
    elif kernel == "lin_step":
        Lambda_tilde = kernel_linear_step(Lambda)
    elif kernel == "poly":
        Lambda_tilde = kernel_poly(Lambda, t)
    elif kernel == "poly_step":
        Lambda_tilde = kernel_poly_step(Lambda, r=50, p=2, q=2)

    print("number cluster k :", np.count_nonzero(Lambda_tilde))
    # L_tilde = U * Lambda_tilde * U.T
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

    K_tilde_test = K_tilde[X_labeled.shape[0]:X_labeled.shape[0] + X_test.shape[0], 0:X_labeled.shape[0]]

    K_tilde_unlabeled = K_tilde[X_labeled.shape[0] + X_unlabeled.shape[0]:
                                X_labeled.shape[0] + X_unlabeled.shape[0] + X_test.shape[0],
                                X_labeled.shape[0] + X_unlabeled.shape[0]:
                                X_labeled.shape[0] + X_unlabeled.shape[0] + X_test.shape[0]]

    print("passed the cluster kernels")

    print(K_tilde_labeled)
    print(K_tilde_test)

    return K_tilde_labeled, K_tilde_unlabeled, K_tilde_test
