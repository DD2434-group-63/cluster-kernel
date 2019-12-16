import numpy as np


def kernel_linear(x1, x2):
    return np.dot(x1, x2.T)


def kernel_quadratic(x1, x2):
    return (np.dot(x1, x2.T) ** 2)


def kernel_poly(x1, x2, p):
    return np.power(np.dot(x1, x2) + 1, p)


def kernel_rbf(x1, x2, sigma):
    return np.exp(-np.linalg.norm(x1 - x2)**2) / (2 * sigma**2)
