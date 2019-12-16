import argparse
import os

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

C = 10
p = 3
SIGMA = 3
KERNEL = "rbf"


def kernel_function(x, y):
    """
    Implementetion of kernel functions.
    """
    if KERNEL == 'linear':
        return np.dot(x, y)
    elif KERNEL == "poly":
        return np.power(np.dot(x, y) + 1, p)
    elif KERNEL == "rbf":
        return np.exp(-np.linalg.norm(x - y)**2) / (2 * SIGMA**2)


def pMat(inputs, targets):
    """
    Computes matrix of kernels times targets s.t.
    P_{ij} = t_i t_j K(x_i, x_j)
    """
    n = targets.shape[0]
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P[i][j] = targets[i] * targets[j] * kernel_function(inputs[i], inputs[j])
    return np.array([P])


def objective(alpha, p_matrix):
    """
    Optimization objective.
    """
    return 0.5 * np.dot(alpha, np.dot(alpha, p_matrix)) - np.sum(alpha)


def zerofun(alpha, targets):
    """
    Function to help implement constraint \sum_i \alpha_i\t_i = 0
    """
    return np.dot(alpha, targets)


def non_zero_alpha(alpha, inputs, targets):
    """
    Finds the support vectors given optimal alpha.
    """
    sv_inputs = []
    sv_targets = []
    non_zero_alphas = []
    for i, v in enumerate(alpha):
        if v > 1e-05:
            non_zero_alphas.append(v)
            sv_inputs.append(inputs[i])
            sv_targets.append(targets[i])
    return np.array(non_zero_alphas), np.array(sv_inputs), np.array(sv_targets)


def bias(alpha, inputs, targets, non_zero_alphas, sv_inputs, sv_targets):
    """
    Computes the bias given optimal alpha and support vectors.
    """
    b = 0
    N = inputs.shape[0]
    for i in range(N):
        b += alpha[i] * targets[i] * kernel_function(sv_inputs[0], inputs[i])
    return b - sv_targets[0]


def indicator(s, b, alpha, inputs, targets):
    """
    Predicts class of s.
    """
    ind = 0
    N = inputs.shape[0]
    for i in range(N):
        ind += alpha[i] * targets[i] * kernel_function(s, inputs[i])
    return ind - b


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

    # Initialization
    start = np.zeros(N)  # alpha initial value
    B = [(0, C) for b in range(N)]  # bounds on alpha
    XC = {'type': 'eq', 'fun': lambda alpha: zerofun(alpha, train_targets)}  # optimization constraints
    p_matrix = pMat(train_inputs, train_targets)  # matrix of kernel times targets (see assignment pdf)

    # Optimize
    ret = minimize(objective, start, args=p_matrix[0], bounds=B, constraints=XC)
    alpha = ret['x']
    print("Optimization succeeded:", ret['success'])

    # Find support vectors
    non_zero_alphas, sv_inputs, sv_targets = non_zero_alpha(alpha, train_inputs, train_targets)

    # Calculate bias
    b = bias(alpha, train_inputs, train_targets, non_zero_alphas, sv_inputs, sv_targets)

    # Predict on test set, and calculate error
    test_predictions = np.zeros(N_test)
    for n in range(N_test):
        test_predictions[n] = indicator(test_inputs[n, :], b, alpha, train_inputs, train_targets)

    test_predictions[test_predictions < 0] = -1
    test_predictions[test_predictions >= 0] = 1

    print(test_predictions)

    """
    # Plot results
    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)
    grid = np.array([[indicator([x, y], b, alpha, train_inputs, train_targets) for x in xgrid] for y in ygrid])
    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
    plt.legend(["class A", "class B"])
    plt.show()
    """
if __name__ == '__main__':
    main()
