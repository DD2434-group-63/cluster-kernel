import numpy as np
from data import *
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt


N = targets.shape[0]
C = 10
p = 3
SIGMA = 3
KERNEL = "rbf"




def kernel_function(x, y):
    if KERNEL == 'linear':
        return np.dot(x,y)
    elif KERNEL == "poly":
        return np.power(np.dot(x, y) + 1, p)
    elif KERNEL == "rbf":
        return np.exp(-np.linalg.norm(x - y)**2) / (2 * SIGMA**2)


def pMat(x):
    n = targets.shape[0]
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P[i][j] = targets[i] * targets[j] * kernel_function(x[i], x[j])
    return np.array([P])


def objective(alpha, p_matrix):
    return 0.5 * np.dot(alpha, np.dot(alpha, p_matrix)) - np.sum(alpha)


def zerofun(alpha):
    return np.dot(alpha, targets)


def non_zero_alpha(alpha):
    sv_inputs = []
    sv_targets =[]
    non_zero_alphas = []
    for i,v in enumerate(alpha):
        if v > 1e-05:
            non_zero_alphas.append(v)
            sv_inputs.append(inputs[i])
            sv_targets.append(targets[i])
    return np.array(non_zero_alphas),np.array(sv_inputs),np.array(sv_targets)


def bias(alpha, non_zero_alphas,sv_inputs,sv_targets):
    b = 0
    for i in range(N):
        b += alpha[i]*targets[i]*kernel_function(sv_inputs[0],inputs[i])
    return b - sv_targets[0]


def indicator(x, y, b, alpha):
    s = [x,y]
    ind = 0
    for i in range(N):
        ind += alpha[i]*targets[i]*kernel_function(s,inputs[i])
    return ind - b


def decision_boundry():
    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)
    grid = np.array([[indicator(x, y, b) for x in xgrid] for y in ygrid])
    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
    return


def main():
    start = np.zeros(N)
    B = [(0, C) for b in range(N)]
    XC = {'type': 'eq', 'fun': zerofun}
    p_matrix = pMat(inputs)
    ret = minimize(objective, start,args=p_matrix[0], bounds=B, constraints=XC)
    alpha = ret['x']
    print(ret['success'])

    non_zero_alphas,sv_inputs,sv_targets = non_zero_alpha(alpha)
    b = bias(alpha, non_zero_alphas,sv_inputs,sv_targets)

    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)
    grid = np.array([[indicator(x, y, b, alpha) for x in xgrid] for y in ygrid])
    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
    plt.legend(["class A", "class B"])
 #   plt.title("Polynomial. p=4, c=10")
    plt.show()


if __name__ == '__main__':
    main()
