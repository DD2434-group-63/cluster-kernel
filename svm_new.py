import numpy as np
import random as rnd

import kernel_functions as k

# Kernel hyperparameters
SIGMA = 3
POWER = 3


class SVM():
    """
    SVM implementation. uses SMO for training.
    """
    def __init__(self, max_iter=10000, kernel_type='linear', C=1.0, epsilon=0.001):
        self.kernels = {
            'linear': k.kernel_linear,
            'quadratic': k.kernel_quadratic,
            'poly': lambda x1, x2: k.kernel_poly(x1, x2, POWER),
            'rbf': lambda x1, x2: k.kernel_rbf(x1, x2, SIGMA)
        }
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.epsilon = epsilon

    def fit(self, inputs, targets):
        """
        Optimization
        """
        # Initialization
        n, d = inputs.shape[0], inputs.shape[1]
        alpha = np.zeros((n))
        kernel = self.kernels[self.kernel_type]
        count = 0
        while True:
            count += 1
            alpha_prev = np.copy(alpha)
            for j in range(0, n):

                i = self.get_rnd_int(0, n - 1, j)  # Get random int i ~= j
                inputs_i, inputs_j, targets_i, targets_j = inputs[i, :], inputs[j, :], targets[i], targets[j]
                k_ij = kernel(inputs_i, inputs_i) + kernel(inputs_j, inputs_j) - 2 * kernel(inputs_i, inputs_j)
                if k_ij == 0:
                    continue
                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
                (L, H) = self.compute_L_H(self.C, alpha_prime_j, alpha_prime_i, targets_j, targets_i)

                # Compute model parameters
                self.w = self.calc_w(alpha, inputs, targets)
                self.b = self.calc_b(inputs, targets, self.w)

                # Compute E_i, E_j
                E_i = self.E(inputs_i, targets_i, self.w, self.b)
                E_j = self.E(inputs_j, targets_j, self.w, self.b)

                # Set new alpha values
                alpha[j] = alpha_prime_j + float(targets_j * (E_i - E_j)) / k_ij
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)

                alpha[i] = alpha_prime_i + targets_i * targets_j * (alpha_prime_j - alpha[j])

            # Print progress
            print("Iteration {}, Objective = {}".format(count, self.objective(inputs, targets, alpha)))

            # Check convergence
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.epsilon:
                break

            if count >= self.max_iter:
                print("Iteration number exceeded the max of %d iterations" % (self.max_iter))
                return

        # Compute final model parameters
        self.b = self.calc_b(inputs, targets, self.w)
        if self.kernel_type == 'linear':
            self.w = self.calc_w(alpha, inputs, targets)

        # Get support vectors
        alpha_idx = np.where(alpha > 0)[0]
        support_vectors = inputs[alpha_idx, :]
        return support_vectors, count

    def objective(self, inputs, targets, alpha):

        kernel = self.kernels[self.kernel_type]
        result = np.sum(alpha)
        for i in range(inputs.shape[0]):
            for j in range(inputs.shape[0]):
                result += -0.5 * targets[i] * targets[j] * kernel(inputs[i, :], inputs[j, :]) * alpha[i] * alpha[j]

        return result

    def predict(self, inputs):
        return self.h(inputs, self.w, self.b)

    def calc_b(self, inputs, targets, w):
        b_tmp = targets - np.dot(w, inputs.T)
        return np.mean(b_tmp)

    def calc_w(self, alpha, inputs, targets):
        return np.dot(inputs.T, np.multiply(alpha, targets))

    # Prediction
    def h(self, X, w, b):
        return np.sign(np.dot(w, X.T) + b).astype(int)

    # Prediction error
    def E(self, x_k, y_k, w, b):
        return self.h(x_k, w, b) - y_k

    def compute_L_H(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        if(y_i != y_j):
            return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
        else:
            return (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))

    def get_rnd_int(self, a, b, z):
        i = z
        cnt = 0
        while i == z and cnt < 1000:
            i = rnd.randint(a, b)
            cnt = cnt + 1
        return i

    # Define kernels
    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)

    def kernel_quadratic(self, x1, x2):
        return (np.dot(x1, x2.T) ** 2)
