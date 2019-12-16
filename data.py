import numpy as np
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

""""
-   THe indicator is kind of a classifier if it returns 1 it belongs to class A and if -1 it belongs to class B. (func 3)

-   The Dual Formulation has some computational advantages by applying the kernel trick and 
eliminating the phi function and W. (func 4)

-   a kernel function and computes the scalar value corresponding to φ(⃗xi) · φ(⃗xj ). It transform the dataset
non-linearly for more complex decision boundary. 


-   The subset of the data points ⃗xi corresponding to non-zero values of αi are called the support vectors
because they are located exactly on the margins.










"""


np.random.seed(100)

classA = np. concatenate((np.random.randn(10, 2) * 0.5 + [0,0],
                          np.random.randn(10, 2) * 0.5 + [-1.5, 0.5]))
#classA = np.random.randn(50, 2) * 2 + [0.0, 0]
#classB = np.random.randn(50, 2) * 2 + [0.0, 0]

classB = np. concatenate((np.random.randn(20, 2) * 0.5 + [0,-1.5],
                          np.random.randn(10, 2) * 0.5 + [2,2],
                          np.random.randn(10, 2) * 0.5 + [-4,3]))

inputs = np.concatenate((classA, classB))
targets = np.concatenate(
    (np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

N = inputs.shape[0]  # Number of rows (samples)
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]  # data points for class A and B
targets = targets[permute]  # Target values for class A and B


plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
plt.axis('equal')  # Force same scale on both axes
# plt.savefig('svmplot.pdf') # Save a copy in a file

#plt.show()
