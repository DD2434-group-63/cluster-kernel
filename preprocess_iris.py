import numpy as np
from  sklearn import  datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
inputs = iris.data
targets = iris.target

"""
# remove the last class
inputs = inputs[:100]
targets = targets[:100]
"""
# split into trainning and test set
train_x,test_inputs,train_y,test_targets=train_test_split(inputs, targets, test_size=.9)

# get index of the first sample from each class and use them as labeled training data
index_classA = next((i for i, j in enumerate(train_y) if j==0), None)
index_classB = next((i for i, j in enumerate(train_y) if j>0), None)
train_classA = np.asarray([train_x[index_classA]])
train_classB = np.asarray([train_x[index_classB]])
train_inputs = np.vstack([train_classA, train_classB])
train_targets = [0, 1]
# unlabeled
if index_classA > index_classB:
    train_unlabeled = np.delete(train_x, index_classA, 0)
    train_unlabeled = np.delete(train_unlabeled, index_classB, 0)
else:
    train_unlabeled = np.delete(train_x, index_classB, 0)
    train_unlabeled = np.delete(train_unlabeled, index_classA, 0)
test_classA = np.zeros((len(test_inputs),(len(test_inputs[0])))
test_classB = np.zeros((len(test_inputs),(len(test_inputs[0])))

for i in range(len(test_targets)):
    if test_targets[i] == 0:
        test_classA[i] = test_inputs[i]
    else:
        test_classB[i] = test_inputs[i]
for i in range(len(test_classA)-1,-1,-1):
    if sum(test_classA[i]) == 0:
        test_classA = np.delete(test_classA, i, 0)
    if sum(test_classB[i]) == 0:
        test_classB = np.delete(test_classB, i, 0)

test_classA = np.asarray(test_classA)
test_classB = np.asarray(test_classB)

path = 'data/preprocess_iris/'

np.save(path+'train_classA.npy', train_classA)
np.save(path+'train_classB.npy', train_classB)
np.save(path+'train_unlabeled.npy', train_unlabeled)
np.save(path+'test_classA.npy', test_classA)
np.save(path+'test_classB.npy', test_classB)
