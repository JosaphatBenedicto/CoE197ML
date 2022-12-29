import numpy as np
import random
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

def plotter(data, label=['train']):
    """ Plots data. """
    
    for d,l in zip(data, label):
        x, y = d
        plt.scatter(x, y, label=l)
    
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    
    plt.show()
    
def fcn(x, degree, coeffs):
    """
    Given x, solves for y given x and coefficients
    
    Input/s:
        x         : ndarray of input data
        degree    : degree of the best fitting polynomial
        coeffs    : polynomial coefficients
        
    Output/s:
        y         : output after evaluating function
    """
    
    assert(len(coeffs) == 3)
    
    
    y = coeffs[0] + coeffs[1]*x + coeffs[2]*x**2
    return y

valtrain = open("data_train.csv", "r").read().splitlines()
del valtrain[0]
random.shuffle(valtrain)

val_train, val_validate = valtrain[:50], valtrain[50:]

xtrain, ytrain=[],[]
for x in val_train:
    vals_train = x.split(",")
    xtrain.append(float(vals_train[0]))
    ytrain.append(float(vals_train[1]))

xvalidate, yvalidate=[],[]
for x in val_validate:
    vals_validate = x.split(",")
    xvalidate.append(float(vals_validate[0]))
    yvalidate.append(float(vals_validate[1]))

x, y = np.array(xtrain, dtype="float64"), np.array(ytrain, dtype="float64")
n_obs = x.shape[0]
if n_obs != y.shape[0]:
    raise ValueError("'x' and 'y' lengths do not match")
xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]

valtest = open("data_test.csv", "r").read().splitlines()

xtest, ytest=[],[]
for x in valtest[1:]:
    vals_test = x.split(",")
    xtest.append(float(vals_test[0]))
    ytest.append(float(vals_test[1]))

    