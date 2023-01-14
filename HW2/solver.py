import numpy as np
import pandas as pd
import random
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import tinygrad.nn.optim as optim
from tinygrad.tensor import Tensor

#Loss function
def calc_loss(y_pred, y_gt):
    """
    Calculate the MSE loss
    
    Input/s:
        y_pred : predicted y of model
        y_gt   : ground truth y
    """
    mse = ((y_pred-y_gt)**2).mean()
    return mse

#Gradient


#Function Placeholder
def fcn(xval, yval, coeffs):
    return None


#Prepare Dataset
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

xtrain = Tensor(xtrain, requires_grad=False)
xvalidate = Tensor(xvalidate, requires_grad=False)
xtest = Tensor(xtest, requires_grad=False)
ytrain = Tensor(ytrain, requires_grad=False)
yvalidate = Tensor(yvalidate, requires_grad=False)    
ytest = Tensor(ytest, requires_grad=False)


#Prepare Model
class Datavalues:
    def __init__(self, x_val, y_val):
        self.x_val = x_val
        self.y_val = y_val
        
    def forward(self, x):
        return x.dot(self.x_val).relu().dot(self.y_val).logsoftmax()


model = Datavalues(xtrain,ytrain)
optim = optim.SGD([model.x_val, model.y_val], lr=0.001)


#Parameters
batch = 4
lr = 0.0001
max_epochs = 5000
deg = []
coeffs = [] 


out = model.forward(xtrain)
loss = out.mul(ytrain).mean()
optim.zero_grad()
loss.backward()
optim.step()