import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

def sigmoid(z):
    g = 1/(1+np.exp(-z))
    return g

def sigmoidGradient(z):
    g = np.zeros(z.shape)
    a = sigmoid(z)
    g = a*(1-a)
    return g

def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,
                                   X, y, Lambda):
    Theta1 = np.reshape(nn_params[0:hidden_layer_size * (input_layer_size + 1)],
                 (hidden_layer_size, (input_layer_size + 1)))
    Theta2 = np.reshape(nn_params[( (hidden_layer_size * (input_layer_size + 1))):],
                 (num_labels, (hidden_layer_size + 1)))
    m,n = X.shape
    J = 0;
    X = np.concatenate((np.ones((m,1)),X),axis=1)
    m,n = X.shape
    a1 = np.zeros((n,m))
    a2 = np.zeros((Theta1.shape[0]+1,m))
    a3 = np.zeros((num_labels,m))
    yk = np.zeros((num_labels,m))
    epsilon = 1e-15
    for ii in np.arange(m):
        a1[:,ii] = X[ii,:].T
        z2 = Theta1@a1[:,ii]
        a2[:,ii] = np.concatenate((np.array([1]),sigmoid(z2)))
        z3 = Theta2@a2[:,ii]
        a3[:,ii]= sigmoid(z3)
        h = a3[:,ii]
        yk[y[ii]-1,ii] = 1
        J += np.sum((-yk[:,ii]).T*np.log(h+epsilon)-(1-yk[:,ii]).T*np.log(1-h+epsilon))
    J = J/m
    b = np.sum(np.sum(Theta1[:,1:]**2))+np.sum(np.sum(Theta2[:,1:]**2))
    J += Lambda/(2*m)*b
    return J

