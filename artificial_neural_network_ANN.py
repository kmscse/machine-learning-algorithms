# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

# Sigmoid activation function
def sigmoid(z):
    g = 1/(1+np.exp(-z))
    return g

# Gradient of the sigmoid function
def sigmoidGradient(z):
    g = np.zeros(z.shape)
    a = sigmoid(z)
    g = a*(1-a)
    return g

# Cost function for the neural network
def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,
                                   X, y, Lambda):
    # Reshaping the parameters                                   
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
    # Forward propagation                                   
    for ii in np.arange(m):
        a1[:,ii] = X[ii,:].T
        z2 = Theta1@a1[:,ii]
        a2[:,ii] = np.concatenate((np.array([1]),sigmoid(z2)))
        z3 = Theta2@a2[:,ii]
        a3[:,ii]= sigmoid(z3)
        h = a3[:,ii]
        yk[y[ii]-1,ii] = 1
        # Cost computation
        J += np.sum((-yk[:,ii]).T*np.log(h+epsilon)-(1-yk[:,ii]).T*np.log(1-h+epsilon))
    J = J/m
    b = np.sum(np.sum(Theta1[:,1:]**2))+np.sum(np.sum(Theta2[:,1:]**2))
    J += Lambda/(2*m)*b
    return J

# Gradient computation for the neural network
def gradient(nn_params,input_layer_size,hidden_layer_size,num_labels,
                                   X, y, Lambda):
    # Reshaping the parameters                                   
    Theta1 = np.reshape(nn_params[0:hidden_layer_size * (input_layer_size + 1)],
                 (hidden_layer_size, (input_layer_size + 1)))
    Theta2 = np.reshape(nn_params[( (hidden_layer_size * (input_layer_size + 1))):],
                 (num_labels, (hidden_layer_size + 1)))
    m,n = X.shape
    J = 0;
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    X = np.concatenate((np.ones((m,1)),X),axis=1)
    m,n = X.shape
    a1 = np.zeros((n,m))
    a2 = np.zeros((Theta1.shape[0]+1,m))
    a3 = np.zeros((num_labels,m))
    yk = np.zeros((num_labels,m))
    # Forward propagation                                   
    for ii in np.arange(m):
        a1[:,ii] = X[ii,:].T
        z2 = Theta1@a1[:,ii]
        a2[:,ii] = np.concatenate((np.array([1]),sigmoid(z2)))
        z3 = Theta2@a2[:,ii]
        a3[:,ii]= sigmoid(z3)
        h = a3[:,ii]
        yk[y[ii]-1,ii] = 1
        # Cost computation
        J += np.sum((-yk[:,ii]).T*np.log(h)-(1-yk[:,ii]).T*np.log(1-h))
    J = J/m
    b = np.sum(np.sum(Theta1[:,1:]**2))+np.sum(np.sum(Theta2[:,1:]**2))
    J += Lambda/(2*m)*b

    # Backpropagation                                   
    D1 = np.zeros(Theta1.shape)
    D2 = np.zeros(Theta2.shape)
    for ii in np.arange(m):
        d3 = a3[:,ii] - yk[:,ii]
        d2 = (Theta2.T@d3)*sigmoidGradient(np.concatenate((np.array([1]),(Theta1@a1[:,ii]).reshape(-1))))
        D2 += d3.reshape(num_labels,1)@(a2[:,ii].reshape(hidden_layer_size+1,1)).T
        D1 += d2[1:].reshape(hidden_layer_size,1)@(a1[:,ii].reshape(input_layer_size+1,1)).T
    D2[:,0] = 1/m*D2[:,0]
    D1[:,0] = 1/m*D1[:,0]
    
    D2[:,1:] = 1/m*D2[:,1:] + Lambda/m *Theta2[:,1:]
    D1[:,1:] = 1/m*D1[:,1:] + Lambda/m *Theta1[:,1:]
    Theta1_grad = D1;
    Theta2_grad = D2;
    grad = np.concatenate((Theta1_grad.reshape(-1),Theta2_grad.reshape(-1)),axis=0)
    return grad

# Function to initialize weights for debugging
def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros((fan_out, 1 + fan_in))
    W = np.reshape(np.sin(np.arange(0,W.size)), W.shape) / 10
    return W

# Function to compute numerical gradient
def computeNumericalGradient(nn_params,input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):
    theta = nn_params
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for p in np.arange(theta.size):
        perturb[p] = e
        loss1 = nnCostFunction(nn_params-perturb, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
        loss2 = nnCostFunction(nn_params+perturb, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0
    return numgrad

# Function to check gradients
def checkNNGradients(Lambda):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
    
    X  = debugInitializeWeights(m, input_layer_size - 1)
    y  = 1 + np.mod(np.arange(m), num_labels).T
    nn_params = np.concatenate((Theta1.reshape(-1), Theta2.reshape(-1)),axis=0)
    grad = gradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
    numgrad = computeNumericalGradient(nn_params,input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
    
    print(np.concatenate((numgrad,grad)))
    print('The above two columns you get should be very similar.\n \
          (Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n')
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad);

    print('If your backpropagation implementation is correct, then \n \
         the relative difference will be small (less than 1e-9). \n \
         \nRelative Difference: {:.3e}\n'.format(diff))

# Function to display data
def displayData(X,nCols):
    m,n = X.shape
    nRows = int(m/nCols)
    plt.figure(1,figsize=(6,10))
    for i in np.arange(m):
        plt.subplot(nRows,nCols,i+1)
        plt.imshow(X[i,:].reshape(20,20,order='F'),extent=[-1,1,-1,1]) 
        plt.axis('off')

# Function to predict output
def predict(Theta1,Theta2, X):
    
    m, n = X.shape
    NLabels, o = Theta2.shape

    # You need to return the following variables correctly 
    p = np.zeros((m))
    h1 = sigmoid(np.concatenate((np.ones((m, 1)),X),axis=1)@Theta1.T)
    h2 = sigmoid(np.concatenate((np.ones((m, 1)),h1),axis=1)@Theta2.T);
    for i in range(m):
        p[i] = np.where(h2[i,:]==np.max(h2[i,:]))[0][0]+1
    return p

# Loading and preparing data
df = pd.read_csv('ex3data1.csv',header = None)
df.head()
print(df.head())
X= np.array(df.loc[:,0:399].values)
y = np.array(df[400].values,ndmin=2).reshape(-1).T
m, n = X.shape

# randomly select 100 data points
randIndex = np.random.permutation(m)
sel = X[randIndex[0:100],:]
displayData(sel,10)

# Neural network parameters
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)

# theta_ini=np.random.rand(n + 1, 1) * 1e-3
epsilon_init = 0.12
initial_Theta1 = np.random.rand(hidden_layer_size,input_layer_size+1)* 2 * epsilon_init - epsilon_init
initial_Theta2 = np.random.rand(num_labels,hidden_layer_size+1)* 2 * epsilon_init - epsilon_init

initial_nn_params = np.concatenate((initial_Theta1.reshape(-1), initial_Theta2.reshape(-1)),axis=0)

# Checking gradients
Lambda = 0
checkNNGradients(Lambda)

Lambda = 3
checkNNGradients(Lambda)

# Optimizing neural network parameters
sol_opt =  optimize.fmin_cg(nnCostFunction,initial_nn_params,fprime=gradient,
                                args=(input_layer_size,hidden_layer_size,
                                      num_labels,X,y.astype(int),Lambda),
                                maxiter=200,gtol=1e-5)
theta_opt = sol_opt

Theta1 = np.reshape(theta_opt[0:hidden_layer_size * (input_layer_size + 1)],
                 (hidden_layer_size, (input_layer_size + 1)))
Theta2 = np.reshape(theta_opt[( (hidden_layer_size * (input_layer_size + 1))):],
                 (num_labels, (hidden_layer_size + 1)))

# Predicting outputs
p = predict(Theta1, Theta2, X)
accuracy = np.mean(1*(p==y))*100
print('\nTraining Set Accuracy: {:.2f}\n'.format(accuracy))

# Displaying predicted data
sel = np.random.permutation(m)
n = 20
sel = sel[0:n]
X_sel = X[sel,:]
displayData(X_sel,5)
p = predict(Theta1,Theta2, X_sel)
p[np.where(p==10)]=0
k = sel//500
k[np.where(k==10)]=0
print('\t\tTarget\t\tPrediction')
for i in np.arange(n+1):
    print('\t\t{:d}\t\t\t{:d}'.format(k[i],int(p[i])))
