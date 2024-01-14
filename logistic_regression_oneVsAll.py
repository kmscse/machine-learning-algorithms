import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

def sigmoid(z):
    g = 1/(1+np.exp(-z))
    return g

def costFunction(theta,X,y,Lambda):
    m = len(y)
    J = 0
    h = sigmoid(X@theta)
    epsilon = 1e-15
    J = 1/m*(-y.T@np.log(h+epsilon)-(1-y).T@np.log(1-h+epsilon))+Lambda/(2*m)*np.sum((theta[1:])**2)
    return J

def gradient(theta,X,y,Lambda):
    m = len(y)
    grad = np.zeros(theta.shape)
    h = sigmoid(X@theta)
    grad[0] = 1/m*(X[:,0].reshape(1,m)@(h-y))
    grad[1:] = 1/m*(X[:,1:].T@(h-y))+Lambda/m*theta[1:]
    return grad

def oneVsAll(X,y,nLabels,Lambda):
    m,n = X.shape
    all_theta = np.zeros((nLabels,n+1))
    X = np.concatenate((np.ones((m,1)), X),axis=1)
    #theta_ini = np.zeros((n+1,1)) #np.random.randint(0,1,(n+1,1))
    theta_ini=np.random.rand(n + 1, 1) * 1e-3
    for k in np.arange(1,nLabels+1):
        #theta_opt =  optimize.minimize(costFunction, theta_ini,(X,y,Lambda), method='BFGS',jac=gradient)
        theta_opt =  optimize.fmin_cg(costFunction, theta_ini,fprime=gradient,args=(X,(y == k).astype(int),Lambda),maxiter=1000,gtol=1e-5)
        all_theta[k-1,:] = theta_opt # (theta.x).reshape(1,n+1)
    return all_theta
        
def displayData(X,nCols):
    m,n = X.shape
    nRows = int(m/nCols)
    plt.figure(1,figsize=(6,10))
    for i in np.arange(m):
        plt.subplot(nRows,nCols,i+1)
        plt.imshow(X[i,:].reshape(20,20,order='F'),extent=[-1,1,-1,1]) 
        plt.axis('off')

def predictOneVsAll(all_theta, X):
    
    m, n = X.shape
    NLabels, o = all_theta.shape

    # You need to return the following variables correctly 
    p = np.zeros((m))

    # Add ones to the X data matrix
    X = np.concatenate((np.ones((m, 1)),X),axis=1)
    a=sigmoid(X@all_theta.T)
    for i in np.arange(m):
        #print(a[i,:])
        p[i] = np.where(a[i,:]==np.max(a[i,:]))[0][0]+1
    return p
   
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

theta_t = np.array([-2, -1, 1, 2])
X_t = np.concatenate((np.ones((5,1)),np.reshape(np.arange(1,16),(5,3),order='F')/10),axis=1)
y_t = np.array([1,0,1,0,1])
Lambda_t = 3
J = costFunction(theta_t, X_t, y_t, Lambda_t)
grad = gradient(theta_t, X_t, y_t, Lambda_t)
print(J)
print(grad)

nLabels = 10
Lambda = 0.1
all_theta = oneVsAll(X, y, nLabels, Lambda)
p = predictOneVsAll(all_theta, X)
accuracy = np.mean(1*(p==y))*100
print('\nTraining Set Accuracy: {:.2f}\n'.format(accuracy))

sel = np.random.permutation(m)
n = 20
sel = sel[0:n]
X_sel = X[sel,:]
displayData(X_sel,5)
p = predictOneVsAll(all_theta, X_sel)
p[np.where(p==10)]=0
k = sel//500
k[np.where(k==10)]=0
print('\t\tTarget\t\tPrediction')
for i in np.arange(n):
    print('\t\t{:d}\t\t\t{:d}'.format(k[i],int(p[i])))