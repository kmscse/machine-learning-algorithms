import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def multivariateGaussian(X, mu, Sigma2):
    n = len(mu)

    if Sigma2.ndim == 1:
        Sigma2 = np.diag(Sigma2)

    X_minus_mu = X - mu.reshape(1, -1)
    p1 = (2 * np.pi) ** (- n/ 2)
    p2 = np.linalg.det(Sigma2) ** (-0.5)
    p3 = np.exp(-0.5 * np.sum(np.dot(X_minus_mu, np.linalg.pinv(Sigma2)) * X_minus_mu, axis=1))

    p = p1 * p2 * p3

    return p

def estimateGaussian(X):
    # Useful variables
    m, n = X.shape

    # Initialize variables to store mean and variance
    mu = np.zeros(n)
    sigma2 = np.zeros(n)

    # Compute mean of the data for each feature
    mu = np.mean(X, axis=0)

    # Compute variance of the data for each feature
    sigma2 = np.var(X, axis=0) * (m - 1) / m

    return mu, sigma2

# Visualize the fit
def visualizeFit(X, mu, sigma2):
    x1, x2 = np.meshgrid(np.arange(0, 35, 0.5), np.arange(0, 35, 0.5))
    X_grid = np.column_stack((x1.flatten(), x2.flatten()))
    Z = multivariateGaussian(X_grid, mu, sigma2)
    Z = Z.reshape(x1.shape)

    plt.figure()
    plt.plot(X[:, 0], X[:, 1], 'bx')
    plt.axis([0, 30, 0, 30])
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.contour(x1, x2, Z, levels=[10**exp for exp in range(-20, 0, 3)], colors='k')

    

# Find the best threshold
def selectThreshold(yval, pval):
    best_epsilon = 0
    best_F1 = 0
    step_size = (max(pval) - min(pval)) / 1000
    for epsilon in np.arange(min(pval), max(pval), step_size):
        predictions = (pval < epsilon).astype(np.float64)
        tp = np.sum((predictions == 1).astype(np.float64))
        fp = np.sum(((predictions == 1) & (yval == epsilon)).astype(np.float64))
        fn = np.sum(((predictions == 0) & (yval == epsilon)).astype(np.float64))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1 = 2 * precision * recall / (precision + recall)
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
    return best_epsilon, best_F1