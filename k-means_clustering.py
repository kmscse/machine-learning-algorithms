import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def computeCentroids(X, idx, K):
    # Useful variables
    m, n = X.shape

    # Initialize centroids matrix
    centroids = np.zeros((K, n))

    # Loop over every centroid and compute the mean of all points that belong to it
    for i in range(1, K + 1):
        # Find indices of data points assigned to centroid i
        index = np.where(idx == i)[0]

        # Compute the mean of the data points assigned to centroid i
        if len(index) > 0:
            centroids[i - 1, :] = np.mean(X[index, :], axis=0)

    return centroids

def featureNormalize(X):
    # Calculate the mean and standard deviation for each feature
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    # Normalize the features
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma

def findClosestCentroids(X, centroids):
    # Set K
    K = centroids.shape[0]

    # Initialize idx
    idx = np.zeros(X.shape[0], dtype=int)

    # Iterate over each example in X
    for i in range(X.shape[0]):
        # Compute the squared Euclidean distance to each centroid
        distances = np.sum((X[i] - centroids) ** 2, axis=1)

        # Find the index of the closest centroid
        idx[i] = np.argmin(distances) +1 # Add 1 to match MATLAB's indexing

    return idx

def kMeansInitCentroids(X, K):
    # Initialize centroids as zeros
    centroids = np.zeros((K, X.shape[1]))

    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])

    # Take the first K examples as centroids
    centroids = X[randidx[:K], :]

    return centroids


def pca(X):
    # Useful values
    m, n = X.shape

    # Initialize U and S
    U = np.zeros((n, n))
    S = np.zeros((n, n))

    # Compute the covariance matrix
    Sigma = (1 / m) * X.T @ X

    # Compute the eigenvectors and eigenvalues of the covariance matrix
    U, S, _ = np.linalg.svd(Sigma)

    return U, S

def plotDataPoints(X, idx, K):
    # Create a color palette
    palette = plt.cm.hsv(np.linspace(0, 1, K + 1))
    colors = palette[idx, :]

    # Plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=15)

    # Set labels and title (customize as needed)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Data Points Colored by Cluster Assignment')

    # Show the plot
    plt.show()