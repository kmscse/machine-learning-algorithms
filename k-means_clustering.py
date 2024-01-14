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