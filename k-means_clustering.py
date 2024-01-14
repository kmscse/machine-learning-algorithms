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
    
def plotProgresskMeans(X, centroids, previous, idx, K, i):
    # Plot the examples with colors assigned to each centroid
    plotDataPoints(X, idx, K)

    # Plot the centroids as black x's
    plt.plot(centroids[:, 0], centroids[:, 1], 'kx', markersize=10, linewidth=3)
    

    # Plot lines connecting previous and current centroids
    for j in range(centroids.shape[0]):
        plt.plot([previous[j, 0], centroids[j, 0]], [previous[j, 1], centroids[j, 1]], 'k-')

    # Title
    plt.title(f'Iteration number {i}')

    # Show the plot
    plt.show()
    
def projectData(X, U, K):
    # Initialize Z with zeros
    Z = np.zeros((X.shape[0], K))

    # Get the reduced U matrix containing the top K eigenvectors
    U_reduce = U[:, :K]

    # Project the data onto the reduced dimensional space
    Z = X @ U_reduce

    return Z


def recoverData(Z, U, K):
    # Initialize X_rec with zeros
    X_rec = np.zeros((Z.shape[0], U.shape[0]))

    # Get the reduced U matrix containing the top K eigenvectors
    U_reduce = U[:, :K]

    # Project back onto the original space using the top K eigenvectors
    X_rec = Z @ U_reduce.T

    return X_rec


def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m, dtype=int)

    # Plot the data if we are plotting progress
    if plot_progress:
        import matplotlib.pyplot as plt
        plt.figure()

    # Run K-Means
    for i in range(max_iters):
        # Output progress
        print(f'K-Means iteration {i + 1}/{max_iters}...')
        
        # For each example in X, assign it to the closest centroid
        idx = findClosestCentroids(X, centroids)

        # Optionally, plot progress here
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            input('Press enter to continue.\n')

        # Given the memberships, compute new centroids
        centroids = computeCentroids(X, idx, K)

    # Hold off if we are plotting progress
    if plot_progress:
        plt.show()

    return centroids, idx

def displayData(X, example_width=None):
    # Set example_width automatically if not passed in
    if example_width is None:
        example_width = int(np.round(np.sqrt(X.shape[1])))

    # Compute rows, cols
    m, n = X.shape
    example_height = n // example_width

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    # Between images padding
    pad = 1

    # Setup blank display
    display_array = -np.ones((pad + display_rows * (example_height + pad),
                              pad + display_cols * (example_width + pad)))

    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex >= m:
                break

            # Copy the patch
            max_val = np.max(np.abs(X[curr_ex, :]))
            display_array[pad + j * (example_height + pad):pad + j * (example_height + pad) + example_height,
                          pad + i * (example_width + pad):pad + i * (example_width + pad) + example_width] = (
                X[curr_ex, :].reshape((example_height, example_width)) / max_val
            )
            curr_ex += 1

    # Display Image
    plt.figure()
    h = plt.imshow(display_array, cmap='gray', extent=[-1, 1, -1, 1])

    # Do not show axis
    plt.axis('off')

    plt.show()

    return h, display_array



def drawLine(p1, p2, *args, **kwargs):
    # Create a line from point p1 to point p2
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], *args, **kwargs)


# Load an example dataset
data = loadmat('ex7data2.mat')
X = data['X']

# Select an initial set of centroids
K = 4
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Find the closest centroids for the examples using the initial_centroids
idx = findClosestCentroids(X, initial_centroids)
print('Closest centroids for the first 3 examples:', idx[:3])

# Compute means based on the closest centroids found in the previous part

centroids = computeCentroids(X, idx, K)
print('Centroids computed after initial finding of closest centroids:')
print(centroids)

# Settings for running K-Means
max_iters = 10
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Run K-Means algorithm
fig, ax = plt.subplots()
plotProgresskMeans(X, initial_centroids, initial_centroids, idx, K, 1)
plt.xlabel('Press ENTER in command window to advance', fontweight='bold', fontsize=14)
runkMeans(X, initial_centroids, max_iters, True)
plt.close(fig)

# Example for image compression
A = plt.imread('colorful_image.jpg')

# Display the original image
plt.figure()
plt.imshow(A)
plt.title('Original')

A = A / 255.0  # Normalize image

img_size = A.shape
X = A.reshape(-1, 3)

K = 5
max_iters = 10

# Initialize centroids and run K-Means

initial_centroids = kMeansInitCentroids(X, K)
centroids, _ = runkMeans(X, initial_centroids, max_iters)

# Find closest cluster members

idx = findClosestCentroids(X, centroids)-1
X_recovered = centroids[idx, :]

# Reshape the recovered image into proper dimensions
X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3)

# Display the compressed image
fig = plt.figure()
ax = fig.add_subplot(121)


ax.imshow(A)
ax.set_title('Original')
ax.set_aspect('equal','box')

ax=fig.add_subplot(122)
ax.imshow(X_recovered)
ax.set_title(f'Compressed, with {K} colors.')
ax.set_aspect('equal','box')

plt.show()