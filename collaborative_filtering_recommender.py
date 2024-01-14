import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.io import loadmat

def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, _lambda):
    # Unfold the U and W matrices from params
    X = params[:num_movies * num_features].reshape(num_movies, num_features)
    Theta = params[num_movies * num_features:].reshape(num_users, num_features)

    # Initialize variables
    J = 0
    X_grad = np.zeros_like(X)
    Theta_grad = np.zeros_like(Theta)
    # Calculate the cost J (without regularization)
    error = (X @ Theta.T - Y) * R
    J = 0.5 * np.sum(error**2)

    # Regularization terms
    reg_term = (_lambda / 2) * (np.sum(Theta**2) + np.sum(X**2))

    # Add regularization to the cost J
    J += reg_term
    
    X_grad = error @ Theta + _lambda * X
    Theta_grad = error.T @ X + _lambda * Theta

    grad = np.concatenate((X_grad.ravel(), Theta_grad.ravel()))
    
    return J, grad

def grad(params, Y, R, num_users, num_movies, num_features, _lambda):
    # Unfold the U and W matrices from params
    X = params[:num_movies * num_features].reshape(num_movies, num_features)
    Theta = params[num_movies * num_features:].reshape(num_users, num_features)

    X_grad = np.zeros_like(X)
    Theta_grad = np.zeros_like(Theta)

    # Calculate the cost J (without regularization)
    error = (X @ Theta.T - Y) * R

    # Compute gradients (with regularization)
    X_grad = error @ Theta + _lambda * X
    Theta_grad = error.T @ X + _lambda * Theta

    grad = np.concatenate((X_grad.ravel(), Theta_grad.ravel()))

    return grad

def load_movie_list():
    # Read the fixed movie list from 'movie_ids.txt' and return a list of movie names

    movie_list = []
    
    with open('movie_ids.txt', 'r', encoding='ISO-8859-1') as file:
        for line in file:
            # Split the line into an index and movie name
            idx, movie_name = line.strip().split(None, 1)
            # Append the movie name to the list
            movie_list.append(movie_name)
    
    return movie_list

# Normalize ratings
def normalize_ratings(Y, R):
    Ymean = np.sum(Y, axis=1) / np.sum(R, axis=1)
    Ynorm = (Y.T - Ymean).T
    return Ynorm, Ymean


# Load the dataset (ex8data1.mat)
data1 = loadmat('ex8_movies.mat')
Y = data1['Y']
R = data1['R']
# From the matrix, compute statistics like average rating.
print(f'Average rating for movie 1 (Toy Story): {np.mean(Y[0, R[0, :]])} / 5\n\n')

# Visualize the ratings matrix by plotting it with imshow
plt.imshow(Y, cmap='YlGnBu', aspect='auto')
plt.ylabel('Movies')
plt.xlabel('Users')
plt.colorbar()
plt.show()

plt.imshow(R, cmap='YlGnBu', aspect='auto')
plt.colorbar()
plt.show()

# Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
# Load your data and parameters here
data2 = loadmat('ex8_movieParams.mat')
X = data2['X']
Theta = data2['Theta']
num_featuresX = data2['num_features']
num_movies = data2['num_movies']
num_users = data2['num_users']

# Reduce the data set size so that this runs faster
num_users = 4
num_movies = 5
num_features = 3
X = X[:num_movies, :num_features]
Theta = Theta[:num_users, :num_features]
Y = Y[:num_movies, :num_users]
R = R[:num_movies, :num_users]



params = np.concatenate((X.ravel(), Theta.ravel()))
J = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 0)
print(f'Cost at loaded parameters: {J}')

# Check gradients (if needed)
# You can check gradients using appropriate methods or libraries

# Evaluate cost function with regularization
_lambda = 1.5
J = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, _lambda)
print(f'Cost at loaded parameters (lambda = {_lambda}): {J}')

# Load movie list
movieList = load_movie_list()

# Initialize my ratings
my_ratings = np.zeros(1682)

# Set ratings for movies you've seen
my_ratings[916] = 2 # Lost in Space (1998) Rating: 5.2 out of 10
my_ratings[912] = 3 # U.S. Marshalls (1998) Rating: 6.6 out of 10
my_ratings[1126] = 3 # Old Man and the Sea, The (1958) Rating: 6.9 out of 10
my_ratings[1211] = 3 # Blue Sky (1994) Rating: 6.4 out of 10
my_ratings[1310] = 3 # Walk in the Sun, A (1945) Rating: 6.9 out of 10
my_ratings[1507] = 3 # Three Lives and Only One Death (1996) Rating: 6.8 out of 10
my_ratings[1529] = 4 # Underground (1995) Rating: 8.1 out of 10
my_ratings[1590] = 2 # To Have, or Not (1995) Rating: 6.6 out of 10
my_ratings[1599] = 3 # Someone Else's America (1995) Rating: 6.9 out of 10
my_ratings[1618] = 2 # King of New York (1990) Rating 6.9 out of 10
my_ratings[1681] = 5 # You So Crazy (1994) Rating 6.6 out of 10

print('\n\nNew user ratings:')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(f'Rated {my_ratings[i]} for {movieList[i]}')

# Load data
# Load data again to reset
data1 = loadmat('ex8_movies.mat')
Y = data1['Y']
R = data1['R']

# Add your ratings to the data matrix
Y = np.column_stack((my_ratings, Y))
R = np.column_stack((my_ratings != 0, R))

Ynorm, Ymean = normalize_ratings(Y, R)

# Useful values
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10

# Set initial parameters (Theta, X)
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)
initial_parameters = np.concatenate((X.ravel(), Theta.ravel()))

# Set options for minimize
options = {'maxiter': 100}

# Set regularization parameter
_lambda = 10

# Minimize the cost function
result = minimize(lambda x: cofiCostFunc(x, Ynorm, R, num_users, num_movies, num_features, _lambda),
                  initial_parameters, method='L-BFGS-B', jac=True, options=options)

# Unfold the returned theta back into U and W
theta = result.x
X = theta[:num_movies * num_features].reshape(num_movies, num_features)
Theta = theta[num_movies * num_features:].reshape(num_users, num_features)

# Make recommendations
h = X @ Theta.T
my_predictions = h[:, 0] + Ymean

# Sort predictions in descending order
sorted_indices = np.argsort(my_predictions)[::-1]

print('\nTop recommendations for you:')
for i in range(10):
    j = sorted_indices[i]
    print(f'Predicting rating {my_predictions[j]:.1f} for movie {movieList[j]}')

print('\nOriginal ratings provided:')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(f'Rated {my_ratings[i]} for {movieList[i]}')