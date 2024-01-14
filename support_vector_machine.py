import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
import re
import csv 
import pandas as pd

def linearKernel(x1, x2):
    x1 = x1.reshape(-1)
    x2 = x2.reshape(-1)
    sim = x1.T@x2
    return sim

def gaussianKernel(x1, x2, sigma):
    x1 = x1.reshape(-1)
    x2 = x2.reshape(-1)

    sim = 0
    sim = np.exp(-np.sum((x1-x2)**2)/2/sigma**2)
    return sim

def getVocabList():
    vocabList = []
    with open("vocab.txt",mode="rt") as f:
        w = csv.reader(f,delimiter="\t")
        for row in w:
           vocabList.append(row[1])
    return vocabList 

def processEmail(email_contents):
        
    vocabList = getVocabList()
    
    word_indices = []
    
    email_contents = email_contents.lower()
    
    pattern = r'<[^<>]+>'
    replacement = r' '
    email_contents = re.sub(pattern, replacement, email_contents)
    
    pattern = '[0-9]+'
    replacement = 'number'
    email_contents = re.sub(pattern, replacement, email_contents)
    
    pattern = r'(http|https)://[^\s]*'
    replacement = 'httpaddr'
    email_contents = re.sub(pattern, replacement, email_contents)
    
    
    print('\n==== Processed Email ====\n\n')
    
    l = 0
    
    while len(email_contents)  != 0:
        delimiter_chars = r' @$/#.-:&*+=\[\]?!(){},\'">_<;%\r\n'

        # Split the string using the delimiter characters
        str_token, email_contents = re.split('[' + delimiter_chars + ']+', email_contents, 1)

        str_token = re.sub('[^a-zA-Z0-9]', '', str_token)
   
        stemmer = PorterStemmer()    
        try:
        # Apply the Porter stemmer to the string
            str_token = stemmer.stem(str_token.strip())
        except:
             str_token = ''
             # Continue with the next iteration if str is empty
        #if not str_token:
         #   continue
        
        # Skip the word if it is too short
        if len(str_token) < 1:
           continue
    
        for ii in range(len(vocabList)):
            if vocabList[ii]==str_token:
                word_indices.append(ii)
                
            if (l + len(str_token) + 1) > 78:
                print('\n')
                l = 0
        
            print('%s '.format(str_token))
            l += len(str_token) + 1
        print('\n\n=========================\n')
    return word_indices

def svmTrain(X, Y, C, kernelFunction, tol=1e-3, max_passes=5):
    m, n = X.shape

    # Map 0 to -1
    Y[Y == 0] = -1

    alphas = np.zeros(m)
    b = 0
    E = np.zeros(m)
    passes = 0
    eta = 0
    L = 0
    H = 0

    K = np.zeros((m, m))

    if kernelFunction.__name__ == 'linearKernel':
        K = X@X.T #np.dot(X, X.T)
    elif 'gaussianKernel' in kernelFunction.__name__:
        X2 = np.sum(X ** 2, axis=1)
        K = X2[:, np.newaxis] + X2[np.newaxis, :] - 2 * np.dot(X, X.T)
        sigma = kernelFunction.args[2]
        K = np.exp(-K / (2 * sigma ** 2))
    else:
        for i in range(m):
            for j in range(i, m):
                K[i, j] = kernelFunction(X[i, :], X[j, :])
                K[j, i] = K[i, j]

    print('\nTraining ...')
    dots = 12
    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(m):
            E[i] = b + np.sum(alphas * Y * K[:, i]) - Y[i]
            if ((Y[i] * E[i] < -tol and alphas[i] < C) or
                (Y[i] * E[i] > tol and alphas[i] > 0)):

                j = np.random.randint(m)
                while j == i:
                    j = np.random.randint(m)

                E[j] = b + np.sum(alphas * Y * K[:, j]) - Y[j]

                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]

                if Y[i] == Y[j]:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                else:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])

                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                alphas[j] = alphas[j] - (Y[j] * (E[i] - E[j])) / eta
                alphas[j] = np.clip(alphas[j], L, H)

                if np.abs(alphas[j] - alpha_j_old) < tol:
                    alphas[j] = alpha_j_old
                    continue

                alphas[i] = alphas[i] + Y[i] * Y[j] * (alpha_j_old - alphas[j])

                b1 = b - E[i] - Y[i] * (alphas[i] - alpha_i_old) * K[i, j] - \
                    Y[j] * (alphas[j] - alpha_j_old) * K[i, j]
                b2 = b - E[j] - Y[i] * (alphas[i] - alpha_i_old) * K[i, j] - \
                    Y[j] * (alphas[j] - alpha_j_old) * K[j, j]

                if 0 < alphas[i] and alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] and alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                num_changed_alphas += 1

        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0

        print('.', end='')
        dots += 1
        if dots > 78:
            dots = 0
            print()
            
    print(' Done! \n\n')

    idx = alphas > 0
    model = {
        'X': X[idx, :],
        'y': Y[idx],
        'kernelFunction': kernelFunction,
        'b': b,
        'alphas': alphas[idx],
        'w': np.dot((alphas * Y), X)
    }

    return model

def svmPredict(model, X):
    '''
    SVMPREDICT returns a vector of predictions using a trained SVM model
    (svmTrain). 
       pred = SVMPREDICT(model, X) returns a vector of predictions using a 
       trained SVM model (svmTrain). X is a mxn matrix where there each 
       example is a row. model is a svm model returned from svmTrain.
       predictions pred is a m x 1 column of predictions of {0, 1} values.
        
     Check if we are getting a column vector, if so, then assume that we only
     need to do prediction for a single example
    '''
    if (X.shape[1] == 1):
        # Examples should be in rows
        X = X.T
    
    # Dataset 
    m = X.shape[0]
    p = np.zeros((m, 1))
    pred = np.zeros((m, 1))
    
    if model["kernelFunction"] == 'linearKernel':
        # We can use the weights and bias directly if working with the 
        # linear kernel
        p = X @ model['w'] + model['b']
    elif  model["kernelFunction"] =='gaussianKernel':
        # Vectorized RBF Kernel
        # This is equivalent to computing the kernel on every pair of examples
        X1 = np.sum(X**2, axis=1)
        X2 = np.sum(model['X']**2, axis=1)
        s = X.shape[0]
        A = (X2.reshape((s,1)) + (-2*(X@(model['X']).T)[:,np.newaxis])).reshape((s,s))
        K = (X1.reshape((s,1)) + A[:,np.newaxis]).reshape((s,s))
        K = model['kernelFunction'](1, 0)**K
        K = model['y'].T@K[np.newaxis,:]
        K = model['alphas'].T@K[np.newaxis,:]
        p = np.sum(K, axis=1)
    else:
        # Other Non-linear kernel
        for i in np.arange(m):
            prediction = 0
            for j in np.arange(model['X'].shape[0]):
                prediction = prediction + model['alphas'][j] *(model['y'][j]).astype(np.float64)* model['kernelFunction']((X[i,:]).T,(model['X'][j,:]).T)
            p[i] = prediction + model['b']
    
    # Convert predictions into 0 / 1
    pred[np.where(p >= 0)] =  1
    pred[np.where(p <  0)] =  0
    
    return pred

def plotData(X,y):
    plt.figure
    plt.plot(X[np.where(y==1),0],X[np.where(y==1),1],'k+',linewidth = 2,markersize=7)
    plt.plot(X[np.where(y==0),0],X[np.where(y==0),1],'ko',markerfacecolor = 'y',markersize=7)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(['Admitted', 'Not admitted'])
    #plt.show()
    
def visualizeBoundaryLinear(X, y, model):
    '''
    VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by theSVM
       VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary 
       learned by the SVM and overlays the data on it
    '''
    w = model['w']
    b = model['b']
    xp = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
    yp = -(w[0]*xp + b)/w[1]
    plotData(X, y)
    plt.plot(xp, yp, '-b')

def visualizeBoundary(X, y, model):
    # VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
    # VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision
    # boundary learned by the SVM and overlays the data on it
    
    # Plot the training data on top of the boundary
    plotData(X, y)

    # Make classification predictions over a grid of values
    x1plot = np.linspace(min(X[:,0]), max(X[:,0]), 100)
    x2plot = np.linspace(min(X[:,1]), max(X[:,1]), 100)
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros_like(X1)
    
    for i in range(X1.shape[1]):
        this_X = np.column_stack((X1[:, i], X2[:, i]))
        vals[:, i] = svmPredict(model, this_X).reshape(-1)
    
    # Plot the SVM boundary
    plt.contour(X1, X2, vals, levels=[0.5], colors='b')
    plt.show()

def dataset3Params(X, y, Xval, yval):
    C_range = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_range = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    C = C_range[0]
    sigma = sigma_range[0]
    pred_error_old = 1

    for ii in range(len(C_range)):
        for jj in range(len(sigma_range)):
            model = svmTrain(X, y, C_range[ii], lambda x1, x2: gaussianKernel(x1, x2, sigma_range[jj]))
            predictions = svmPredict(model, Xval)
            pred_error_new = np.mean((predictions != yval).astype(np.float64))

            if pred_error_new < pred_error_old:
                C = C_range[ii]
                sigma = sigma_range[jj]
                pred_error_old = pred_error_new

    return C, sigma

def emailFeatures(word_indices):
    # Total number of words in the dictionary
    n = 1899

    # Initialize the feature vector
    x = np.zeros(n)

    # Set the corresponding indices to 1
    x[word_indices] = 1

    return x

# Usage example
word_indices = [60, 100, 33, 44, 10, 53, 60, 58, 5]
feature_vector = emailFeatures(word_indices)
print(feature_vector)

# Implement svmTrain, svmPredict, and gaussianKernel functions if not already defined

# Usage example
# C, sigma = dataset3Params(X, y, Xval, yval)

# Spam classifier
file_contents = open('emailSample1.txt', 'r').read()
word_indices = processEmail(file_contents)
print(word_indices)

features = emailFeatures(word_indices)
print('Length of feature vector:', len(features))
print('Number of non-zero entries:', np.sum(features > 0))

# Load ex6data3
df4 = pd.read_csv('ex5data4x.csv',header = None)
df4.head()
print(df4.head())
m,n = df4.shape
X4= np.array(df4.loc[:,0:n-2].values)
y4 = np.array(df4.loc[:,n-1].values,ndmin=2).reshape(-1).T
y4_copy = y4.copy()

# Load ex6data3
dfVal = pd.read_csv('ex5data4xVal.csv',header = None)
dfVal.head()
print(dfVal.head())
m,n = dfVal.shape
XVal= np.array(dfVal.loc[:,0:n-2].values)
yVal = np.array(dfVal.loc[:,n-1].values,ndmin=2).reshape(-1).T
yVal_copy = yVal.copy()

C = 0.1
model = svmTrain(X4, y4, C, lambda x1, x2: linearKernel(x1, x2))
p = svmPredict(model, X4)
print('Training Accuracy:', np.mean(np.double(p.reshape(-1) == y4_copy)) * 100)

p = svmPredict(model, XVal)
print('Test Accuracy:', np.mean((p.reshape(-1) == yVal_copy).astype(np.float64)) * 100)