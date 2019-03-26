"""                      Week3 - Programming Assignment 2

https://github.com/Kulbear/deep-learning-coursera/blob/master/Neural%20Networks%20and%20Deep%20Learning/Planar%20data%20classification%20with%20one%20hidden%20layer.ipynb

notes :
Implement a 2-class classification neural network with a single hidden layer
Implement forward and backward propagation

Instructions:

"""


import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1) # set a seed so that the results are consistent

X, Y = load_planar_dataset()
# print(X.shape) # contains your features (x1, x2)
# print(Y.shape) # contains your labels (red:0, blue:1).
# print(X.size)
# print(Y.size)
# plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral)
# plt.show()



# # first build a logistic regression classifier in order to compare with neural network output
# # Train the logistic regression classifier
# clf = sklearn.linear_model.LogisticRegressionCV(cv=5)
# clf.fit(X.T, Y.T.ravel())
# # Plot the decision boundary for logistic regression
# plot_decision_boundary(lambda x: clf.predict(x), X, Y)
# plt.title("Logistic Regression")
# plt.show()
# # Print accuracy
# LR_predictions = clf.predict(X.T)
# print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
#        '% ' + "(percentage of correctly labelled datapoints)")

# Interpretation: The dataset is not linearly separable, so logistic regression doesn't perform well.
# Hopefully a neural network will do better. Let's try this now!

def layer_sizes(X, Y):
    """
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0] # size of input layer
    n_h = 4          # size of hidden layer
    n_y = Y.shape[0] # size of output layer
    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    """
    Returns:
        W1 -- weight matrix of shape (n_h, n_x)
        b1 -- bias vector of shape (n_h, 1)
        W2 -- weight matrix of shape (n_y, n_h)
        b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.
    b1 = np.zeros((n_h,1))
    b2 = np.zeros((n_y,1))
    W1 = np.random.randn(n_h, n_x) * 0.01
    W2 = np.random.randn(n_y, n_h) * 0.01
    return b1, W1, b2, W2

n_x, n_h, n_y = layer_sizes(X, Y)
b1, W1, b2, W2 = initialize_parameters(n_x, n_h, n_y)


def forward_propagation(X, b1, W1, b2, W2):
    """
    Returns:
    A2 -- The sigmoid output of the second activation
    "Z1", "A1", "Z2" and "A2"
    """
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)

    return Z1, A1, Z2, A2
    
# Z1, A1, Z2, A2 = forward_propagation(X, b1, W1, b2, W2)

def compute_cost(A2, Y, W1, W2):
    """
    Computes the cross-entropy cost given in equation 
    $$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large{(} \small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right) \large{)} \small\tag{13}$$
    """
    m = Y.shape[1] # number of example
    # print(Y.shape)
    # print(A2.shape)
    cost = (-1/m) * ( np.dot(np.log(A2),Y.T) + np.dot(np.log(1 - A2),(1 - Y).T) )
    cost = np.squeeze(cost)     
    return cost


def backward_propagation(b1, W1, b2, W2, Z1, A1, Z2, A2, X, Y):
    """
    Returns:
        dW1, db1, dW2, db2
    """  
    m = X.shape[1]
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2)) # tanh(z1) = A1  (tanh()).prime = 1 - tanh()**2
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate=1.2):
    """
    Updates parameters using the gradient descent 
    """
    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2
    return W1, b1, W2, b2


