"""                      Week2 - Programming Assignment1

Instructions:
https://github.com/Kulbear/deep-learning-coursera/blob/master/Neural%20Networks%20and%20Deep%20Learning/Logistic%20Regression%20with%20a%20Neural%20Network%20mindset.ipynb
    * Do not use loops (for/while) in your code, unless the instructions explicitly ask you to do so.

to install PIL use this command:
sudo pip3 install -U Pillow

"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# index = 2
# plt.imshow(train_set_x_orig[index])
# plt.show()
# print ("y = " + str(train_set_y[:,index]) + ", it's a '" + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") +  "' picture.")

"""
Exercise: Find the values for:
- m_train (number of training examples)
- m_test (number of test examples)
- num_px (= height = width of a training image)
"""
# print (train_set_x_orig.shape)
# print("m_train = " + str(train_set_x_orig.shape[0]))
# print("m_test = " + str(test_set_x_orig.shape[0]))
# print("num_px = " + str(train_set_x_orig.shape[1]))


#train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[1]*train_set_x_orig.shape[2]*train_set_x_orig.shape[3],train_set_x_orig.shape[0]).T
#doesn't work!

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T # a tricky way to flatten a matrix X
# of shape (a,b,c,d) to a matrix X_flatten of shape (b*c*d, a)
#print(train_set_x_flatten.shape)
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T 
#print(test_set_x_flatten.shape)
#print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))


#standardize our dataset.
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.



def sigmoid(z):
    return 1 / (1 + np.exp(-z)) 

# print ("sigmoid(0) = " + str(sigmoid(0)))
# print ("sigmoid(9.2) = " + str(sigmoid(9.2)))


def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    return w, b

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """

    m = X.shape[1]
    y_hat = sigmoid(np.dot(w.T,X) + b)
    db = (1/m)*(np.sum(y_hat - Y))
    dw = (1/m)*(np.dot(X,(y_hat - Y).T))
    cost = (-1/m) * (np.dot(Y,np.log(y_hat).T) + np.dot((1-Y),np.log(1-y_hat).T))
    cost = np.squeeze(cost)
    return cost, dw, db


# w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
# cost, dw, db = propagate(w, b, X, Y)
# print ("dw = " + str(dw))
# print ("db = " + str(db))
# print ("cost = " + str(cost))


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    costs = []
    for _ in range(num_iterations):
        cost, dw, db = propagate(w, b, X, Y)
        costs.append(cost)
        # w -= learning_rate * dw
        # b -= learning_rate * db
        w = w - learning_rate * dw  
        b = b - learning_rate * db
    
    return costs, w, b, dw, db


w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
costs, w, b, dw, db = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
# print ("w = " + str(w))
# print ("b = " + str(b))
# print ("dw = " + str(dw))
# print ("db = " + str(db))


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    Y_hat = sigmoid(np.dot(w.T,X) + b)
    Y_prediction = np.round(Y_hat) #0 (if activation <= 0.5) or 1 (if activation > 0.5)
    return Y_prediction

print("predictions = " + str(predict(w, b, X)))


