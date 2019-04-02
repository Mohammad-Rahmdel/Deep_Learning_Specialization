"""                      Week1 - Programming Assignment 3
https://github.com/Kulbear/deep-learning-coursera/blob/master/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Gradient%20Checking.ipynb
"""

import numpy as np
from testCases import *
from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector

def forward_propagation_n(X, Y, parameters):
    """
    Implements the forward propagation (and computes the cost) presented in Figure 3.
    
    Arguments:
    X -- training set for m examples
    Y -- labels for m examples 
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
            
    Returns:
    cost -- the cost function (logistic cost for one example)
    """

    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cost = (1/m) * sum(np.squeeze((-np.log(A3)* Y) + (-np.log(1 - A3))*(1 - Y)))
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    
    return cost, cache
    

def reluDerivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x


def backward_propagation_n(X, Y, cache):
    """
    Implement the backward propagation presented in figure 2.
    
    Arguments:
    X -- input datapoint, of shape (input size, 1)
    Y -- true "label"
    cache -- cache output from forward_propagation_n()
    
    Returns:
    gradients -- A dictionary with the gradients of the cost with respect to each parameter, activation and pre-activation variables.
    """
    
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1 / m * np.dot(dZ3, A2.T)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = dA2 * reluDerivative(Z2)
    dW2 = 1 / m * np.dot(dZ2, A1.T) 
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * reluDerivative(Z1)
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True) 
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients



def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
    
    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters. 
    x -- input datapoint, of shape (input size, 1)
    y -- true "label"
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    

    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
               
    for i in range(num_parameters):
        theta_plus = np.squeeze(np.copy(parameters_values))      
        theta_plus[i] += epsilon

        theta_minus = np.squeeze(np.copy(parameters_values))  
        theta_minus[i] -= epsilon
    
        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_plus))
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_minus))
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
        

    difference_numerator = np.linalg.norm(gradapprox - grad)
    difference_denominator = np.linalg.norm(gradapprox) + np.linalg.norm(grad)
    difference = difference_numerator / difference_denominator

    if difference > 1e-7:
        print("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference


""" notes : 
-Gradient Checking is slow! Approximating the gradient with dJ/d(theta) is computationally costly.
For this reason, we don't run gradient checking at every iteration during training. 
Just a few times to check if the gradient is correct. 

-Gradient Checking, at least as we've presented it, doesn't work with dropout. You would usually run the gradient check 
algorithm without dropout to make sure your backprop is correct, then add dropout. 

-Gradient checking is slow, so we don't run it in every iteration of training. You would usually run it 
only to make sure your code is correct, then turn it off and use backprop for the actual learning process. 
"""
 
