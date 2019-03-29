"""                      Week4 - Programming Assignment 3 

Instructions:
https://github.com/Kulbear/deep-learning-coursera/blob/master/Neural%20Networks%20and%20Deep%20Learning/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step.ipynb 
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v2 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward


plt.rcParams['figure.figsize'] = (5.0, 4.0) 
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


np.random.seed(1)




def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    parameters = {}
    L = len(layer_dims) #number of layer (counting input layer)

    for l in range(1, L):
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * 0.01
    
    return parameters


def linear_forward(A_prev, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Returns:
    cache -- a python dictionary containing "A_prev", "W" and "b" ; stored for computing the backward pass efficiently
    """
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)    
    return Z, cache



def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    Z, linear_cache = linear_forward(A_prev, W, b) 
    if activation == "sigmoid" :
        A, activation_cache = sigmoid(Z)
    elif activation== "relu" :
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)
    return A, cache



def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """
    cache = []
    l = int(len(parameters) / 2)  #number of layers
    A_prev = X
    for i in range(1,l):
        b = parameters["b" + str(i)]
        W = parameters["W" + str(i)]
        A_prev, cache2 = linear_activation_forward(A_prev, W, b, "relu")
        cache.append(cache2)

    b = parameters["b" + str(l)]
    W = parameters["W" + str(l)]
    Y_hat, cache2 = linear_activation_forward(A_prev, W, b, "sigmoid")
    cache.append(cache2)

    return Y_hat, cache



def compute_cost(AL, Y):
    """
    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]
    cost = (-1 / m) * ( np.dot(np.log(AL),Y.T) + np.dot(np.log(1 - AL),(1 - Y).T) )
    cost = np.squeeze(cost)   
    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev ,dW ,db 
    """
    A_prev, W, _ = cache
    m = dZ.shape[1]
    dW = (1 / m) * (np.dot(dZ, A_prev.T))
    db = np.squeeze((1 / m) * (np.sum(dZ, axis=1, keepdims=True)))
    dA_prev = np.dot(W.T, dZ)
    return dA_prev ,dW ,db 



def reluDerivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation : "sigmoid" or "relu"
    
    Returns:
    dA_prev, dW, db 
    """
    linear_cache, activation_cache = cache
    Z = activation_cache

    if activation=="sigmoid":
        dZ = sigmoid_backward(dA, Z)
    elif activation=="relu":
        dZ = dA * reluDerivative(Z)
        # dZ = relu_backward(dA, Z)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db 


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)]
             grads["dW" + str(l)] 
             grads["db" + str(l)] 
    """
    grads = {}
    L = len(caches) # the number of layers
    # Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
    dA = dAL
    print(dAL)
    cache = caches[L-1]
    dA_prev, dW, db = linear_activation_backward(dA, cache, "sigmoid")
    grads["dA" + str(L)] = dA_prev #or dA ??????????????????????????????????????????????????????????
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db
    dA = dA_prev

    for i in reversed(range(1,L)): # l-1 to 1
        cache = caches[i-1]
        dA_prev, dW, db = linear_activation_backward(dA, cache, "relu")
        grads["dA" + str(i)] = dA_prev #or dA ??????????????????????????????????????????????????????????
        grads["dW" + str(i)] = dW
        grads["db" + str(i)] = db
        dA = dA_prev

    return grads


Y_assess, AL, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)
print(grads)
print(AL)



def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)]
                  parameters["b" + str(l)]
    """
    L = len(parameters) // 2 
    # L = len(grads) // 3 
    for i in range(1,L+1):
        parameters["W" + str(i)] = parameters["W" + str(i)] - learning_rate * grads["dW" + str(i)]
        parameters["b" + str(i)] = parameters["b" + str(i)] - learning_rate * grads["db" + str(i)]

    return parameters


