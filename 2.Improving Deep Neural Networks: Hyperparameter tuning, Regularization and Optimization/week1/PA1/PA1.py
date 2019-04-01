"""                      Week1 - Programming Assignment 1
https://github.com/andersy005/deep-learning-specialization-coursera/blob/master/02-Improving-Deep-Neural-Networks/week1/Programming-Assignments/Initialization/Initialization.ipynb
https://github.com/Kulbear/deep-learning-coursera/blob/master/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Initialization.ipynb
"""


import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec


plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = load_dataset()
# plt.show()
# plt.close()


def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
    """
    Implements a three-layer neural network: RELU->RELU->SIGMOID.
    
    Arguments:
    X -- input data, of shape (2[x,y coordinates], number of examples)
    Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
    initialization -- flag to choose which initialization to use ("zeros","random" or "he")
    """
        
    grads = {}
    costs = [] # to keep track of the loss
    m = X.shape[1] # number of examples
    layers_dims = [X.shape[0], 10, 5, 1]
    
    # Initialize parameters dictionary.
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation
        a3, cache = forward_propagation(X, parameters)
        # Loss
        cost = compute_loss(a3, Y)
        # Backward propagation.
        grads = backward_propagation(X, Y, cache)
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        # Print the loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
            
    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


def initialize_parameters_zeros(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    parameters = {}
    L = len(layers_dims) - 1
    for i in range(1, L+1):
        parameters["W" + str(i)] = np.zeros((layers_dims[i], layers_dims[i - 1]))
        parameters["b" + str(i)] = np.zeros((layers_dims[i],1))
    
    return parameters


# parameters = model(train_X, train_Y, initialization = "zeros")
# print ("On the train set:")
# predictions_train = predict(train_X, train_Y, parameters)
# print ("On the test set:")
# predictions_test = predict(test_X, test_Y, parameters)
# plt.title("Model with Zeros initialization")
# axes = plt.gca()
# axes.set_xlim([-1.5,1.5])
# axes.set_ylim([-1.5,1.5])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)



def initialize_parameters_random(layers_dims):
    # np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1
    for i in range(1, L+1):
        parameters["W" + str(i)] = np.random.randn(layers_dims[i], layers_dims[i - 1]) * 10
        parameters["b" + str(i)] = np.zeros((layers_dims[i],1))
    
    return parameters


# parameters = model(train_X, train_Y, initialization = "random")
# print ("On the train set:")
# predictions_train = predict(train_X, train_Y, parameters)
# print ("On the test set:")
# predictions_test = predict(test_X, test_Y, parameters)
# plt.title("Model with large random initialization")
# axes = plt.gca()
# axes.set_xlim([-1.5,1.5])
# axes.set_ylim([-1.5,1.5])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
# """
# The cost starts very high. This is because with large random-valued weights, the last activation (sigmoid) outputs results 
# that are very close to 0 or 1 for some examples, and when it gets that example wrong it incurs a very high loss for 
# that example. Indeed, when $\log(a^{[3]}) = \log(0)$, the loss goes to infinity. 
# """


def initialize_parameters_he(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1
    for i in range(1, L+1):
        parameters["W" + str(i)] = np.random.randn(layers_dims[i], layers_dims[i - 1]) * np.sqrt(2/layers_dims[i - 1])
        parameters["b" + str(i)] = np.zeros((layers_dims[i],1))
    
    return parameters



# parameters = model(train_X, train_Y, initialization = "he")
# print ("On the train set:")
# predictions_train = predict(train_X, train_Y, parameters)
# print ("On the test set:")
# predictions_test = predict(test_X, test_Y, parameters)
# plt.title("Model with He initialization")
# axes = plt.gca()
# axes.set_xlim([-1.5,1.5])
# axes.set_ylim([-1.5,1.5])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
"""
**Model**                               	    **Train accuracy**       **Problem/Comment** 
3-layer NN with zeros initialization                   50%             fails to break symmetry
3-layer NN with large random initialization            83%                too large weights 
3-layer NN with He initialization                      99%                recommended method 

-Different initializations lead to different results
-Random initialization is used to break symmetry and make sure different hidden units can learn different things
-Don't intialize to values that are too large
-He initialization works well for networks with ReLU activations.

"""


