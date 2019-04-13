"""                      Week4 - Programming Assignment 7

In this assignment, you will:

    Implement the neural style transfer algorithm
    Generate novel artistic images using your algorithm

Most of the algorithms you've studied optimize a cost function to get a set of parameter values. 
In Neural Style Transfer, you'll optimize a cost function to get pixel values!

"""
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf


"""
2 - Transfer Learning

Neural Style Transfer (NST) uses a previously trained convolutional network, and builds on top of that. 
The idea of using a network trained on a different task and applying it to a new task is called transfer learning.

Following the original NST paper (https://arxiv.org/abs/1508.06576), we will use the VGG network. 
Specifically, we'll use VGG-19, a 19-layer version of the VGG network. 
This model has already been trained on the very large ImageNet database, and thus has 
learned to recognize a variety of low level features (at the earlier layers) and high level features 
(at the deeper layers). 

"""

# TODO
# model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
# print(model)

# TODO
# model["input"].assign(image)
# sess.run(model["conv4_2"])


content_image = scipy.misc.imread("images/louvre.jpg")
style_image = scipy.misc.imread("images/monet_800600.jpg")
# imshow(style_image)
# imshow(content_image)
# plt.show()



def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.reshape(a_C, [m, n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, [m, n_H * n_W, n_C])
    
    # compute the cost with tensorflow (≈1 line)
    J_content = tf.subtract(a_C_unrolled, a_G_unrolled)
    J_content = tf.reduce_sum(tf.square(J_content))
    J_content = J_content / (4 * n_H * n_W * n_C)
    return J_content


"""
Style matrix
The style matrix is also called a "Gram matrix." In linear algebra, the Gram matrix G of a set of vectors (v1,…,vn)(v1,…,vn)
is the matrix of dot products. In other words, Gij compares how similar vi is to vj: 
If they are highly similar, you would expect them to have a large dot product, and thus for GijGij to be large.

The result is a matrix of dimension (nC,nC) where nC is the number of filters. The value Gij measures how 
similar the activations of filter i are to the activations of filter j.

One important part of the gram matrix is that the diagonal elements such as Gii also measures how active filter i is.
For example, suppose filter i is detecting vertical textures in the image. Then Gii measures how common vertical textures 
are in the image as a whole: If Gii is large, this means that the image has a lot of vertical texture.

By capturing the prevalence of different types of features (Gii), as well as how much different features occur together 
(Gij), the Style matrix G measures the style of an image. 
"""


def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    GA = tf.tensordot(A, tf.transpose(A), axes=1)
    return GA



def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """    
    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images to have them of shape (n_C, n_H*n_W) 
    # problematic hint!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    a_S = tf.reshape(a_S, [n_H * n_W, n_C])
    a_S = tf.transpose(a_S)
    a_G = tf.reshape(a_G, [n_H * n_W, n_C])
    a_G = tf.transpose(a_G)
    # Computing gram_matrices for both images S and G 
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG)))
    J_style_layer = J_style_layer / ((2 * n_C * n_H * n_W)**2)
    
    return J_style_layer




STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


"""
The style of an image can be represented using the Gram matrix of a hidden layer's activations. 
However, we get even better results combining this representation from multiple different layers. 
This is in contrast to the content representation, where usually using just a single hidden layer is sufficient.
"""


# GRADED FUNCTION: total_cost

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """

    J = alpha * J_content + beta * J_style
    
    return J


# TODO
# 4 - Solving the optimization problem