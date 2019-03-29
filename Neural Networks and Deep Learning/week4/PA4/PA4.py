"""                      Week4 - Programming Assignment 4

Instructions:
https://github.com/Mohammad-Rahmdel/deep-learning-specialization-coursera/blob/master/01-Neural-Networks-and-Deep-Learning/week4/Programming-Assignments/Deep_Neural_Network_Application_Image_Classification/Deep_Neural_Network_Application_v3.ipynb

https://github.com/Mohammad-Rahmdel/deep-learning-coursera/blob/master/Neural%20Networks%20and%20Deep%20Learning/Deep%20Neural%20Network%20-%20Application.ipynb
"""

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *


plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


np.random.seed(1)



train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# # Example of a picture
# index = 10
# plt.imshow(train_x_orig[index])
# print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
# plt.show()



train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.