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
from dnn_app_utils import *


plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


np.random.seed(1)