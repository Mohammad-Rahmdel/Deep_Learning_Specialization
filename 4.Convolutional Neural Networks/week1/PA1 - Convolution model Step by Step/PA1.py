"""                      Week1 - Programming Assignment 1
https://qbxthvkswvvyvanaimjbxl.coursera-apps.org/notebooks/week1/Convolution%20model%20-%20Step%20by%20Step%20-%20v2.ipynb
https://github.com/Gurupradeep/deeplearning.ai-Assignments/blob/master/Convolutional_Neural_Networks/Week1/Convolution%2Bmodel%2B-%2BStep%2Bby%2BStep%2B-%2Bv2.ipynb
"""


import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)